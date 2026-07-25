//! Stride-aware reduction / cumulative kernels.
//!
//! Dispatch (same philosophy as ufunc / indexing):
//! 1. suffix reduced axes + C-contiguous slice → contiguous chunk fold
//! 2. otherwise → general strided walk, itself split into an outer
//!    traversal and a coalesced reduced-axis inner run (see
//!    [`ReducedRuns`]). A single reduced axis coalesces to exactly one run,
//!    so it needs no separate dispatch arm or duplicated kernel.
//!
//! Contiguity is decided once at the dispatch site via
//! [`Array::as_c_contiguous_slice`]; fast-path kernels never fall back.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::reduce::axis::{
    normalize_axis, AxisTraversalPlan, ReducePlan, TraversalSchedule,
};
use crate::run::RunPlan;
use crate::shape::c_order_strides;
use crate::stride_iter::StrideCursor;

enum ReductionPath<'a, T> {
    SuffixContiguous(&'a [T]),
    PrefixContiguous(&'a [T]),
    GeneralStrided,
}

#[inline]
fn reduction_path<'a, T: Scalar>(
    a: &'a Array<T>,
    plan: &ReducePlan,
) -> ReductionPath<'a, T> {
    let contiguous = a.as_c_contiguous_slice();
    match plan.traversal_schedule(a.ndim(), contiguous.is_some()) {
        TraversalSchedule::SuffixContiguous => {
            ReductionPath::SuffixContiguous(contiguous.unwrap())
        }
        TraversalSchedule::PrefixContiguous {
            reduced_len,
            output_len,
        } => {
            debug_assert_eq!(reduced_len, plan.inner_n);
            debug_assert_eq!(output_len, plan.outer_n);
            ReductionPath::PrefixContiguous(contiguous.unwrap())
        }
        TraversalSchedule::GeneralStrided => ReductionPath::GeneralStrided,
    }
}

/// Fold over selected axes starting from `initial`.
pub(crate) fn reduce_fold<T, Acc, F>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    initial: Acc,
    accumulate: F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
{
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    reduce_fold_plan(a, &plan, initial, accumulate)
}

/// Fold using a precomputed [`ReducePlan`] (avoids re-resolving axes).
pub(crate) fn reduce_fold_plan<T, Acc, F>(
    a: &Array<T>,
    plan: &ReducePlan,
    initial: Acc,
    mut accumulate: F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
{
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }

    match reduction_path(a, plan) {
        ReductionPath::SuffixContiguous(slice) => {
            fold_contiguous_chunks(slice, plan, initial, &mut accumulate)
        }
        ReductionPath::PrefixContiguous(slice) => {
            fold_prefix_contiguous(slice, plan, initial, &mut accumulate)
        }
        ReductionPath::GeneralStrided => {
            fold_strided_general(a, plan, initial, &mut accumulate)
        }
    }
}

fn fold_prefix_contiguous<T, Acc, F>(
    slice: &[T],
    plan: &ReducePlan,
    initial: Acc,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
{
    let mut out = vec![initial; plan.outer_n];
    if plan.inner_n > 0 {
        for row in slice.chunks_exact(plan.outer_n) {
            for (acc, &value) in out.iter_mut().zip(row) {
                *acc = accumulate(*acc, value);
            }
        }
    }
    Array::from_vec(out, &plan.out_shape)
}

fn fold_contiguous_chunks<T, Acc, F>(
    slice: &[T],
    plan: &ReducePlan,
    initial: Acc,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
{
    let mut out = Vec::with_capacity(plan.outer_n);
    if plan.inner_n == 0 {
        out.resize(plan.outer_n, initial);
        return Array::from_vec(out, &plan.out_shape);
    }

    for chunk in slice.chunks_exact(plan.inner_n) {
        let mut acc = initial;
        for &x in chunk {
            acc = accumulate(acc, x);
        }
        out.push(acc);
    }
    debug_assert_eq!(out.len(), plan.outer_n);
    Array::from_vec(out, &plan.out_shape)
}

/// Sum selected axes, using independent partial accumulators for contiguous
/// suffix reductions and the generic fold for other layouts.
pub(crate) fn reduce_sum<T, Acc, F, G>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    initial: Acc,
    accumulate: F,
    combine: G,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
    G: FnMut(Acc, Acc) -> Acc,
{
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    reduce_sum_plan(a, &plan, initial, accumulate, combine)
}

/// Sum using a precomputed plan. The eight independent accumulators remove the
/// loop-carried dependency that prevents SIMD/ILP on contiguous chunks.
pub(crate) fn reduce_sum_plan<T, Acc, F, G>(
    a: &Array<T>,
    plan: &ReducePlan,
    initial: Acc,
    mut accumulate: F,
    mut combine: G,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
    G: FnMut(Acc, Acc) -> Acc,
{
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }

    match reduction_path(a, plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.outer_n);
            if plan.inner_n == 0 {
                out.resize(plan.outer_n, initial);
                return Array::from_vec(out, &plan.out_shape);
            }

            for chunk in slice.chunks_exact(plan.inner_n) {
                let mut partials = [initial; 8];
                let mut blocks = chunk.chunks_exact(8);
                for block in &mut blocks {
                    for lane in 0..8 {
                        partials[lane] =
                            accumulate(partials[lane], block[lane]);
                    }
                }

                let mut acc = initial;
                for partial in partials {
                    acc = combine(acc, partial);
                }
                for &x in blocks.remainder() {
                    acc = accumulate(acc, x);
                }
                out.push(acc);
            }
            debug_assert_eq!(out.len(), plan.outer_n);
            Array::from_vec(out, &plan.out_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            fold_prefix_contiguous(slice, plan, initial, &mut accumulate)
        }
        ReductionPath::GeneralStrided => {
            fold_strided_general(a, plan, initial, &mut accumulate)
        }
    }
}

fn fold_strided_general<T, Acc, F>(
    a: &Array<T>,
    plan: &ReducePlan,
    initial: Acc,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
{
    let mut out = Vec::with_capacity(plan.outer_n);
    let (outer_strides, reduced_strides) =
        plan.outer_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.outer_shape, [&outer_strides]);
    let reduced = ReducedRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut reduced_cursor = reduced.cursor(a.offset() as isize);

    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        let mut acc = initial;
        if plan.inner_n > 0 {
            reduced_cursor.reset([outer_base as isize]);
            for run_i in 0..reduced.outer_n {
                let mut pos = reduced_cursor.buffer_index(0) as isize;
                for _ in 0..reduced.inner_len {
                    acc = accumulate(acc, a.data[pos as usize]);
                    pos += reduced.inner_stride;
                }
                if run_i + 1 < reduced.outer_n {
                    reduced_cursor.advance();
                }
            }
        }
        out.push(acc);
    });

    Array::from_vec(out, &plan.out_shape)
}

/// Coalesced reduced-axis layout shared by every general-strided reduction
/// kernel below.
///
/// Built once per call from `plan.reduced_shape`/reduced strides; each outer
/// (output) position then resets a cursor over [`Self::outer_shape`] to its
/// own base offset and walks [`Self::inner_len`] elements at a fixed
/// [`Self::inner_stride`] — replacing the old per-reduced-element
/// N-dimensional carry with an outer traversal over far fewer axes (often
/// exactly one) plus a linear inner run.
struct ReducedRuns {
    plan: RunPlan<1>,
    outer_n: usize,
    inner_len: usize,
    inner_stride: isize,
}

impl ReducedRuns {
    fn new(reduced_shape: &[usize], reduced_strides: &[isize]) -> Self {
        let plan = RunPlan::new(reduced_shape, [reduced_strides]);
        let outer_n = plan.outer_len();
        let inner_len = plan.inner_len();
        let inner_stride = plan.inner_stride(0);
        Self {
            plan,
            outer_n,
            inner_len,
            inner_stride,
        }
    }

    /// A cursor over [`Self::outer_shape`], ready to be `reset()` to each
    /// output position's base offset.
    fn cursor(&self, offset: isize) -> StrideCursor<'_, 1> {
        self.plan.cursor([offset])
    }
}

fn extremum_plan<T: Scalar>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    op_name: &str,
) -> Result<ReducePlan> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.outer_n > 0 && plan.inner_is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} of empty array / empty axis"
        )));
    }
    Ok(plan)
}

fn extremum_prefix_contiguous<T, F, N>(
    slice: &[T],
    plan: &ReducePlan,
    mut is_better: F,
    is_nan: N,
) -> Result<Array<T>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let (first_row, remaining) = slice.split_at(plan.outer_n);
    let mut out = first_row.to_vec();
    for row in remaining.chunks_exact(plan.outer_n) {
        for (best, &candidate) in out.iter_mut().zip(row) {
            if !is_nan(*best)
                && (is_nan(candidate) || is_better(candidate, *best))
            {
                *best = candidate;
            }
        }
    }
    Array::from_vec(out, &plan.out_shape)
}

/// Boolean min/max is all/any respectively. The contiguous kernel uses
/// bitwise accumulation so there is no comparison or generic dispatch in the
/// hot loop.
pub(crate) fn reduce_bool_extremum<const MIN: bool>(
    a: &Array<bool>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    let op_name = if MIN { "min" } else { "max" };
    let plan = extremum_plan(a, axes, keepdims, op_name)?;
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.outer_n);
            for chunk in slice.chunks_exact(plan.inner_n) {
                let mut best = MIN;
                if MIN {
                    for &candidate in chunk {
                        best &= candidate;
                    }
                } else {
                    for &candidate in chunk {
                        best |= candidate;
                    }
                }
                out.push(best);
            }
            Array::from_vec(out, &plan.out_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            let mut out = vec![MIN; plan.outer_n];
            for row in slice.chunks_exact(plan.outer_n) {
                if MIN {
                    for (best, &candidate) in out.iter_mut().zip(row) {
                        *best &= candidate;
                    }
                } else {
                    for (best, &candidate) in out.iter_mut().zip(row) {
                        *best |= candidate;
                    }
                }
            }
            Array::from_vec(out, &plan.out_shape)
        }
        ReductionPath::GeneralStrided => {
            if MIN {
                extremum_strided_general(
                    a,
                    &plan,
                    &mut |candidate, best| !candidate && best,
                    &|_| false,
                )
            } else {
                extremum_strided_general(
                    a,
                    &plan,
                    &mut |candidate, best| candidate && !best,
                    &|_| false,
                )
            }
        }
    }
}

pub(crate) fn reduce_i64_min(
    a: &Array<i64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<i64>> {
    reduce_i64_extremum(a, axes, keepdims, "min", |candidate, best| {
        candidate < best
    })
}

pub(crate) fn reduce_i64_max(
    a: &Array<i64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<i64>> {
    reduce_i64_extremum(a, axes, keepdims, "max", |candidate, best| {
        candidate > best
    })
}

/// Integer min/max uses a concrete `Ord` kernel with no NaN branch.
fn reduce_i64_extremum<F>(
    a: &Array<i64>,
    axes: Option<&[isize]>,
    keepdims: bool,
    op_name: &str,
    mut is_better: F,
) -> Result<Array<i64>>
where
    F: FnMut(i64, i64) -> bool,
{
    let plan = extremum_plan(a, axes, keepdims, op_name)?;
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.outer_n);
            for chunk in slice.chunks_exact(plan.inner_n) {
                let mut best = chunk[0];
                for &candidate in &chunk[1..] {
                    if is_better(candidate, best) {
                        best = candidate;
                    }
                }
                out.push(best);
            }
            Array::from_vec(out, &plan.out_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            extremum_prefix_contiguous(slice, &plan, &mut is_better, |_| false)
        }
        ReductionPath::GeneralStrided => {
            extremum_strided_general(a, &plan, &mut is_better, &|_| false)
        }
    }
}

/// Floating min/max keeps NaN propagation while using independent comparison
/// chains for NaN-free contiguous chunks. NaNs are still observed in logical
/// C order, so the first NaN value is propagated as before.
pub(crate) fn reduce_f64_extremum<const MIN: bool>(
    a: &Array<f64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<f64>> {
    let op_name = if MIN { "min" } else { "max" };
    let plan = extremum_plan(a, axes, keepdims, op_name)?;
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.outer_n);
            for chunk in slice.chunks_exact(plan.inner_n) {
                if chunk[0].is_nan() {
                    out.push(f64::NAN);
                    continue;
                }
                let identity = if MIN {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
                let mut partials = [identity; 8];
                let mut nan_masks = [false; 8];
                let mut blocks = chunk.chunks_exact(8);
                for block in &mut blocks {
                    for lane in 0..8 {
                        let candidate = block[lane];
                        nan_masks[lane] |= candidate.is_nan();
                        if if MIN {
                            candidate < partials[lane]
                        } else {
                            candidate > partials[lane]
                        } {
                            partials[lane] = candidate;
                        }
                    }
                }

                let mut best = identity;
                for candidate in partials {
                    if if MIN {
                        candidate < best
                    } else {
                        candidate > best
                    } {
                        best = candidate;
                    }
                }
                let mut has_nan =
                    nan_masks.into_iter().fold(false, |acc, flag| acc | flag);
                for &candidate in blocks.remainder() {
                    has_nan |= candidate.is_nan();
                    if if MIN {
                        candidate < best
                    } else {
                        candidate > best
                    } {
                        best = candidate;
                    }
                }
                out.push(if has_nan { f64::NAN } else { best });
            }
            Array::from_vec(out, &plan.out_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            if MIN {
                extremum_prefix_contiguous(
                    slice,
                    &plan,
                    |candidate, best| candidate < best,
                    f64::is_nan,
                )
            } else {
                extremum_prefix_contiguous(
                    slice,
                    &plan,
                    |candidate, best| candidate > best,
                    f64::is_nan,
                )
            }
        }
        ReductionPath::GeneralStrided => {
            if MIN {
                extremum_strided_general(
                    a,
                    &plan,
                    &mut |candidate, best| candidate < best,
                    &f64::is_nan,
                )
            } else {
                extremum_strided_general(
                    a,
                    &plan,
                    &mut |candidate, best| candidate > best,
                    &f64::is_nan,
                )
            }
        }
    }
}

fn extremum_strided_general<T, F, N>(
    a: &Array<T>,
    plan: &ReducePlan,
    is_better: &mut F,
    is_nan: &N,
) -> Result<Array<T>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let mut out = Vec::with_capacity(plan.outer_n);
    let (outer_strides, reduced_strides) =
        plan.outer_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.outer_shape, [&outer_strides]);
    let reduced = ReducedRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut reduced_cursor = reduced.cursor(a.offset() as isize);

    // The overwhelmingly common case is a fully coalesced reduced layout
    // (single axis, or several axes that merge into one linear run). A flat
    // `pos += stride` loop lets the compiler keep `best`/`pos` in registers
    // and avoids wrapping every element in an extra run-counting loop, which
    // otherwise measurably hurts this branch-heavy (NaN-checking) kernel.
    if reduced.outer_n == 1 {
        let inner_len = reduced.inner_len;
        let inner_stride = reduced.inner_stride;
        outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
            let mut pos = outer_base as isize;
            let mut best = a.data[pos as usize];
            if !is_nan(best) {
                for _ in 1..inner_len {
                    pos += inner_stride;
                    let candidate = a.data[pos as usize];
                    if is_nan(candidate) {
                        best = candidate;
                        break;
                    }
                    if is_better(candidate, best) {
                        best = candidate;
                    }
                }
            }
            out.push(best);
        });
        return Array::from_vec(out, &plan.out_shape);
    }

    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        reduced_cursor.reset([outer_base as isize]);
        let mut best = a.data[reduced_cursor.buffer_index(0)];

        if !is_nan(best) {
            'reduced: for run_i in 0..reduced.outer_n {
                let mut pos = reduced_cursor.buffer_index(0) as isize;
                for _ in 0..reduced.inner_len {
                    let candidate = a.data[pos as usize];
                    if is_nan(candidate) {
                        best = candidate;
                        break 'reduced;
                    }
                    if is_better(candidate, best) {
                        best = candidate;
                    }
                    pos += reduced.inner_stride;
                }
                if run_i + 1 < reduced.outer_n {
                    reduced_cursor.advance();
                }
            }
        }
        out.push(best);
    });

    Array::from_vec(out, &plan.out_shape)
}

/// Flat argmin/argmax → 0-D `i64` linear index.
pub(crate) fn arg_extremum_flat<T, F, N>(
    a: &Array<T>,
    op_name: &str,
    mut is_better: F,
    is_nan: N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let n = a.size();
    if n == 0 {
        return Err(Error::InvalidArgument(format!(
            "{op_name} of empty array"
        )));
    }

    if let Some(slice) = a.as_c_contiguous_slice() {
        return arg_extremum_flat_contiguous(slice, &mut is_better, &is_nan);
    }
    arg_extremum_flat_strided(a, &mut is_better, &is_nan)
}

fn arg_extremum_flat_contiguous<T, F, N>(
    slice: &[T],
    is_better: &mut F,
    is_nan: &N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let mut best_linear = 0_i64;
    let mut best_value = slice[0];
    if is_nan(best_value) {
        return Array::from_vec(vec![0], &[]);
    }
    for (linear, &candidate) in slice.iter().enumerate().skip(1) {
        if is_nan(candidate) {
            return Array::from_vec(vec![linear as i64], &[]);
        }
        if is_better(candidate, best_value) {
            best_value = candidate;
            best_linear = linear as i64;
        }
    }
    Array::from_vec(vec![best_linear], &[])
}

fn arg_extremum_flat_strided<T, F, N>(
    a: &Array<T>,
    is_better: &mut F,
    is_nan: &N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    // Flat C-order walk over a non-contiguous buffer: coalesce the full
    // shape into outer runs + a fixed-stride inner loop, then keep a single
    // linear counter across runs (the argmin/argmax result is a flat index).
    let run_plan = RunPlan::new(a.shape(), [a.strides()]);
    debug_assert!(run_plan.inner_len() > 0 && run_plan.outer_len() > 0);
    let mut best_value = a.data[a.offset()];
    if is_nan(best_value) {
        return Array::from_vec(vec![0], &[]);
    }
    let mut best_linear = 0_i64;
    let mut linear = 0_i64;
    let mut skip_first = true;

    let nan_index = run_plan.try_for_each(
        [a.offset() as isize],
        |run| -> std::result::Result<(), i64> {
            let mut pos = run.bases[0] as isize;
            for _ in 0..run.len {
                if skip_first {
                    skip_first = false;
                } else {
                    let candidate = a.data[pos as usize];
                    if is_nan(candidate) {
                        return Err(linear);
                    }
                    if is_better(candidate, best_value) {
                        best_value = candidate;
                        best_linear = linear;
                    }
                }
                linear += 1;
                pos += run.strides[0];
            }
            Ok(())
        },
    );
    if let Err(index) = nan_index {
        return Array::from_vec(vec![index], &[]);
    }

    Array::from_vec(vec![best_linear], &[])
}

/// Argmin/argmax along a single axis.
pub(crate) fn arg_extremum_axis<T, F, N>(
    a: &Array<T>,
    axis: isize,
    mut is_better: F,
    is_nan: N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let axis = normalize_axis(axis, a.ndim())?;
    let plan = AxisTraversalPlan::new(a.shape(), axis);
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.outer_shape);
    }
    if plan.axis_len == 0 {
        return Err(Error::InvalidArgument(format!(
            "cannot reduce over axis {axis} with size 0"
        )));
    }

    // Contiguous last-axis: each outer row is one contiguous chunk.
    if plan.is_last_axis(a.ndim()) {
        if let Some(slice) = a.as_c_contiguous_slice() {
            return arg_extremum_axis_contiguous(
                slice,
                &plan,
                &mut is_better,
                &is_nan,
            );
        }
    }

    arg_extremum_axis_strided(a, &plan, &mut is_better, &is_nan)
}

fn arg_extremum_axis_contiguous<T, F, N>(
    slice: &[T],
    plan: &AxisTraversalPlan,
    is_better: &mut F,
    is_nan: &N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let mut out = Vec::with_capacity(plan.outer_n);
    for chunk in slice.chunks_exact(plan.axis_len) {
        let mut best_axis_index = 0_i64;
        let mut best_value = chunk[0];
        if !is_nan(best_value) {
            for (axis_index, &candidate) in chunk.iter().enumerate().skip(1) {
                if is_nan(candidate) {
                    best_axis_index = axis_index as i64;
                    break;
                }
                if is_better(candidate, best_value) {
                    best_value = candidate;
                    best_axis_index = axis_index as i64;
                }
            }
        }
        out.push(best_axis_index);
    }
    Array::from_vec(out, &plan.outer_shape)
}

fn arg_extremum_axis_strided<T, F, N>(
    a: &Array<T>,
    plan: &AxisTraversalPlan,
    is_better: &mut F,
    is_nan: &N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let mut out = Vec::with_capacity(plan.outer_n);
    let axis_stride = a.strides()[plan.axis];
    let outer_strides = plan.outer_strides(a.strides());

    let outer_runs = RunPlan::new(&plan.outer_shape, [&outer_strides]);
    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        let mut buf = outer_base as isize;
        let mut best_value = a.data[buf as usize];
        let mut best_axis_index = 0_i64;

        if !is_nan(best_value) {
            for axis_index in 1..plan.axis_len {
                buf += axis_stride;
                let candidate = a.data[buf as usize];
                if is_nan(candidate) {
                    best_axis_index = axis_index as i64;
                    break;
                }
                if is_better(candidate, best_value) {
                    best_value = candidate;
                    best_axis_index = axis_index as i64;
                }
            }
        }
        out.push(best_axis_index);
    });

    Array::from_vec(out, &plan.outer_shape)
}

/// Cumulative scan along one axis (or flat if `axis is None`).
pub(crate) fn cumulate<T, Acc, C, F>(
    a: &Array<T>,
    axis: Option<isize>,
    to_acc: C,
    accumulate: F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc + Copy,
    F: FnMut(Acc, T) -> Acc,
{
    match axis {
        None => cumulate_flat(a, to_acc, accumulate),
        Some(ax) => cumulate_axis(a, ax, to_acc, accumulate),
    }
}

fn cumulate_flat<T, Acc, C, F>(
    a: &Array<T>,
    to_acc: C,
    mut accumulate: F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc,
    F: FnMut(Acc, T) -> Acc,
{
    let n = a.size();
    if n == 0 {
        return Array::from_vec(Vec::new(), &[0]);
    }

    if let Some(slice) = a.as_c_contiguous_slice() {
        return cumulate_flat_contiguous(slice, to_acc, &mut accumulate);
    }
    cumulate_flat_strided(a, n, to_acc, &mut accumulate)
}

fn cumulate_flat_contiguous<T, Acc, C, F>(
    slice: &[T],
    to_acc: C,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc,
    F: FnMut(Acc, T) -> Acc,
{
    let n = slice.len();
    let mut out = Vec::with_capacity(n);
    let mut acc = to_acc(slice[0]);
    out.push(acc);
    for &x in &slice[1..] {
        acc = accumulate(acc, x);
        out.push(acc);
    }
    Array::from_vec(out, &[n])
}

fn cumulate_flat_strided<T, Acc, C, F>(
    a: &Array<T>,
    n: usize,
    to_acc: C,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc,
    F: FnMut(Acc, T) -> Acc,
{
    // Flat cumulative scan over a non-contiguous buffer. Same coalesced
    // outer/inner decomposition as [`arg_extremum_flat_strided`]; the running
    // accumulator is carried across every run so visitation stays C-order.
    let mut out = Vec::with_capacity(n);
    let run_plan = RunPlan::new(a.shape(), [a.strides()]);
    debug_assert!(run_plan.inner_len() > 0 && run_plan.outer_len() > 0);
    let mut acc = to_acc(a.data[a.offset()]);
    out.push(acc);
    let mut skip_first = true;

    run_plan.for_each([a.offset() as isize], |run| {
        let mut pos = run.bases[0] as isize;
        for _ in 0..run.len {
            if skip_first {
                skip_first = false;
            } else {
                acc = accumulate(acc, a.data[pos as usize]);
                out.push(acc);
            }
            pos += run.strides[0];
        }
    });

    debug_assert_eq!(out.len(), n);
    Array::from_vec(out, &[n])
}

fn cumulate_axis<T, Acc, C, F>(
    a: &Array<T>,
    axis: isize,
    to_acc: C,
    mut accumulate: F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc + Copy,
    F: FnMut(Acc, T) -> Acc,
{
    let axis = normalize_axis(axis, a.ndim())?;
    let plan = AxisTraversalPlan::new(a.shape(), axis);
    let n = a.size();
    if n == 0 {
        return Array::from_vec(Vec::new(), a.shape());
    }

    // Contiguous last axis: scan each contiguous row in place order.
    if plan.is_last_axis(a.ndim()) {
        if let Some(slice) = a.as_c_contiguous_slice() {
            return cumulate_axis_contiguous(
                slice,
                &plan,
                a.shape(),
                to_acc,
                &mut accumulate,
            );
        }
    }

    cumulate_axis_strided(a, &plan, to_acc, accumulate)
}

fn cumulate_axis_contiguous<T, Acc, C, F>(
    slice: &[T],
    plan: &AxisTraversalPlan,
    shape: &[usize],
    to_acc: C,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc,
    F: FnMut(Acc, T) -> Acc,
{
    let mut out = Vec::with_capacity(slice.len());
    for chunk in slice.chunks_exact(plan.axis_len) {
        let mut acc = to_acc(chunk[0]);
        out.push(acc);
        for &x in &chunk[1..] {
            acc = accumulate(acc, x);
            out.push(acc);
        }
    }
    Array::from_vec(out, shape)
}

fn cumulate_axis_strided<T, Acc, C, F>(
    a: &Array<T>,
    plan: &AxisTraversalPlan,
    to_acc: C,
    mut accumulate: F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc + Copy,
    F: FnMut(Acc, T) -> Acc,
{
    let n = a.size();
    let in_stride = a.strides()[plan.axis];
    let out_strides = c_order_strides(a.shape());
    let out_stride = out_strides[plan.axis];
    let in_outer_strides = plan.outer_strides(a.strides());
    let out_outer_strides = plan.outer_strides(&out_strides);

    let mut out = Vec::with_capacity(n);
    out.resize(n, to_acc(a.data[a.offset]));

    let outer_runs = RunPlan::new(
        &plan.outer_shape,
        [&in_outer_strides, &out_outer_strides],
    );
    outer_runs.for_each_element(
        [a.offset() as isize, 0],
        |[input_base, output_base]| {
            let mut in_buf = input_base as isize;
            let mut out_buf = output_base as isize;
            let mut acc = to_acc(a.data[in_buf as usize]);
            out[out_buf as usize] = acc;

            for _ in 1..plan.axis_len {
                in_buf += in_stride;
                out_buf += out_stride;
                acc = accumulate(acc, a.data[in_buf as usize]);
                out[out_buf as usize] = acc;
            }
        },
    );

    Array::from_vec(out, a.shape())
}

/// Population variance (two-pass mean, then squared deviations) → `f64`.
pub(crate) fn reduce_var<T, C>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    to_f64: C,
) -> Result<Array<f64>>
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }
    if plan.inner_is_empty() {
        return Err(Error::InvalidArgument(
            "var of empty array / empty axis".into(),
        ));
    }

    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            var_contiguous_chunks(slice, &plan, &to_f64)
        }
        ReductionPath::PrefixContiguous(slice) => {
            var_prefix_contiguous(slice, &plan, &to_f64)
        }
        ReductionPath::GeneralStrided => var_strided_general(a, &plan, &to_f64),
    }
}

#[inline]
fn merge_eight_f64(partials: [f64; 8]) -> f64 {
    let pair_0 = partials[0] + partials[1];
    let pair_1 = partials[2] + partials[3];
    let pair_2 = partials[4] + partials[5];
    let pair_3 = partials[6] + partials[7];
    (pair_0 + pair_1) + (pair_2 + pair_3)
}

#[inline]
fn converted_sum_chunk<T, C>(chunk: &[T], to_f64: &C) -> f64
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let mut partials = [0.0; 8];
    let mut blocks = chunk.chunks_exact(8);
    for block in &mut blocks {
        for lane in 0..8 {
            partials[lane] += to_f64(block[lane]);
        }
    }
    let mut sum = merge_eight_f64(partials);
    for &x in blocks.remainder() {
        sum += to_f64(x);
    }
    sum
}

#[inline]
fn squared_deviation_sum_chunk<T, C>(chunk: &[T], mean: f64, to_f64: &C) -> f64
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let mut partials = [0.0; 8];
    let mut blocks = chunk.chunks_exact(8);
    for block in &mut blocks {
        for lane in 0..8 {
            let delta = to_f64(block[lane]) - mean;
            partials[lane] += delta * delta;
        }
    }
    let mut sum = merge_eight_f64(partials);
    for &x in blocks.remainder() {
        let delta = to_f64(x) - mean;
        sum += delta * delta;
    }
    sum
}

#[inline]
fn variance_chunk<T, C>(chunk: &[T], count: f64, to_f64: &C) -> f64
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let mean = converted_sum_chunk(chunk, to_f64) / count;
    squared_deviation_sum_chunk(chunk, mean, to_f64) / count
}

fn var_contiguous_chunks<T, C>(
    slice: &[T],
    plan: &ReducePlan,
    to_f64: &C,
) -> Result<Array<f64>>
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let count = plan.inner_n as f64;
    let mut out = Vec::with_capacity(plan.outer_n);
    for chunk in slice.chunks_exact(plan.inner_n) {
        out.push(variance_chunk(chunk, count, to_f64));
    }
    Array::from_vec(out, &plan.out_shape)
}

fn var_prefix_contiguous<T, C>(
    slice: &[T],
    plan: &ReducePlan,
    to_f64: &C,
) -> Result<Array<f64>>
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let count = plan.inner_n as f64;
    let mut means = vec![0.0; plan.outer_n];
    for row in slice.chunks_exact(plan.outer_n) {
        for (sum, &value) in means.iter_mut().zip(row) {
            *sum += to_f64(value);
        }
    }
    for mean in &mut means {
        *mean /= count;
    }

    let mut variances = vec![0.0; plan.outer_n];
    for row in slice.chunks_exact(plan.outer_n) {
        for ((sum, &mean), &value) in variances.iter_mut().zip(&means).zip(row)
        {
            let delta = to_f64(value) - mean;
            *sum += delta * delta;
        }
    }
    for variance in &mut variances {
        *variance /= count;
    }
    Array::from_vec(variances, &plan.out_shape)
}

fn var_strided_general<T, C>(
    a: &Array<T>,
    plan: &ReducePlan,
    to_f64: &C,
) -> Result<Array<f64>>
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let count = plan.inner_n as f64;
    let mut out = Vec::with_capacity(plan.outer_n);
    let (outer_strides, reduced_strides) =
        plan.outer_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.outer_shape, [&outer_strides]);
    let reduced = ReducedRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut reduced_cursor = reduced.cursor(a.offset() as isize);

    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        let base = outer_base as isize;

        reduced_cursor.reset([base]);
        let mut sum = 0.0;
        for run_i in 0..reduced.outer_n {
            let mut pos = reduced_cursor.buffer_index(0) as isize;
            for _ in 0..reduced.inner_len {
                sum += to_f64(a.data[pos as usize]);
                pos += reduced.inner_stride;
            }
            if run_i + 1 < reduced.outer_n {
                reduced_cursor.advance();
            }
        }
        let mean = sum / count;

        reduced_cursor.reset([base]);
        let mut squared_deviation_sum = 0.0;
        for run_i in 0..reduced.outer_n {
            let mut pos = reduced_cursor.buffer_index(0) as isize;
            for _ in 0..reduced.inner_len {
                let delta = to_f64(a.data[pos as usize]) - mean;
                squared_deviation_sum += delta * delta;
                pos += reduced.inner_stride;
            }
            if run_i + 1 < reduced.outer_n {
                reduced_cursor.advance();
            }
        }
        out.push(squared_deviation_sum / count);
    });

    Array::from_vec(out, &plan.out_shape)
}

/// Map every logical element of an **owned, C-contiguous, offset-0** array.
///
/// Intended only for freshly allocated reduction results (mean / std).
pub(crate) fn map_owned_contiguous<T, F>(mut a: Array<T>, mut f: F) -> Array<T>
where
    T: Scalar,
    F: FnMut(T) -> T,
{
    debug_assert_eq!(a.offset(), 0, "map_owned_contiguous requires offset 0");
    debug_assert!(
        a.is_c_contiguous(),
        "map_owned_contiguous requires C-contiguous layout"
    );
    debug_assert!(
        a.is_writable(),
        "map_owned_contiguous requires a writable array"
    );
    debug_assert_eq!(
        a.size(),
        a.data.len(),
        "map_owned_contiguous requires the buffer to match the logical size"
    );

    let data = Arc::make_mut(&mut a.data);
    for x in data.iter_mut() {
        *x = f(*x);
    }
    a
}
