//! Min, max, and boolean extremum reduction kernels.
//!
//! Each dtype picks suffix chunk folds, prefix row scans, or a general
//! strided path via [`super::reduction_path`]. Floating kernels branch on NaN
//! propagation; ignore mode delegates to
//! [`super::reduce_ignore_with_counts`].

use super::*;
use crate::error::Error;

/// Build a [`ReducePlan`] for extremum reductions.
///
/// Thin wrapper around [`ReducePlan::new`] shared by min/max entry points.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `_op_name` - Operation label (reserved for diagnostics).
///
/// # Returns
///
/// Axis geometry for the extremum kernel dispatch.
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
fn build_extremum_plan<T: Scalar>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    op_name: &'static str,
) -> Result<ReducePlan> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.output_len > 0 && plan.reduction_is_empty() {
        return Err(Error::EmptyReduction { op: op_name });
    }
    Ok(plan)
}

/// Prefix-layout extremum: each output column scans down contiguous rows.
///
/// Used when reduced axes form a leading block and the buffer is
/// C-contiguous. NaN propagation stops updating a slot once NaN is seen.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `plan` - Reduction axis geometry.
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
///
/// # Returns
///
/// An array of extrema shaped like `plan.output_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    if plan.reduction_is_empty() && slice.len() >= plan.output_len {
        return Array::from_vec(
            slice[..plan.output_len].to_vec(),
            &plan.output_shape,
        );
    }
    if slice.len() < plan.output_len {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    let (first_row, remaining) = slice.split_at(plan.output_len);
    let mut out = first_row.to_vec();
    for row in remaining.chunks_exact(plan.output_len) {
        for (best, &candidate) in out.iter_mut().zip(row) {
            if !is_nan(*best)
                && (is_nan(candidate) || is_better(candidate, *best))
            {
                *best = candidate;
            }
        }
    }
    Array::from_vec(out, &plan.output_shape)
}

/// Boolean min/max (`MIN=true` → all-like, `MIN=false` → any-like).
///
/// `reduce_min` on booleans returns false only when every element is false;
/// `reduce_max` returns true when any element is true.
///
/// # Arguments
///
/// * `a` - Boolean input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// A `bool` array shaped like the reduction output.
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
pub(crate) fn reduce_bool_extremum<const MIN: bool>(
    a: &Array<bool>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    let op_name = if MIN { "min" } else { "max" };
    let plan = build_extremum_plan(a, axes, keepdims, op_name)?;
    reduce_bool_with_plan::<MIN>(a, &plan)
}

/// Boolean all/any with correct identities on empty reduced axes.
///
/// Empty reduced slices yield `ALL` (true for `all`, false for `any`).
///
/// # Arguments
///
/// * `a` - Boolean input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// A `bool` array of logical reductions.
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
pub(crate) fn reduce_bool_logical<const ALL: bool>(
    a: &Array<bool>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    reduce_bool_with_plan::<ALL>(a, &plan)
}

/// Shared boolean fold using AND (`ALL=true`) or OR (`ALL=false`).
///
/// Dispatches to suffix chunks, prefix rows, or general strided traversal.
///
/// # Arguments
///
/// * `a` - Boolean input array.
/// * `plan` - Precomputed reduction geometry.
///
/// # Returns
///
/// A `bool` array shaped like `plan.output_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
fn reduce_bool_with_plan<const AND: bool>(
    a: &Array<bool>,
    plan: &ReducePlan,
) -> Result<Array<bool>> {
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    if plan.reduction_len == 0 {
        return Array::from_vec(vec![AND; plan.output_len], &plan.output_shape);
    }
    match reduction_path(a, plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.output_len);
            for chunk in slice.chunks_exact(plan.reduction_len) {
                let mut best = AND;
                if AND {
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
            Array::from_vec(out, &plan.output_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            let mut out = vec![AND; plan.output_len];
            for row in slice.chunks_exact(plan.output_len) {
                if AND {
                    for (best, &candidate) in out.iter_mut().zip(row) {
                        *best &= candidate;
                    }
                } else {
                    for (best, &candidate) in out.iter_mut().zip(row) {
                        *best |= candidate;
                    }
                }
            }
            Array::from_vec(out, &plan.output_shape)
        }
        ReductionPath::GeneralStrided => {
            if AND {
                extremum_strided_general(
                    a,
                    plan,
                    &mut |candidate, best| !candidate && best,
                    &|_| false,
                )
            } else {
                extremum_strided_general(
                    a,
                    plan,
                    &mut |candidate, best| candidate && !best,
                    &|_| false,
                )
            }
        }
    }
}

/// Integer minimum over selected axes.
///
/// Uses eight-lane partial minima on contiguous suffix chunks.
///
/// # Arguments
///
/// * `a` - `i64` input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// An `i64` array of minima.
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
pub(crate) fn reduce_i64_min(
    a: &Array<i64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<i64>> {
    reduce_i64_extremum(a, axes, keepdims, "min", |candidate, best| {
        candidate < best
    })
}

/// Integer maximum over selected axes.
///
/// # Arguments
///
/// * `a` - `i64` input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// An `i64` array of maxima.
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
pub(crate) fn reduce_i64_max(
    a: &Array<i64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<i64>> {
    reduce_i64_extremum(a, axes, keepdims, "max", |candidate, best| {
        candidate > best
    })
}

/// Shared `i64` min/max kernel with layout dispatch.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `op_name` - `"min"` or `"max"` (selects empty-slice sentinel).
/// * `is_better` - Comparison predicate.
///
/// # Returns
///
/// An `i64` array shaped like the reduction output.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or allocation fails.
fn reduce_i64_extremum<F>(
    a: &Array<i64>,
    axes: Option<&[isize]>,
    keepdims: bool,
    op_name: &'static str,
    mut is_better: F,
) -> Result<Array<i64>>
where
    F: FnMut(i64, i64) -> bool,
{
    let plan = build_extremum_plan(a, axes, keepdims, op_name)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.output_len);
            for chunk in slice.chunks_exact(plan.reduction_len) {
                // Eight-lane partial min/max, then merge.
                let mut partials = [chunk[0]; 8];
                let mut blocks = chunk.chunks_exact(8);
                for block in &mut blocks {
                    for lane in 0..8 {
                        if is_better(block[lane], partials[lane]) {
                            partials[lane] = block[lane];
                        }
                    }
                }
                let mut best = partials[0];
                for &candidate in &partials[1..] {
                    if is_better(candidate, best) {
                        best = candidate;
                    }
                }
                for &candidate in blocks.remainder() {
                    if is_better(candidate, best) {
                        best = candidate;
                    }
                }
                out.push(best);
            }
            Array::from_vec(out, &plan.output_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            extremum_prefix_contiguous(slice, &plan, &mut is_better, |_| false)
        }
        ReductionPath::GeneralStrided => {
            extremum_strided_general(a, &plan, &mut is_better, &|_| false)
        }
    }
}

/// Floating min/max with NaN propagation in logical C order.
///
/// Any NaN in a reduced slice poisons that output slot. Suffix chunks use
/// eight-lane partial extrema with a consolidated NaN mask.
///
/// # Arguments
///
/// * `a` - `f64` input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// An `f64` array of extrema (NaN when any input in the slice was NaN).
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
pub(crate) fn reduce_f64_extremum<const MIN: bool>(
    a: &Array<f64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<f64>> {
    let op_name = if MIN { "min" } else { "max" };
    let plan = build_extremum_plan(a, axes, keepdims, op_name)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.output_len);
            for chunk in slice.chunks_exact(plan.reduction_len) {
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
            Array::from_vec(out, &plan.output_shape)
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

/// Floating min/max skipping NaN elements.
///
/// Delegates to [`super::reduce_ignore_with_counts`] with ±infinity identity
/// and NaN sentinels for all-NaN slices.
///
/// # Arguments
///
/// * `a` - `f64` input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// An `f64` array of extrema over finite elements only.
///
/// # Errors
///
/// Returns an error when axis indices are invalid.
pub(crate) fn reduce_f64_extremum_ignore<const MIN: bool>(
    a: &Array<f64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<f64>> {
    let op_name = if MIN { "min" } else { "max" };
    let plan = build_extremum_plan(a, axes, keepdims, op_name)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    let better = |candidate: f64, best: f64| {
        if MIN {
            candidate < best
        } else {
            candidate > best
        }
    };
    let (result, _) = super::fold::reduce_ignore_with_counts(
        a,
        &plan,
        if MIN {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        },
        f64::NAN,
        |best, candidate| {
            if better(candidate, best) {
                candidate
            } else {
                best
            }
        },
        |best, candidate| {
            if better(candidate, best) {
                candidate
            } else {
                best
            }
        },
        f64::is_nan,
    )?;
    Ok(result)
}

/// General-strided min/max over arbitrary layouts.
///
/// Walks outer kept-axis runs and inner [`ReducedAxisRuns`] for each output
/// slot. Short-circuits on the first NaN when propagating.
///
/// # Arguments
///
/// * `a` - Strided input array.
/// * `plan` - Reduction axis geometry.
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
///
/// # Returns
///
/// An array of extrema shaped like `plan.output_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    let mut out = Vec::with_capacity(plan.output_len);
    let (outer_strides, reduced_strides) =
        plan.kept_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
    let reduced = ReducedAxisRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut reduced_cursor = reduced.cursor(a.offset() as isize);

    // Single coalesced run: simple inner loop without run-grid overhead.
    if reduced.run_count == 1 {
        let inner_len = reduced.run_len;
        let inner_stride = reduced.operand_stride;
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
        return Array::from_vec(out, &plan.output_shape);
    }

    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        reduced_cursor.reset([outer_base as isize]);
        let mut best = a.data[reduced_cursor.operand_offset(0)];

        if !is_nan(best) {
            'reduced: for run_i in 0..reduced.run_count {
                let mut pos = reduced_cursor.operand_offset(0) as isize;
                for _ in 0..reduced.run_len {
                    let candidate = a.data[pos as usize];
                    if is_nan(candidate) {
                        best = candidate;
                        break 'reduced;
                    }
                    if is_better(candidate, best) {
                        best = candidate;
                    }
                    pos += reduced.operand_stride;
                }
                if run_i + 1 < reduced.run_count {
                    reduced_cursor.advance();
                }
            }
        }
        out.push(best);
    });

    Array::from_vec(out, &plan.output_shape)
}
