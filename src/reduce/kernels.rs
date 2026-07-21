//! Stride-aware reduction / cumulative kernels.
//!
//! Dispatch (same philosophy as ufunc / indexing):
//! 1. suffix reduced axes + C-contiguous slice → contiguous chunk fold
//! 2. single reduced axis → fixed `pos += stride` inner loop
//! 3. otherwise → general multi-index stride walk
//!
//! Contiguity is decided once at the dispatch site via
//! [`Array::as_c_contiguous_slice`]; fast-path kernels never fall back.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::reduce::axis::{normalize_axis, AxisTraversalPlan, ReducePlan};
use crate::shape::c_order_strides;
use crate::stride_iter::{StrideCursor, StrideIter};

enum ReductionPath<'a, T> {
    Contiguous(&'a [T]),
    SingleAxis(usize),
    GeneralStrided,
}

#[inline]
fn reduction_path<'a, T: Scalar>(
    a: &'a Array<T>,
    plan: &ReducePlan,
) -> ReductionPath<'a, T> {
    if plan.is_suffix_reduction(a.ndim()) {
        if let Some(slice) = a.as_c_contiguous_slice() {
            return ReductionPath::Contiguous(slice);
        }
    }
    match plan.single_reduced_axis() {
        Some(axis) => ReductionPath::SingleAxis(axis),
        None => ReductionPath::GeneralStrided,
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
        ReductionPath::Contiguous(slice) => {
            fold_contiguous_chunks(slice, plan, initial, &mut accumulate)
        }
        ReductionPath::SingleAxis(axis) => {
            fold_single_axis(a, plan, axis, initial, &mut accumulate)
        }
        ReductionPath::GeneralStrided => {
            fold_strided_general(a, plan, initial, &mut accumulate)
        }
    }
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

fn fold_single_axis<T, Acc, F>(
    a: &Array<T>,
    plan: &ReducePlan,
    axis: usize,
    initial: Acc,
    accumulate: &mut F,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
{
    let axis_len = a.shape()[axis];
    let axis_stride = a.strides()[axis];
    let (outer_strides, _) = plan.outer_reduced_strides(a.strides());

    let mut out = Vec::with_capacity(plan.outer_n);
    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    for outer_i in 0..plan.outer_n {
        let mut acc = initial;
        if axis_len > 0 {
            let mut buf = outer_cursor.buffer_index(0) as isize;
            for inner_i in 0..axis_len {
                acc = accumulate(acc, a.data[buf as usize]);
                if inner_i + 1 < axis_len {
                    buf += axis_stride;
                }
            }
        }
        out.push(acc);
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

    Array::from_vec(out, &plan.out_shape)
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
    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    let mut inner_cursor = StrideCursor::new(
        &plan.reduced_shape,
        [&reduced_strides],
        [a.offset() as isize],
    );

    for outer_i in 0..plan.outer_n {
        let mut acc = initial;
        if plan.inner_n > 0 {
            inner_cursor.reset([outer_cursor.buffer_index(0) as isize]);
            for inner_i in 0..plan.inner_n {
                acc = accumulate(acc, a.data[inner_cursor.buffer_index(0)]);
                if inner_i + 1 < plan.inner_n {
                    inner_cursor.advance();
                }
            }
        }
        out.push(acc);
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

    Array::from_vec(out, &plan.out_shape)
}

/// Min/max over axes with NaN propagation.
pub(crate) fn reduce_extremum<T, F, N>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    op_name: &str,
    mut is_better: F,
    is_nan: N,
) -> Result<Array<T>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }
    if plan.inner_is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} of empty array / empty axis"
        )));
    }

    match reduction_path(a, &plan) {
        ReductionPath::Contiguous(slice) => {
            extremum_contiguous_chunks(slice, &plan, &mut is_better, &is_nan)
        }
        ReductionPath::SingleAxis(axis) => {
            extremum_single_axis(a, &plan, axis, &mut is_better, &is_nan)
        }
        ReductionPath::GeneralStrided => {
            extremum_strided_general(a, &plan, &mut is_better, &is_nan)
        }
    }
}

fn extremum_from_chunk<T, F, N>(chunk: &[T], is_better: &mut F, is_nan: &N) -> T
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let mut best = chunk[0];
    if is_nan(best) {
        return best;
    }
    for &candidate in &chunk[1..] {
        if is_nan(candidate) {
            return candidate;
        }
        if is_better(candidate, best) {
            best = candidate;
        }
    }
    best
}

fn extremum_contiguous_chunks<T, F, N>(
    slice: &[T],
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
    for chunk in slice.chunks_exact(plan.inner_n) {
        out.push(extremum_from_chunk(chunk, is_better, is_nan));
    }
    Array::from_vec(out, &plan.out_shape)
}

fn extremum_single_axis<T, F, N>(
    a: &Array<T>,
    plan: &ReducePlan,
    axis: usize,
    is_better: &mut F,
    is_nan: &N,
) -> Result<Array<T>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let axis_len = a.shape()[axis];
    let axis_stride = a.strides()[axis];
    let (outer_strides, _) = plan.outer_reduced_strides(a.strides());

    let mut out = Vec::with_capacity(plan.outer_n);
    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    for outer_i in 0..plan.outer_n {
        let mut buf = outer_cursor.buffer_index(0) as isize;
        let mut best = a.data[buf as usize];
        if !is_nan(best) {
            for _ in 1..axis_len {
                buf += axis_stride;
                let candidate = a.data[buf as usize];
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
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

    Array::from_vec(out, &plan.out_shape)
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
    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    let mut inner_cursor = StrideCursor::new(
        &plan.reduced_shape,
        [&reduced_strides],
        [a.offset() as isize],
    );

    for outer_i in 0..plan.outer_n {
        inner_cursor.reset([outer_cursor.buffer_index(0) as isize]);
        let mut best = a.data[inner_cursor.buffer_index(0)];

        if !is_nan(best) {
            for _ in 1..plan.inner_n {
                inner_cursor.advance();
                let candidate = a.data[inner_cursor.buffer_index(0)];
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
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

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
    let mut offsets = StrideIter::new(a.shape(), a.strides(), a.offset());
    let first = offsets.next().expect("non-empty array has a first offset");
    let mut best_linear = 0_i64;
    let mut best_value = a.data[first];

    if is_nan(best_value) {
        return Array::from_vec(vec![0], &[]);
    }

    for (flat, buf_idx) in offsets.enumerate() {
        let linear = flat + 1;
        let candidate = a.data[buf_idx];
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

    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    for outer_i in 0..plan.outer_n {
        let mut buf = outer_cursor.buffer_index(0) as isize;
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
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

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
    let mut out = Vec::with_capacity(n);
    let mut offsets = StrideIter::new(a.shape(), a.strides(), a.offset());
    let first = offsets.next().expect("non-empty array has a first offset");
    let mut acc = to_acc(a.data[first]);
    out.push(acc);

    for buf_idx in offsets {
        acc = accumulate(acc, a.data[buf_idx]);
        out.push(acc);
    }
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

    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&in_outer_strides, &out_outer_strides],
        [a.offset() as isize, 0],
    );
    for outer_i in 0..plan.outer_n {
        let mut in_buf = outer_cursor.buffer_index(0) as isize;
        let mut out_buf = outer_cursor.buffer_index(1) as isize;
        let mut acc = to_acc(a.data[in_buf as usize]);
        out[out_buf as usize] = acc;

        for _ in 1..plan.axis_len {
            in_buf += in_stride;
            out_buf += out_stride;
            acc = accumulate(acc, a.data[in_buf as usize]);
            out[out_buf as usize] = acc;
        }
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

    Array::from_vec(out, a.shape())
}

/// Population variance (Welford) → `f64`.
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
        ReductionPath::Contiguous(slice) => {
            var_contiguous_chunks(slice, &plan, &to_f64)
        }
        ReductionPath::SingleAxis(axis) => {
            var_single_axis(a, &plan, axis, &to_f64)
        }
        ReductionPath::GeneralStrided => var_strided_general(a, &plan, &to_f64),
    }
}

fn welford_chunk<T, C>(chunk: &[T], count: f64, to_f64: &C) -> f64
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let mut sample_count = 0_f64;
    let mut mean = 0_f64;
    let mut m2 = 0_f64;
    for &x in chunk {
        let value = to_f64(x);
        sample_count += 1.0;
        let delta = value - mean;
        mean += delta / sample_count;
        m2 += delta * (value - mean);
    }
    m2 / count
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
        out.push(welford_chunk(chunk, count, to_f64));
    }
    Array::from_vec(out, &plan.out_shape)
}

fn var_single_axis<T, C>(
    a: &Array<T>,
    plan: &ReducePlan,
    axis: usize,
    to_f64: &C,
) -> Result<Array<f64>>
where
    T: Scalar,
    C: Fn(T) -> f64,
{
    let count = plan.inner_n as f64;
    let axis_len = a.shape()[axis];
    let axis_stride = a.strides()[axis];
    let (outer_strides, _) = plan.outer_reduced_strides(a.strides());

    let mut out = Vec::with_capacity(plan.outer_n);
    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    for outer_i in 0..plan.outer_n {
        let mut sample_count = 0_f64;
        let mut mean = 0_f64;
        let mut m2 = 0_f64;
        let mut buf = outer_cursor.buffer_index(0) as isize;
        for inner_i in 0..axis_len {
            let value = to_f64(a.data[buf as usize]);
            sample_count += 1.0;
            let delta = value - mean;
            mean += delta / sample_count;
            m2 += delta * (value - mean);
            if inner_i + 1 < axis_len {
                buf += axis_stride;
            }
        }
        out.push(m2 / count);
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

    Array::from_vec(out, &plan.out_shape)
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
    let mut outer_cursor = StrideCursor::new(
        &plan.outer_shape,
        [&outer_strides],
        [a.offset() as isize],
    );
    let mut inner_cursor = StrideCursor::new(
        &plan.reduced_shape,
        [&reduced_strides],
        [a.offset() as isize],
    );

    for outer_i in 0..plan.outer_n {
        inner_cursor.reset([outer_cursor.buffer_index(0) as isize]);
        let mut sample_count = 0_f64;
        let mut mean = 0_f64;
        let mut m2 = 0_f64;
        for inner_i in 0..plan.inner_n {
            let value = to_f64(a.data[inner_cursor.buffer_index(0)]);
            sample_count += 1.0;
            let delta = value - mean;
            mean += delta / sample_count;
            m2 += delta * (value - mean);
            if inner_i + 1 < plan.inner_n {
                inner_cursor.advance();
            }
        }
        out.push(m2 / count);
        if outer_i + 1 < plan.outer_n {
            outer_cursor.advance();
        }
    }

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
