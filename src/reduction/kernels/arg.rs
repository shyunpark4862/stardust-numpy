//! Argmin and argmax kernels with NaN propagate and ignore variants.
//!
//! Flat reductions maintain a single linear index in C order. Axis
//! reductions emit one index per outer position. Contiguous last-axis inputs
//! scan row chunks directly; strided inputs use outer runs plus an inner
//! axis stride loop.
//!
//! Callers may supply an `is_terminal` predicate so a winning value ends the
//! scan early. Boolean reductions use this for NumPy-compatible behavior:
//! `argmin` stops at the first `false`, `argmax` at the first `true`.

use super::*;
use crate::error::Error;

/// Flat argmin/argmax with NaN propagation in logical C order.
///
/// Scans the entire array in C-order linear index. The first NaN encountered
/// wins immediately (NumPy propagate semantics). When `is_terminal` holds for
/// the current best value, the scan stops because no later element can win.
/// Dispatches to a contiguous slice fast path when layout allows.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `_op_name` - Operation label (reserved for diagnostics).
/// * `is_better` - Comparison: `true` when `candidate` beats `best`.
/// * `is_nan` - Predicate marking NaN values that poison the result.
/// * `is_terminal` - Predicate marking a value that cannot be improved.
///
/// # Returns
///
/// A 0-D `i64` array holding the winning linear index.
///
/// # Errors
///
/// Returns an error when allocation fails.
pub(crate) fn arg_extremum_flat<T, F, N, E>(
    a: &Array<T>,
    op_name: &'static str,
    mut is_better: F,
    is_nan: N,
    is_terminal: E,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
    E: Fn(T) -> bool,
{
    let n = a.size();
    if n == 0 {
        return Err(Error::EmptyReduction { op: op_name });
    }

    if let Some(slice) = a.as_c_contiguous_slice() {
        return arg_extremum_flat_contiguous(
            slice,
            &mut is_better,
            &is_nan,
            &is_terminal,
        );
    }
    arg_extremum_flat_strided(a, &mut is_better, &is_nan, &is_terminal)
}

/// Contiguous flat arg-extremum with NaN propagation.
///
/// Linear scan over a dense slice. Stops at the first NaN and returns its
/// index, or stops early when `is_terminal` marks the current best value.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
/// * `is_terminal` - Early-exit predicate for values that cannot be improved.
///
/// # Returns
///
/// A 0-D `i64` array with the extremum or first-NaN index.
///
/// # Errors
///
/// Returns an error when allocation fails.
fn arg_extremum_flat_contiguous<T, F, N, E>(
    slice: &[T],
    is_better: &mut F,
    is_nan: &N,
    is_terminal: &E,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
    E: Fn(T) -> bool,
{
    let mut best_linear = 0_i64;
    let mut best_value = slice[0];
    // First NaN wins immediately (NumPy propagate semantics).
    if is_nan(best_value) {
        return Array::from_vec(vec![0], &[]);
    }
    // Boolean argmin/argmax use this to stop once the winning value is found.
    if is_terminal(best_value) {
        return Array::from_vec(vec![0], &[]);
    }
    for (linear, &candidate) in slice.iter().enumerate().skip(1) {
        if is_nan(candidate) {
            return Array::from_vec(vec![linear as i64], &[]);
        }
        if is_better(candidate, best_value) {
            best_value = candidate;
            best_linear = linear as i64;
            if is_terminal(candidate) {
                break;
            }
        }
    }
    Array::from_vec(vec![best_linear], &[])
}

/// Strided flat arg-extremum with NaN propagation.
///
/// Coalesces the full shape into outer runs and walks elements in C-order
/// linear index without requiring a contiguous buffer. Honors `is_terminal`
/// the same way as the contiguous flat path.
///
/// # Arguments
///
/// * `a` - Strided input array.
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
/// * `is_terminal` - Early-exit predicate for values that cannot be improved.
///
/// # Returns
///
/// A 0-D `i64` array with the extremum or first-NaN index.
///
/// # Errors
///
/// Returns an error when allocation fails.
fn arg_extremum_flat_strided<T, F, N, E>(
    a: &Array<T>,
    is_better: &mut F,
    is_nan: &N,
    is_terminal: &E,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
    E: Fn(T) -> bool,
{
    // Coalesce the full shape into outer runs + inner stride steps.
    let run_plan = RunPlan::new(a.shape(), [a.strides()]);
    let mut best_value = a.data[a.offset()];
    if is_nan(best_value) {
        return Array::from_vec(vec![0], &[]);
    }
    if is_terminal(best_value) {
        return Array::from_vec(vec![0], &[]);
    }
    let mut best_linear = 0_i64;
    let mut linear = 0_i64;
    let mut skip_first = true;

    let early_index = run_plan.try_for_each(
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
                        if is_terminal(candidate) {
                            return Err(linear);
                        }
                    }
                }
                linear += 1;
                pos += run.strides[0];
            }
            Ok(())
        },
    );
    if let Err(index) = early_index {
        return Array::from_vec(vec![index], &[]);
    }

    Array::from_vec(vec![best_linear], &[])
}

/// Argmin/argmax along one axis with NaN propagation.
///
/// Emits one axis-relative index per outer position. Uses a contiguous row
/// fast path when the scanned axis is last and the buffer is C-contiguous.
/// Each row scan may stop early when `is_terminal` becomes true.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to find extrema (may be negative).
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
/// * `is_terminal` - Predicate marking a value that cannot be improved.
///
/// # Returns
///
/// An `i64` array with shape equal to all dimensions except `axis`.
///
/// # Errors
///
/// Returns an error when the axis is out of range or allocation fails.
pub(crate) fn arg_extremum_axis<T, F, N, E>(
    a: &Array<T>,
    axis: isize,
    op_name: &'static str,
    mut is_better: F,
    is_nan: N,
    is_terminal: E,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
    E: Fn(T) -> bool,
{
    let axis = resolve_axis(axis, a.ndim())?;
    let plan = AxisTraversalPlan::new(a.shape(), axis);
    checked_allocation_len::<i64>(plan.output_len)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.kept_shape);
    }
    if plan.axis_len == 0 {
        return Err(Error::EmptyReduction { op: op_name });
    }

    // Last axis + C-contiguous: each output row is one memory chunk.
    if plan.is_last_axis(a.ndim()) {
        if let Some(slice) = a.as_c_contiguous_slice() {
            return arg_extremum_axis_contiguous(
                slice,
                &plan,
                &mut is_better,
                &is_nan,
                &is_terminal,
            );
        }
    }

    arg_extremum_axis_strided(a, &plan, &mut is_better, &is_nan, &is_terminal)
}

/// Contiguous last-axis arg-extremum with NaN propagation.
///
/// Splits the flat contiguous buffer into row chunks of length `axis_len`
/// and finds the winning index within each chunk. Stops scanning a row once
/// `is_terminal` holds for the current best value.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `plan` - Single-axis traversal geometry.
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
/// * `is_terminal` - Early-exit predicate for values that cannot be improved.
///
/// # Returns
///
/// An `i64` array shaped like `plan.kept_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
fn arg_extremum_axis_contiguous<T, F, N, E>(
    slice: &[T],
    plan: &AxisTraversalPlan,
    is_better: &mut F,
    is_nan: &N,
    is_terminal: &E,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
    E: Fn(T) -> bool,
{
    let mut out = Vec::with_capacity(plan.output_len);
    for chunk in slice.chunks_exact(plan.axis_len) {
        let mut best_axis_index = 0_i64;
        let mut best_value = chunk[0];
        if !is_nan(best_value) && !is_terminal(best_value) {
            for (axis_index, &candidate) in chunk.iter().enumerate().skip(1) {
                if is_nan(candidate) {
                    best_axis_index = axis_index as i64;
                    break;
                }
                if is_better(candidate, best_value) {
                    best_value = candidate;
                    best_axis_index = axis_index as i64;
                    if is_terminal(candidate) {
                        break;
                    }
                }
            }
        }
        out.push(best_axis_index);
    }
    Array::from_vec(out, &plan.kept_shape)
}

/// Strided single-axis arg-extremum with NaN propagation.
///
/// Walks outer positions with [`RunPlan`], then steps along the axis stride
/// for each inner comparison. Each row scan may stop early when
/// `is_terminal` becomes true.
///
/// # Arguments
///
/// * `a` - Strided input array.
/// * `plan` - Single-axis traversal geometry.
/// * `is_better` - Comparison predicate for finite values.
/// * `is_nan` - NaN detector.
/// * `is_terminal` - Early-exit predicate for values that cannot be improved.
///
/// # Returns
///
/// An `i64` array shaped like `plan.kept_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
fn arg_extremum_axis_strided<T, F, N, E>(
    a: &Array<T>,
    plan: &AxisTraversalPlan,
    is_better: &mut F,
    is_nan: &N,
    is_terminal: &E,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
    E: Fn(T) -> bool,
{
    let mut out = Vec::with_capacity(plan.output_len);
    let axis_stride = a.strides()[plan.axis];
    let outer_strides = plan.kept_strides(a.strides());

    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        let mut buf = outer_base as isize;
        let mut best_value = a.data[buf as usize];
        let mut best_axis_index = 0_i64;

        if !is_nan(best_value) && !is_terminal(best_value) {
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
                    if is_terminal(candidate) {
                        break;
                    }
                }
            }
        }
        out.push(best_axis_index);
    });

    Array::from_vec(out, &plan.kept_shape)
}

/// Flat argmin/argmax skipping NaN elements.
///
/// Ignores NaN values entirely. When every element is NaN, returns
/// [`Error::AllNanSlice`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `_op_name` - Operation label (reserved for diagnostics).
/// * `is_better` - Comparison predicate among finite values.
/// * `is_nan` - Predicate marking values to skip.
///
/// # Returns
///
/// A 0-D `i64` array with the winning linear index among finite elements.
///
/// # Errors
///
/// Returns an error when allocation fails.
pub(crate) fn arg_extremum_flat_ignore<T, F, N>(
    a: &Array<T>,
    op_name: &'static str,
    mut is_better: F,
    is_nan: N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    if a.size() == 0 {
        return Err(Error::EmptyReduction { op: op_name });
    }
    let mut best: Option<(T, i64)> = None;
    let mut linear = 0_i64;
    let runs = RunPlan::new(a.shape(), [a.strides()]);
    runs.for_each([a.offset() as isize], |run| {
        let mut pos = run.bases[0] as isize;
        for _ in 0..run.len {
            let candidate = a.data[pos as usize];
            if !is_nan(candidate)
                && best
                    .map(|(value, _)| is_better(candidate, value))
                    .unwrap_or(true)
            {
                best = Some((candidate, linear));
            }
            linear += 1;
            pos += run.strides[0];
        }
    });
    match best {
        Some((_, index)) => Array::from_vec(vec![index], &[]),
        None => Err(Error::AllNanSlice { op: op_name }),
    }
}

/// Axis argmin/argmax skipping NaN elements.
///
/// Emits one axis-relative index per outer position, considering only
/// finite elements. Any all-NaN row returns [`Error::AllNanSlice`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to find extrema (may be negative).
/// * `_op_name` - Operation label (reserved for diagnostics).
/// * `is_better` - Comparison predicate among finite values.
/// * `is_nan` - Predicate marking values to skip.
///
/// # Returns
///
/// An `i64` array shaped like all dimensions except `axis`.
///
/// # Errors
///
/// Returns an error when the axis is out of range or allocation fails.
pub(crate) fn arg_extremum_axis_ignore<T, F, N>(
    a: &Array<T>,
    axis: isize,
    op_name: &'static str,
    mut is_better: F,
    is_nan: N,
) -> Result<Array<i64>>
where
    T: Scalar,
    F: FnMut(T, T) -> bool,
    N: Fn(T) -> bool,
{
    let axis = resolve_axis(axis, a.ndim())?;
    let plan = AxisTraversalPlan::new(a.shape(), axis);
    checked_allocation_len::<i64>(plan.output_len)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.kept_shape);
    }
    if plan.axis_len == 0 {
        return Err(Error::EmptyReduction { op: op_name });
    }

    let mut out = Vec::with_capacity(plan.output_len);
    let axis_stride = a.strides()[axis];
    let outer_strides = plan.kept_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
    let mut has_all_nan_slice = false;
    outer_runs.for_each_element([a.offset() as isize], |[base]| {
        let mut pos = base as isize;
        let mut best: Option<(T, i64)> = None;
        for index in 0..plan.axis_len {
            let candidate = a.data[pos as usize];
            if !is_nan(candidate)
                && best
                    .map(|(value, _)| is_better(candidate, value))
                    .unwrap_or(true)
            {
                best = Some((candidate, index as i64));
            }
            pos += axis_stride;
        }
        if let Some((_, index)) = best {
            out.push(index);
        } else {
            has_all_nan_slice = true;
        }
    });
    if has_all_nan_slice {
        return Err(Error::AllNanSlice { op: op_name });
    }
    Array::from_vec(out, &plan.kept_shape)
}
