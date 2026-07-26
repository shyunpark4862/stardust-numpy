//! Prefix sum and prefix product (cumsum / cumprod) kernels.
//!
//! Flat scans preserve C-order linear indexing. Axis scans write output in
//! C order even when the input is strided. NaN-ignore variants carry a
//! running accumulator and a `seen` flag so leading NaNs emit NaN without
//! polluting later finite values.

use super::*;

/// Cumulative fold along `axis`, or flat C order when `axis` is `None`.
///
/// Output shape matches the input. NaN is not filtered here; callers use
/// [`cumulate_ignore`] for skip-NaN semantics.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis to scan, or `None` for flat C order.
/// * `to_acc` - Promote input elements to the accumulator type.
/// * `accumulate` - Combine one element into the running prefix value.
///
/// # Returns
///
/// An array of the same shape as `a` with dtype `Acc`.
///
/// # Errors
///
/// Returns an error when the axis is out of range or allocation fails.
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
    checked_allocation_len::<Acc>(a.size())?;
    match axis {
        None => cumulate_flat(a, to_acc, accumulate),
        Some(ax) => cumulate_axis(a, ax, to_acc, accumulate),
    }
}

/// Flat prefix scan dispatcher (contiguous vs strided).
///
/// # Arguments
///
/// * `a` - Input array.
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
///
/// # Returns
///
/// A 1-D-shaped `Acc` array of length `a.size()`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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

/// Contiguous flat prefix scan.
///
/// Single forward pass over a dense slice, emitting running totals.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
///
/// # Returns
///
/// Prefix values with shape `[n]` where `n = slice.len()`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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

/// Strided flat prefix scan.
///
/// Carries the running accumulator across coalesced outer/inner runs while
/// preserving C-order linear output order.
///
/// # Arguments
///
/// * `a` - Strided input array.
/// * `n` - Total element count (`a.size()`).
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
///
/// # Returns
///
/// Prefix values with shape `[n]`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    // Carry the running accumulator across coalesced outer/inner runs.
    let mut out = Vec::with_capacity(n);
    let run_plan = RunPlan::new(a.shape(), [a.strides()]);
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

    Array::from_vec(out, &[n])
}

/// Single-axis prefix scan dispatcher.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis to scan (may be negative).
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
///
/// # Returns
///
/// Prefix values with the same shape as `a`.
///
/// # Errors
///
/// Returns an error when the axis is out of range or allocation fails.
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
    let axis = resolve_axis(axis, a.ndim())?;
    let plan = AxisTraversalPlan::new(a.shape(), axis);
    let n = a.size();
    if n == 0 {
        return Array::from_vec(Vec::new(), a.shape());
    }

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

/// Contiguous last-axis prefix scan.
///
/// Each row chunk of length `axis_len` is scanned independently.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `plan` - Single-axis traversal geometry.
/// * `shape` - Full input shape (output matches).
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
///
/// # Returns
///
/// Prefix values shaped like `shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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

/// Strided single-axis prefix scan with C-order output materialization.
///
/// Input may be arbitrarily strided; output is always laid out in C order.
///
/// # Arguments
///
/// * `a` - Strided input array.
/// * `plan` - Single-axis traversal geometry.
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
///
/// # Returns
///
/// Prefix values shaped like `a`, stored in C order.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    // Output is always materialized in C order.
    let out_strides = c_order_strides_unchecked(a.shape());
    let out_stride = out_strides[plan.axis];
    let in_outer_strides = plan.kept_strides(a.strides());
    let out_outer_strides = plan.kept_strides(&out_strides);

    let mut out = Vec::with_capacity(n);
    out.resize(n, to_acc(a.data[a.offset()]));

    let outer_runs =
        RunPlan::new(&plan.kept_shape, [&in_outer_strides, &out_outer_strides]);
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

/// Cumulative fold with NaN-skipping semantics.
///
/// Leading NaNs emit the `nan` sentinel without updating the accumulator.
/// After the first finite value, later NaNs leave the running total
/// unchanged (NumPy `nancumsum` behavior).
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis to scan, or `None` for flat C order.
/// * `nan` - Sentinel for positions before any finite input.
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step for finite values.
/// * `is_nan` - Predicate marking values to skip.
///
/// # Returns
///
/// An array of the same shape as `a` with dtype `Acc`.
///
/// # Errors
///
/// Returns an error when the axis is out of range or allocation fails.
pub(crate) fn cumulate_ignore<T, Acc, C, F, N>(
    a: &Array<T>,
    axis: Option<isize>,
    nan: Acc,
    to_acc: C,
    accumulate: F,
    is_nan: N,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc + Copy,
    F: FnMut(Acc, T) -> Acc,
    N: Fn(T) -> bool + Copy,
{
    checked_allocation_len::<Acc>(a.size())?;
    match axis {
        None => cumulate_flat_ignore(a, nan, to_acc, accumulate, is_nan),
        Some(axis) => {
            cumulate_axis_ignore(a, axis, nan, to_acc, accumulate, is_nan)
        }
    }
}

/// Update one step of a NaN-aware prefix scan.
///
/// Shared by flat and axis NaN-ignore cumulative kernels. Tracks whether any
/// finite value has been seen yet.
///
/// # Arguments
///
/// * `value` - Current input element.
/// * `acc` - Running accumulator (mutated when a finite value arrives).
/// * `seen` - Whether at least one finite value has been processed.
/// * `nan` - Sentinel for pre-first-finite positions.
/// * `to_acc` - Promotion for the first finite element.
/// * `accumulate` - Combine step after the first finite element.
/// * `is_nan` - NaN detector.
///
/// # Returns
///
/// The output value to store at this position.
fn scan_ignore<T, Acc, C, F, N>(
    value: T,
    acc: &mut Acc,
    seen: &mut bool,
    nan: Acc,
    to_acc: C,
    accumulate: &mut F,
    is_nan: N,
) -> Acc
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc,
    F: FnMut(Acc, T) -> Acc,
    N: Fn(T) -> bool,
{
    if is_nan(value) {
        if *seen {
            *acc
        } else {
            nan
        }
    } else {
        if *seen {
            *acc = accumulate(*acc, value);
        } else {
            *acc = to_acc(value);
            *seen = true;
        }
        *acc
    }
}

/// Flat NaN-ignore prefix scan.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `nan` - Sentinel before the first finite element.
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
/// * `is_nan` - NaN detector.
///
/// # Returns
///
/// Prefix values with shape `[a.size()]`.
///
/// # Errors
///
/// Returns an error when allocation fails.
fn cumulate_flat_ignore<T, Acc, C, F, N>(
    a: &Array<T>,
    nan: Acc,
    to_acc: C,
    mut accumulate: F,
    is_nan: N,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc + Copy,
    F: FnMut(Acc, T) -> Acc,
    N: Fn(T) -> bool + Copy,
{
    let n = a.size();
    if n == 0 {
        return Array::from_vec(Vec::new(), &[0]);
    }
    let mut out = Vec::with_capacity(n);
    let mut acc = nan;
    let mut seen = false;
    if let Some(slice) = a.as_c_contiguous_slice() {
        for &value in slice {
            out.push(scan_ignore(
                value,
                &mut acc,
                &mut seen,
                nan,
                to_acc,
                &mut accumulate,
                is_nan,
            ));
        }
    } else {
        let runs = RunPlan::new(a.shape(), [a.strides()]);
        runs.for_each([a.offset() as isize], |run| {
            let mut pos = run.bases[0] as isize;
            for _ in 0..run.len {
                out.push(scan_ignore(
                    a.data[pos as usize],
                    &mut acc,
                    &mut seen,
                    nan,
                    to_acc,
                    &mut accumulate,
                    is_nan,
                ));
                pos += run.strides[0];
            }
        });
    }
    Array::from_vec(out, &[n])
}

/// Single-axis NaN-ignore prefix scan.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis to scan (may be negative).
/// * `nan` - Sentinel before the first finite element in each scan line.
/// * `to_acc` - Element promotion into `Acc`.
/// * `accumulate` - Prefix combine step.
/// * `is_nan` - NaN detector.
///
/// # Returns
///
/// Prefix values with the same shape as `a`.
///
/// # Errors
///
/// Returns an error when the axis is out of range or allocation fails.
fn cumulate_axis_ignore<T, Acc, C, F, N>(
    a: &Array<T>,
    axis: isize,
    nan: Acc,
    to_acc: C,
    mut accumulate: F,
    is_nan: N,
) -> Result<Array<Acc>>
where
    T: Scalar,
    Acc: Scalar,
    C: Fn(T) -> Acc + Copy,
    F: FnMut(Acc, T) -> Acc,
    N: Fn(T) -> bool + Copy,
{
    let axis = resolve_axis(axis, a.ndim())?;
    let plan = AxisTraversalPlan::new(a.shape(), axis);
    if a.size() == 0 {
        return Array::from_vec(Vec::new(), a.shape());
    }
    if plan.is_last_axis(a.ndim()) {
        if let Some(slice) = a.as_c_contiguous_slice() {
            let mut out = Vec::with_capacity(slice.len());
            for chunk in slice.chunks_exact(plan.axis_len) {
                let mut acc = nan;
                let mut seen = false;
                for &value in chunk {
                    out.push(scan_ignore(
                        value,
                        &mut acc,
                        &mut seen,
                        nan,
                        to_acc,
                        &mut accumulate,
                        is_nan,
                    ));
                }
            }
            return Array::from_vec(out, a.shape());
        }
    }

    let out_strides = c_order_strides_unchecked(a.shape());
    let in_stride = a.strides()[plan.axis];
    let out_stride = out_strides[plan.axis];
    let in_outer = plan.kept_strides(a.strides());
    let out_outer = plan.kept_strides(&out_strides);
    let mut out = vec![nan; a.size()];
    let outer_runs = RunPlan::new(&plan.kept_shape, [&in_outer, &out_outer]);
    outer_runs.for_each_element(
        [a.offset() as isize, 0],
        |[input_base, output_base]| {
            let mut input = input_base as isize;
            let mut output = output_base as isize;
            let mut acc = nan;
            let mut seen = false;
            for _ in 0..plan.axis_len {
                out[output as usize] = scan_ignore(
                    a.data[input as usize],
                    &mut acc,
                    &mut seen,
                    nan,
                    to_acc,
                    &mut accumulate,
                    is_nan,
                );
                input += in_stride;
                output += out_stride;
            }
        },
    );
    Array::from_vec(out, a.shape())
}
