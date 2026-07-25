//! Generic axis folds and NaN-skipping reductions with valid counts.
//!
//! Non-associative folds use a single accumulator per output slot.
//! Associative folds unroll eight partial chains on contiguous suffix
//! chunks. NaN-ignore paths live here so propagate kernels stay branch-free.

use super::*;

/// Generic fold over selected axes (non-associative operations).
///
/// Suitable for logical AND/OR and other operations that cannot use parallel
/// lane combining. Builds a [`ReducePlan`] then delegates to
/// [`reduce_fold_with_plan`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `initial` - Starting accumulator for each output slot.
/// * `accumulate` - Combine one element into the accumulator.
///
/// # Returns
///
/// An array of fold results with dtype `Acc`.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or allocation fails.
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
    reduce_fold_with_plan(a, &plan, initial, accumulate)
}

/// Fold using a pre-built [`ReducePlan`].
///
/// Picks suffix chunks, prefix rows, or general strided traversal from
/// layout metadata.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `plan` - Precomputed reduction geometry.
/// * `initial` - Starting accumulator per output slot.
/// * `accumulate` - Sequential combine step.
///
/// # Returns
///
/// An array shaped like `plan.output_shape` with dtype `Acc`.
///
/// # Errors
///
/// Returns an error when allocation fails.
pub(crate) fn reduce_fold_with_plan<T, Acc, F>(
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
    checked_allocation_len::<Acc>(plan.output_len)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
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

/// Prefix layout: each output slot accumulates down contiguous rows.
///
/// Initializes every slot to `initial`, then scans each row of length
/// `output_len` and updates all slots in parallel.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `plan` - Reduction geometry with prefix reduced axes.
/// * `initial` - Starting value per output slot.
/// * `accumulate` - Sequential combine step.
///
/// # Returns
///
/// Fold results shaped like `plan.output_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    let mut out = vec![initial; plan.output_len];
    if plan.reduction_len > 0 {
        for row in slice.chunks_exact(plan.output_len) {
            for (acc, &value) in out.iter_mut().zip(row) {
                *acc = accumulate(*acc, value);
            }
        }
    }
    Array::from_vec(out, &plan.output_shape)
}

/// Suffix layout: each contiguous chunk folds into one output slot.
///
/// Each chunk has length `plan.reduction_len` and produces one accumulator
/// value.
///
/// # Arguments
///
/// * `slice` - C-contiguous input elements.
/// * `plan` - Reduction geometry with suffix reduced axes.
/// * `initial` - Starting value per chunk.
/// * `accumulate` - Sequential combine step.
///
/// # Returns
///
/// Fold results shaped like `plan.output_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    let mut out = Vec::with_capacity(plan.output_len);
    if plan.reduction_len == 0 {
        out.resize(plan.output_len, initial);
        return Array::from_vec(out, &plan.output_shape);
    }

    for chunk in slice.chunks_exact(plan.reduction_len) {
        let mut acc = initial;
        for &x in chunk {
            acc = accumulate(acc, x);
        }
        out.push(acc);
    }
    Array::from_vec(out, &plan.output_shape)
}

/// Associative fold with automatic plan construction.
///
/// Uses eight-lane partial accumulators on contiguous suffix chunks when
/// layout allows.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `initial` - Identity element for the operation.
/// * `accumulate` - Per-element combine step.
/// * `combine` - Merge partial accumulators from parallel lanes.
///
/// # Returns
///
/// An array of reduced values with dtype `Acc`.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or allocation fails.
pub(crate) fn reduce_associative<T, Acc, F, G>(
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
    reduce_associative_with_plan(a, &plan, initial, accumulate, combine)
}

/// Associative fold: eight partial accumulators on suffix chunks.
///
/// Breaks loop-carried dependencies on contiguous suffix reductions by
/// unrolling eight independent chains, then merging with `combine`.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `plan` - Precomputed reduction geometry.
/// * `initial` - Identity element for the operation.
/// * `accumulate` - Per-element combine step.
/// * `combine` - Merge partial accumulators from parallel lanes.
///
/// # Returns
///
/// An array shaped like `plan.output_shape` with dtype `Acc`.
///
/// # Errors
///
/// Returns an error when allocation fails.
pub(crate) fn reduce_associative_with_plan<T, Acc, F, G>(
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
    checked_allocation_len::<Acc>(plan.output_len)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }

    match reduction_path(a, plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.output_len);
            if plan.reduction_len == 0 {
                out.resize(plan.output_len, initial);
                return Array::from_vec(out, &plan.output_shape);
            }

            for chunk in slice.chunks_exact(plan.reduction_len) {
                // Break loop-carried dependency with eight lanes.
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
            Array::from_vec(out, &plan.output_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            fold_prefix_contiguous(slice, plan, initial, &mut accumulate)
        }
        ReductionPath::GeneralStrided => {
            fold_strided_general(a, plan, initial, &mut accumulate)
        }
    }
}

/// General-strided sequential fold over kept and reduced axes.
///
/// Uses outer [`RunPlan`] walks and inner [`ReducedAxisRuns`] for each
/// output slot.
///
/// # Arguments
///
/// * `a` - Strided input array.
/// * `plan` - Reduction geometry.
/// * `initial` - Starting accumulator per slot.
/// * `accumulate` - Sequential combine step.
///
/// # Returns
///
/// Fold results shaped like `plan.output_shape`.
///
/// # Errors
///
/// Returns an error when allocation fails.
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
    let mut out = Vec::with_capacity(plan.output_len);
    let (outer_strides, reduced_strides) =
        plan.kept_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
    let reduced = ReducedAxisRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut reduced_cursor = reduced.cursor(a.offset() as isize);

    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        let mut acc = initial;
        if plan.reduction_len > 0 {
            reduced_cursor.reset([outer_base as isize]);
            for run_i in 0..reduced.run_count {
                let mut pos = reduced_cursor.operand_offset(0) as isize;
                for _ in 0..reduced.run_len {
                    acc = accumulate(acc, a.data[pos as usize]);
                    pos += reduced.operand_stride;
                }
                if run_i + 1 < reduced.run_count {
                    reduced_cursor.advance();
                }
            }
        }
        out.push(acc);
    });

    Array::from_vec(out, &plan.output_shape)
}

/// NaN-skipping fold returning per-slot valid element counts.
///
/// Skips NaN inputs during accumulation. Slots with zero finite elements
/// receive the `nan` sentinel. Counts enable mean division in the trait
/// layer.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `plan` - Precomputed reduction geometry.
/// * `initial` - Identity before any finite element is seen.
/// * `nan` - Sentinel for all-NaN non-empty slices.
/// * `accumulate` - Per-element combine for finite values.
/// * `combine` - Merge partial accumulators (associative path).
/// * `is_nan` - NaN detector.
///
/// # Returns
///
/// A tuple of `(values, counts)` where `values` has shape
/// `plan.output_shape` and `counts[i]` is the number of finite elements
/// folded into slot `i`.
///
/// # Errors
///
/// Returns an error when allocation fails.
pub(crate) fn reduce_ignore_with_counts<T, Acc, F, G, N>(
    a: &Array<T>,
    plan: &ReducePlan,
    initial: Acc,
    nan: Acc,
    mut accumulate: F,
    mut combine: G,
    is_nan: N,
) -> Result<(Array<Acc>, Vec<usize>)>
where
    T: Scalar,
    Acc: Scalar,
    F: FnMut(Acc, T) -> Acc,
    G: FnMut(Acc, Acc) -> Acc,
    N: Fn(T) -> bool,
{
    checked_allocation_len::<Acc>(plan.output_len)?;
    checked_allocation_len::<usize>(plan.output_len)?;
    if plan.output_len == 0 {
        return Ok((
            Array::from_vec(Vec::new(), &plan.output_shape)?,
            Vec::new(),
        ));
    }
    if plan.reduction_len == 0 {
        return Ok((
            Array::from_vec(
                vec![initial; plan.output_len],
                &plan.output_shape,
            )?,
            vec![0; plan.output_len],
        ));
    }

    let mut values = Vec::with_capacity(plan.output_len);
    let mut counts = Vec::with_capacity(plan.output_len);
    match reduction_path(a, plan) {
        ReductionPath::SuffixContiguous(slice) => {
            for chunk in slice.chunks_exact(plan.reduction_len) {
                let mut partials = [initial; 8];
                let mut lane_counts = [0_usize; 8];
                let mut blocks = chunk.chunks_exact(8);
                for block in &mut blocks {
                    for lane in 0..8 {
                        let value = block[lane];
                        if !is_nan(value) {
                            partials[lane] = accumulate(partials[lane], value);
                            lane_counts[lane] += 1;
                        }
                    }
                }
                let mut acc = initial;
                let mut count = 0;
                for lane in 0..8 {
                    acc = combine(acc, partials[lane]);
                    count += lane_counts[lane];
                }
                for &value in blocks.remainder() {
                    if !is_nan(value) {
                        acc = accumulate(acc, value);
                        count += 1;
                    }
                }
                values.push(if count == 0 { nan } else { acc });
                counts.push(count);
            }
        }
        ReductionPath::PrefixContiguous(slice) => {
            values.resize(plan.output_len, initial);
            counts.resize(plan.output_len, 0);
            for row in slice.chunks_exact(plan.output_len) {
                for ((acc, count), &value) in
                    values.iter_mut().zip(&mut counts).zip(row)
                {
                    if !is_nan(value) {
                        *acc = accumulate(*acc, value);
                        *count += 1;
                    }
                }
            }
            for (value, &count) in values.iter_mut().zip(&counts) {
                if count == 0 {
                    *value = nan;
                }
            }
        }
        ReductionPath::GeneralStrided => {
            let (outer_strides, reduced_strides) =
                plan.kept_reduced_strides(a.strides());
            let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
            let reduced =
                ReducedAxisRuns::new(&plan.reduced_shape, &reduced_strides);
            let mut reduced_cursor = reduced.cursor(a.offset() as isize);
            outer_runs.for_each_element(
                [a.offset() as isize],
                |[outer_base]| {
                    let mut acc = initial;
                    let mut count = 0;
                    reduced_cursor.reset([outer_base as isize]);
                    for run_i in 0..reduced.run_count {
                        let mut pos = reduced_cursor.operand_offset(0) as isize;
                        for _ in 0..reduced.run_len {
                            let value = a.data[pos as usize];
                            if !is_nan(value) {
                                acc = accumulate(acc, value);
                                count += 1;
                            }
                            pos += reduced.operand_stride;
                        }
                        if run_i + 1 < reduced.run_count {
                            reduced_cursor.advance();
                        }
                    }
                    values.push(if count == 0 { nan } else { acc });
                    counts.push(count);
                },
            );
        }
    }
    Ok((Array::from_vec(values, &plan.output_shape)?, counts))
}
