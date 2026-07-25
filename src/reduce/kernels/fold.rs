use super::*;

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

/// Fold using a precomputed [`ReducePlan`] (avoids re-resolving axes).
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
    debug_assert_eq!(out.len(), plan.output_len);
    Array::from_vec(out, &plan.output_shape)
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
    reduce_sum_with_plan(a, &plan, initial, accumulate, combine)
}

/// Sum using a precomputed plan. The eight independent accumulators remove the
/// loop-carried dependency that prevents SIMD/ILP on contiguous chunks.
pub(crate) fn reduce_sum_with_plan<T, Acc, F, G>(
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
            debug_assert_eq!(out.len(), plan.output_len);
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
