use super::*;

fn build_extremum_plan<T: Scalar>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    op_name: &str,
) -> Result<ReducePlan> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.output_len > 0 && plan.reduction_is_empty() {
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

/// Boolean min/max is all/any respectively. The contiguous kernel uses
/// bitwise accumulation so there is no comparison or generic dispatch in the
/// hot loop.
pub(crate) fn reduce_bool_extremum<const MIN: bool>(
    a: &Array<bool>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    let op_name = if MIN { "min" } else { "max" };
    let plan = build_extremum_plan(a, axes, keepdims, op_name)?;
    reduce_bool_with_plan::<MIN>(a, &plan)
}

/// Boolean all/any shares the bitwise min/max kernel but preserves logical
/// reduction identities on empty reduced axes.
pub(crate) fn reduce_bool_logical<const ALL: bool>(
    a: &Array<bool>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    reduce_bool_with_plan::<ALL>(a, &plan)
}

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
    let plan = build_extremum_plan(a, axes, keepdims, op_name)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.output_len);
            for chunk in slice.chunks_exact(plan.reduction_len) {
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

/// Floating min/max keeps NaN propagation while using independent comparison
/// chains for NaN-free contiguous chunks. NaNs are still observed in logical
/// C order, so the first NaN value is propagated as before.
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

    // The overwhelmingly common case is a fully coalesced reduced layout
    // (single axis, or several axes that merge into one linear run). A flat
    // `pos += stride` loop lets the compiler keep `best`/`pos` in registers
    // and avoids wrapping every element in an extra run-counting loop, which
    // otherwise measurably hurts this branch-heavy (NaN-checking) kernel.
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
