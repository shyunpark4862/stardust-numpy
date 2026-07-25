use super::*;

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
    debug_assert!(run_plan.run_len() > 0 && run_plan.run_count() > 0);
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
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.kept_shape);
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
    let mut out = Vec::with_capacity(plan.output_len);
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
    Array::from_vec(out, &plan.kept_shape)
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
    let mut out = Vec::with_capacity(plan.output_len);
    let axis_stride = a.strides()[plan.axis];
    let outer_strides = plan.kept_strides(a.strides());

    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
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

    Array::from_vec(out, &plan.kept_shape)
}
