use super::*;

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
    debug_assert!(run_plan.run_len() > 0 && run_plan.run_count() > 0);
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
    let axis = normalize_axis(axis, a.ndim());
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
    let in_outer_strides = plan.kept_strides(a.strides());
    let out_outer_strides = plan.kept_strides(&out_strides);

    let mut out = Vec::with_capacity(n);
    out.resize(n, to_acc(a.data[a.offset]));

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
    match axis {
        None => cumulate_flat_ignore(a, nan, to_acc, accumulate, is_nan),
        Some(axis) => {
            cumulate_axis_ignore(a, axis, nan, to_acc, accumulate, is_nan)
        }
    }
}

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
    let axis = normalize_axis(axis, a.ndim());
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

    let out_strides = c_order_strides(a.shape());
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
