use super::*;

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
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    if plan.reduction_is_empty() {
        return Array::from_vec(
            vec![f64::NAN; plan.output_len],
            &plan.output_shape,
        );
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

pub(crate) fn reduce_var_ignore_f64(
    a: &Array<f64>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<f64>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    if plan.reduction_is_empty() {
        return Array::from_vec(
            vec![f64::NAN; plan.output_len],
            &plan.output_shape,
        );
    }
    match reduction_path(a, &plan) {
        ReductionPath::SuffixContiguous(slice) => {
            let mut out = Vec::with_capacity(plan.output_len);
            for chunk in slice.chunks_exact(plan.reduction_len) {
                let mut sums = [0.0; 8];
                let mut counts = [0_usize; 8];
                let mut blocks = chunk.chunks_exact(8);
                for block in &mut blocks {
                    for lane in 0..8 {
                        let value = block[lane];
                        if !value.is_nan() {
                            sums[lane] += value;
                            counts[lane] += 1;
                        }
                    }
                }
                let mut sum = merge_eight_f64(sums);
                let mut count = counts.into_iter().sum::<usize>();
                for &value in blocks.remainder() {
                    if !value.is_nan() {
                        sum += value;
                        count += 1;
                    }
                }
                if count == 0 {
                    out.push(f64::NAN);
                    continue;
                }
                let mean = sum / count as f64;
                let mut deviations = [0.0; 8];
                let mut blocks = chunk.chunks_exact(8);
                for block in &mut blocks {
                    for lane in 0..8 {
                        let value = block[lane];
                        if !value.is_nan() {
                            let delta = value - mean;
                            deviations[lane] += delta * delta;
                        }
                    }
                }
                let mut squared = merge_eight_f64(deviations);
                for &value in blocks.remainder() {
                    if !value.is_nan() {
                        let delta = value - mean;
                        squared += delta * delta;
                    }
                }
                out.push(squared / count as f64);
            }
            Array::from_vec(out, &plan.output_shape)
        }
        ReductionPath::PrefixContiguous(slice) => {
            let mut sums = vec![0.0; plan.output_len];
            let mut counts = vec![0_usize; plan.output_len];
            for row in slice.chunks_exact(plan.output_len) {
                for ((sum, count), &value) in
                    sums.iter_mut().zip(&mut counts).zip(row)
                {
                    if !value.is_nan() {
                        *sum += value;
                        *count += 1;
                    }
                }
            }
            let means: Vec<f64> = sums
                .iter()
                .zip(&counts)
                .map(|(&sum, &count)| {
                    if count == 0 {
                        f64::NAN
                    } else {
                        sum / count as f64
                    }
                })
                .collect();
            let mut out = vec![0.0; plan.output_len];
            for row in slice.chunks_exact(plan.output_len) {
                for ((sum, &mean), &value) in
                    out.iter_mut().zip(&means).zip(row)
                {
                    if !value.is_nan() {
                        let delta = value - mean;
                        *sum += delta * delta;
                    }
                }
            }
            for ((value, &count), &mean) in
                out.iter_mut().zip(&counts).zip(&means)
            {
                *value = if mean.is_nan() {
                    f64::NAN
                } else {
                    *value / count as f64
                };
            }
            Array::from_vec(out, &plan.output_shape)
        }
        ReductionPath::GeneralStrided => var_ignore_strided_general(a, &plan),
    }
}

fn var_ignore_strided_general(
    a: &Array<f64>,
    plan: &ReducePlan,
) -> Result<Array<f64>> {
    let mut out = Vec::with_capacity(plan.output_len);
    let (outer_strides, reduced_strides) =
        plan.kept_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
    let reduced = ReducedAxisRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut cursor = reduced.cursor(a.offset() as isize);
    outer_runs.for_each_element([a.offset() as isize], |[base]| {
        cursor.reset([base as isize]);
        let mut sum = 0.0;
        let mut count = 0;
        for run_i in 0..reduced.run_count {
            let mut pos = cursor.operand_offset(0) as isize;
            for _ in 0..reduced.run_len {
                let value = a.data[pos as usize];
                if !value.is_nan() {
                    sum += value;
                    count += 1;
                }
                pos += reduced.operand_stride;
            }
            if run_i + 1 < reduced.run_count {
                cursor.advance();
            }
        }
        if count == 0 {
            out.push(f64::NAN);
            return;
        }
        let mean = sum / count as f64;
        cursor.reset([base as isize]);
        let mut squared = 0.0;
        for run_i in 0..reduced.run_count {
            let mut pos = cursor.operand_offset(0) as isize;
            for _ in 0..reduced.run_len {
                let value = a.data[pos as usize];
                if !value.is_nan() {
                    let delta = value - mean;
                    squared += delta * delta;
                }
                pos += reduced.operand_stride;
            }
            if run_i + 1 < reduced.run_count {
                cursor.advance();
            }
        }
        out.push(squared / count as f64);
    });
    Array::from_vec(out, &plan.output_shape)
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
    let count = plan.reduction_len as f64;
    let mut out = Vec::with_capacity(plan.output_len);
    for chunk in slice.chunks_exact(plan.reduction_len) {
        out.push(variance_chunk(chunk, count, to_f64));
    }
    Array::from_vec(out, &plan.output_shape)
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
    let count = plan.reduction_len as f64;
    let mut means = vec![0.0; plan.output_len];
    for row in slice.chunks_exact(plan.output_len) {
        for (sum, &value) in means.iter_mut().zip(row) {
            *sum += to_f64(value);
        }
    }
    for mean in &mut means {
        *mean /= count;
    }

    let mut variances = vec![0.0; plan.output_len];
    for row in slice.chunks_exact(plan.output_len) {
        for ((sum, &mean), &value) in variances.iter_mut().zip(&means).zip(row)
        {
            let delta = to_f64(value) - mean;
            *sum += delta * delta;
        }
    }
    for variance in &mut variances {
        *variance /= count;
    }
    Array::from_vec(variances, &plan.output_shape)
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
    let count = plan.reduction_len as f64;
    let mut out = Vec::with_capacity(plan.output_len);
    let (outer_strides, reduced_strides) =
        plan.kept_reduced_strides(a.strides());
    let outer_runs = RunPlan::new(&plan.kept_shape, [&outer_strides]);
    let reduced = ReducedAxisRuns::new(&plan.reduced_shape, &reduced_strides);
    let mut reduced_cursor = reduced.cursor(a.offset() as isize);

    outer_runs.for_each_element([a.offset() as isize], |[outer_base]| {
        let base = outer_base as isize;

        reduced_cursor.reset([base]);
        let mut sum = 0.0;
        for run_i in 0..reduced.run_count {
            let mut pos = reduced_cursor.operand_offset(0) as isize;
            for _ in 0..reduced.run_len {
                sum += to_f64(a.data[pos as usize]);
                pos += reduced.operand_stride;
            }
            if run_i + 1 < reduced.run_count {
                reduced_cursor.advance();
            }
        }
        let mean = sum / count;

        reduced_cursor.reset([base]);
        let mut squared_deviation_sum = 0.0;
        for run_i in 0..reduced.run_count {
            let mut pos = reduced_cursor.operand_offset(0) as isize;
            for _ in 0..reduced.run_len {
                let delta = to_f64(a.data[pos as usize]) - mean;
                squared_deviation_sum += delta * delta;
                pos += reduced.operand_stride;
            }
            if run_i + 1 < reduced.run_count {
                reduced_cursor.advance();
            }
        }
        out.push(squared_deviation_sum / count);
    });

    Array::from_vec(out, &plan.output_shape)
}

/// Map every logical element of an **owned, C-contiguous, offset-0** array.
///
/// Intended only for freshly allocated reduction results (mean / std).
pub(crate) fn transform_owned_c_order<T, F>(
    mut a: Array<T>,
    mut f: F,
) -> Array<T>
where
    T: Scalar,
    F: FnMut(T) -> T,
{
    debug_assert_eq!(
        a.offset(),
        0,
        "transform_owned_c_order requires offset 0"
    );
    debug_assert!(
        a.is_c_contiguous(),
        "transform_owned_c_order requires C-contiguous layout"
    );
    debug_assert!(
        a.is_writable(),
        "transform_owned_c_order requires a writable array"
    );
    debug_assert_eq!(
        a.size(),
        a.data.len(),
        "transform_owned_c_order requires the buffer to match the logical size"
    );

    let data = Arc::make_mut(&mut a.data);
    for x in data.iter_mut() {
        *x = f(*x);
    }
    a
}
