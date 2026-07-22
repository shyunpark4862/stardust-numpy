use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Criterion,
};
use sdnp::{
    absolute, add, any, argmax, broadcast_arrays, cumprod, cumsum, divide, eye,
    full, gather, greater, isnan, max as reduce_max, mean, min as reduce_min,
    multiply, negative, ones, prod, scatter, scatter_array, std as reduce_std,
    sum, var, zeros, Array, IndexSpec,
};

const SIDE: usize = 1024;
const ELEMENTS: usize = SIDE * SIDE;

fn matrix(seed: usize) -> Array<f64> {
    let data = (0..ELEMENTS)
        .map(|i| ((i + seed) % 4096) as f64 * 0.25)
        .collect();
    Array::from_vec(data, &[SIDE, SIDE]).unwrap()
}

fn bench_creation(c: &mut Criterion) {
    c.bench_function("full_1024x1024_f64", |b| {
        b.iter(|| {
            black_box(full::<f64>(&[SIDE, SIDE], black_box(3.5)).unwrap())
        })
    });
    c.bench_function("zeros_1024x1024_f64", |b| {
        b.iter(|| black_box(zeros::<f64>(&[SIDE, SIDE]).unwrap()))
    });
    c.bench_function("ones_1024x1024_f64", |b| {
        b.iter(|| black_box(ones::<f64>(&[SIDE, SIDE]).unwrap()))
    });
    c.bench_function("eye_1024_f64", |b| {
        b.iter(|| black_box(eye::<f64>(SIDE).unwrap()))
    });
}

fn bench_views_and_materialization(c: &mut Criterion) {
    let a = matrix(0);
    let transposed = a.transpose();
    let row = Array::from_vec(vec![1.0_f64; SIDE], &[1, SIDE]).unwrap();
    let reshape_shape = [512_isize, 2048_isize];

    c.bench_function("transpose_view_f64", |b| {
        b.iter(|| black_box(black_box(&a).transpose()))
    });
    c.bench_function("reshape_contiguous_view_f64", |b| {
        b.iter(|| black_box(black_box(&a).reshape(&reshape_shape).unwrap()))
    });
    c.bench_function("broadcast_to_view_f64", |b| {
        b.iter(|| {
            black_box(black_box(&row).broadcast_to(&[SIDE, SIDE]).unwrap())
        })
    });
    c.bench_function("broadcast_arrays_f64", |b| {
        b.iter(|| {
            black_box(
                broadcast_arrays(&[black_box(&a), black_box(&row)]).unwrap(),
            )
        })
    });
    c.bench_function("copy_contiguous_f64", |b| {
        b.iter(|| black_box(black_box(&a).copy()))
    });
    c.bench_function("copy_transposed_f64", |b| {
        b.iter(|| black_box(black_box(&transposed).copy()))
    });
    c.bench_function("to_vec_transposed_f64", |b| {
        b.iter(|| black_box(black_box(&transposed).to_vec()))
    });
}

fn bench_reductions(c: &mut Criterion) {
    let a = matrix(0);
    let a_nonzero = matrix(1);
    let axis_last = [1_isize];
    let axis_first = [0_isize];
    let product_data = (0..ELEMENTS)
        .map(|i| 1.0 + (i % 7) as f64 * 0.000_001)
        .collect();
    let product_input = Array::from_vec(product_data, &[SIDE, SIDE]).unwrap();

    c.bench_function("sum_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(sum(black_box(&a), Some(&axis_last), false).unwrap())
        })
    });

    c.bench_function("sum_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(sum(black_box(&a), Some(&axis_first), false).unwrap())
        })
    });

    let cube_data = (0..ELEMENTS).map(|i| (i % 2048) as f64 * 0.5).collect();
    let cube = Array::from_vec(cube_data, &[64, 128, 128]).unwrap();
    let permuted = cube.permute_axes(&[1, 0, 2]).unwrap();
    let general_axes = [0_isize, 2_isize];

    c.bench_function("sum_multi_axis_general_f64", |b| {
        b.iter(|| {
            black_box(
                sum(black_box(&permuted), Some(&general_axes), false).unwrap(),
            )
        })
    });

    c.bench_function("var_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(var(black_box(&a), Some(&axis_last), false).unwrap())
        })
    });
    c.bench_function("var_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(var(black_box(&a), Some(&axis_first), false).unwrap())
        })
    });
    c.bench_function("min_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(black_box(&a_nonzero), Some(&axis_first), false)
                    .unwrap(),
            )
        })
    });
    c.bench_function("prod_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                prod(black_box(&product_input), Some(&axis_last), false)
                    .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(black_box(&a_nonzero), Some(&axis_last), false)
                    .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(black_box(&a_nonzero), Some(&axis_last), false)
                    .unwrap(),
            )
        })
    });
    c.bench_function("mean_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                mean(black_box(&a_nonzero), Some(&axis_last), false).unwrap(),
            )
        })
    });
    c.bench_function("std_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_std(black_box(&a_nonzero), Some(&axis_last), false)
                    .unwrap(),
            )
        })
    });
    c.bench_function("argmax_axis_last_f64", |b| {
        b.iter(|| black_box(argmax(black_box(&a_nonzero), Some(1)).unwrap()))
    });
    c.bench_function("any_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                any(black_box(&a_nonzero), Some(&axis_last), false).unwrap(),
            )
        })
    });
    c.bench_function("cumprod_axis_last_f64", |b| {
        b.iter(|| {
            black_box(cumprod(black_box(&product_input), Some(1)).unwrap())
        })
    });
}

fn bench_elementwise(c: &mut Criterion) {
    let left = matrix(0);
    let right = matrix(17);
    let a = matrix(1);
    let denominator = Array::from_vec(
        (0..ELEMENTS)
            .map(|i| (((i + 17) % 4096) + 1) as f64 * 0.25)
            .collect(),
        &[SIDE, SIDE],
    )
    .unwrap();
    let row = Array::from_vec(
        (0..SIDE).map(|i| (i + 1) as f64).collect(),
        &[1, SIDE],
    )
    .unwrap();

    c.bench_function("add_contiguous_contiguous_f64", |b| {
        b.iter(|| black_box(add(black_box(&left), black_box(&right)).unwrap()))
    });

    let left_t = left.transpose();
    let right_t = right.transpose();
    c.bench_function("add_strided_strided_f64", |b| {
        b.iter(|| {
            black_box(add(black_box(&left_t), black_box(&right_t)).unwrap())
        })
    });
    c.bench_function("negative_contiguous_f64", |b| {
        b.iter(|| black_box(negative(black_box(&a)).unwrap()))
    });
    c.bench_function("absolute_contiguous_f64", |b| {
        b.iter(|| black_box(absolute(black_box(&a)).unwrap()))
    });
    c.bench_function("isnan_contiguous_f64", |b| {
        b.iter(|| black_box(isnan(black_box(&a)).unwrap()))
    });
    c.bench_function("multiply_row_broadcast_f64", |b| {
        b.iter(|| black_box(multiply(black_box(&a), black_box(&row)).unwrap()))
    });
    c.bench_function("divide_contiguous_f64", |b| {
        b.iter(|| {
            black_box(divide(black_box(&a), black_box(&denominator)).unwrap())
        })
    });
    c.bench_function("greater_contiguous_f64", |b| {
        b.iter(|| {
            black_box(greater(black_box(&a), black_box(&denominator)).unwrap())
        })
    });
}

fn bench_cumulative(c: &mut Criterion) {
    let a = matrix(0);

    c.bench_function("cumsum_axis_last_contiguous_f64", |b| {
        b.iter(|| black_box(cumsum(black_box(&a), Some(1)).unwrap()))
    });

    c.bench_function("cumsum_axis_first_strided_f64", |b| {
        b.iter(|| black_box(cumsum(black_box(&a), Some(0)).unwrap()))
    });
}

fn bench_indexing(c: &mut Criterion) {
    let a = matrix(0);
    let fancy_len = ELEMENTS / 4;
    let rows: Vec<i64> = (0..fancy_len).map(|i| (i % SIDE) as i64).collect();
    let cols: Vec<i64> =
        (0..fancy_len).map(|i| ((i * 17) % SIDE) as i64).collect();
    let fancy = [
        IndexSpec::IntegerArray(
            Array::from_vec(rows.clone(), &[fancy_len]).unwrap(),
        ),
        IndexSpec::IntegerArray(
            Array::from_vec(cols.clone(), &[fancy_len]).unwrap(),
        ),
    ];
    let half_slice = [
        IndexSpec::slice(Some(0), Some((SIDE / 2) as i64), None),
        IndexSpec::full(),
    ];
    let reverse = [IndexSpec::slice(None, None, Some(-1)), IndexSpec::full()];
    let mask = Array::from_vec(
        (0..ELEMENTS).map(|i| i % 4 == 0).collect(),
        &[SIDE, SIDE],
    )
    .unwrap();
    let boolean_index = [IndexSpec::BoolArray(mask)];
    let strided_target =
        [IndexSpec::full(), IndexSpec::slice(Some(0), None, Some(2))];
    let fancy_target = [
        IndexSpec::IntegerArray(Array::from_vec(rows, &[fancy_len]).unwrap()),
        IndexSpec::IntegerArray(Array::from_vec(cols, &[fancy_len]).unwrap()),
    ];

    c.bench_function("fancy_gather_multidim_f64", |b| {
        b.iter(|| black_box(gather(black_box(&a), black_box(&fancy)).unwrap()))
    });
    c.bench_function("gather_basic_half_view_f64", |b| {
        b.iter(|| {
            black_box(gather(black_box(&a), black_box(&half_slice)).unwrap())
        })
    });
    c.bench_function("gather_reverse_view_f64", |b| {
        b.iter(|| {
            black_box(gather(black_box(&a), black_box(&reverse)).unwrap())
        })
    });
    c.bench_function("gather_boolean_mask_f64", |b| {
        b.iter(|| {
            black_box(gather(black_box(&a), black_box(&boolean_index)).unwrap())
        })
    });

    let target = [
        IndexSpec::slice(Some(0), Some((SIDE / 2) as i64), None),
        IndexSpec::full(),
    ];
    let source = [
        IndexSpec::slice(Some((SIDE / 2) as i64), None, None),
        IndexSpec::full(),
    ];
    let unshared_values =
        Array::from_vec(vec![3.5_f64; ELEMENTS / 2], &[SIDE / 2, SIDE])
            .unwrap();

    let mut group = c.benchmark_group("scatter_array_f64");
    group.bench_function("unshared_rhs", |b| {
        b.iter_batched(
            || (a.copy(), unshared_values.clone()),
            |(mut destination, values)| {
                scatter_array(
                    black_box(&mut destination),
                    black_box(&target),
                    black_box(&values),
                )
                .unwrap();
                black_box(destination)
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("shared_rhs", |b| {
        b.iter_batched(
            || {
                let destination = a.copy();
                let values = gather(&destination, &source).unwrap();
                (destination, values)
            },
            |(mut destination, values)| {
                scatter_array(
                    black_box(&mut destination),
                    black_box(&target),
                    black_box(&values),
                )
                .unwrap();
                black_box(destination)
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();

    c.bench_function("scatter_scalar_strided_f64", |b| {
        b.iter_batched(
            || a.copy(),
            |mut destination| {
                scatter(
                    black_box(&mut destination),
                    black_box(&strided_target),
                    black_box(3.5),
                )
                .unwrap();
                black_box(destination)
            },
            BatchSize::LargeInput,
        )
    });
    c.bench_function("scatter_scalar_fancy_f64", |b| {
        b.iter_batched(
            || a.copy(),
            |mut destination| {
                scatter(
                    black_box(&mut destination),
                    black_box(&fancy_target),
                    black_box(3.5),
                )
                .unwrap();
                black_box(destination)
            },
            BatchSize::LargeInput,
        )
    });
}

fn benchmark_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .without_plots()
}

criterion_group! {
    name = paths;
    config = benchmark_config();
    targets =
        bench_creation,
        bench_views_and_materialization,
        bench_reductions,
        bench_elementwise,
        bench_cumulative,
        bench_indexing
}
criterion_main!(paths);
