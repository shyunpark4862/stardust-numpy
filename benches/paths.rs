use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Criterion,
};
use sdnp::{
    absolute, add, all, any, argmax, argmin, argsort, broadcast_arrays, clip,
    concatenate, cumprod, cumsum, diag, divide, dot, eye, full, gather,
    greater, isnan, linspace, matmul, max as reduce_max, mean, meshgrid,
    min as reduce_min, multiply, ndenumerate, ndindex, nditer, negative,
    nonzero, ones, outer, power, prod, remainder, scatter, scatter_array,
    sort as sort_array, stack, std as reduce_std, subtract, sum, trace,
    tri_with, tril, triu, trunc_divide, unique, var, where_, zeros, Array,
    Complex64, IndexSpec, MeshgridIndexing, NanPolicy,
};

const SMALL_SIDE: usize = 32;
const LINALG_SIDE: usize = 128;
const MEDIUM_SIDE: usize = 256;
const SIDE: usize = 1024;
const ELEMENTS: usize = SIDE * SIDE;

fn shaped_f64(shape: &[usize], seed: usize) -> Array<f64> {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|i| ((i + seed) % 4096) as f64 * 0.25)
        .collect();
    Array::from_vec(data, shape).unwrap()
}

fn matrix(seed: usize) -> Array<f64> {
    shaped_f64(&[SIDE, SIDE], seed)
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
    let squeezable = a.reshape(&[1, SIDE as isize, SIDE as isize, 1]).unwrap();
    let complex = Array::from_vec(
        (0..ELEMENTS)
            .map(|i| Complex64::new(i as f64 * 0.25, 1.0))
            .collect(),
        &[SIDE, SIDE],
    )
    .unwrap();
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
    c.bench_function("squeeze_all_view_f64", |b| {
        b.iter(|| black_box(black_box(&squeezable).squeeze(None).unwrap()))
    });
    c.bench_function("squeeze_axis_view_f64", |b| {
        b.iter(|| {
            black_box(black_box(&squeezable).squeeze(Some(&[0, -1])).unwrap())
        })
    });
    c.bench_function("astype_contiguous_f64_to_i64", |b| {
        b.iter(|| black_box(black_box(&a).astype::<i64>().unwrap()))
    });
    c.bench_function("astype_strided_f64_to_i64", |b| {
        b.iter(|| black_box(black_box(&transposed).astype::<i64>().unwrap()))
    });
    c.bench_function("astype_same_f64", |b| {
        b.iter(|| black_box(black_box(&a).astype::<f64>().unwrap()))
    });
    c.bench_function("astype_complex_to_f64", |b| {
        b.iter(|| black_box(black_box(&complex).astype::<f64>().unwrap()))
    });
}

fn bench_reductions(c: &mut Criterion) {
    let a = matrix(0);
    let a_nonzero = matrix(1);
    let mut first_nan_data = a_nonzero.to_vec();
    for row in 0..SIDE {
        first_nan_data[row * SIDE] = f64::NAN;
    }
    let first_nan_input =
        Array::from_vec(first_nan_data, &[SIDE, SIDE]).unwrap();
    let mut sparse_nan_data = a_nonzero.to_vec();
    for row in (0..SIDE).step_by(16) {
        sparse_nan_data[row * SIDE + SIDE / 2] = f64::NAN;
    }
    let sparse_nan_input =
        Array::from_vec(sparse_nan_data, &[SIDE, SIDE]).unwrap();
    let axis_last = [1_isize];
    let axis_first = [0_isize];
    let product_data = (0..ELEMENTS)
        .map(|i| 1.0 + (i % 7) as f64 * 0.000_001)
        .collect();
    let product_input = Array::from_vec(product_data, &[SIDE, SIDE]).unwrap();
    let mut product_nan_data = product_input.to_vec();
    for row in (0..SIDE).step_by(16) {
        product_nan_data[row * SIDE + SIDE / 2] = f64::NAN;
    }
    let product_nan_input =
        Array::from_vec(product_nan_data, &[SIDE, SIDE]).unwrap();
    let integer_input = Array::from_vec(
        (0..ELEMENTS)
            .map(|i| ((i * 17 + 31) % 4096) as i64)
            .collect(),
        &[SIDE, SIDE],
    )
    .unwrap();
    let boolean_input = Array::from_vec(
        (0..ELEMENTS).map(|i| i % 3 != 0).collect(),
        &[SIDE, SIDE],
    )
    .unwrap();

    c.bench_function("sum_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                sum(
                    black_box(&a),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });

    c.bench_function("sum_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                sum(
                    black_box(&a),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("sum_total_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                sum(black_box(&a), None, false, sdnp::NanPolicy::Propagate)
                    .unwrap(),
            )
        })
    });

    let cube_data = (0..ELEMENTS).map(|i| (i % 2048) as f64 * 0.5).collect();
    let cube = Array::from_vec(cube_data, &[64, 128, 128]).unwrap();
    let permuted = cube.permute_axes(&[1, 0, 2]).unwrap();
    let mut cube_nan_data = cube.to_vec();
    for index in (0..cube_nan_data.len()).step_by(257) {
        cube_nan_data[index] = f64::NAN;
    }
    let cube_nan = Array::from_vec(cube_nan_data, &[64, 128, 128]).unwrap();
    let permuted_nan = cube_nan.permute_axes(&[1, 0, 2]).unwrap();
    let general_axes = [0_isize, 2_isize];

    c.bench_function("sum_multi_axis_general_f64", |b| {
        b.iter(|| {
            black_box(
                sum(
                    black_box(&permuted),
                    Some(&general_axes),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("sum_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                sum(
                    black_box(&sparse_nan_input),
                    Some(&axis_last),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("sum_ignore_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                sum(
                    black_box(&sparse_nan_input),
                    Some(&axis_first),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("sum_ignore_multi_axis_general_f64", |b| {
        b.iter(|| {
            black_box(
                sum(
                    black_box(&permuted_nan),
                    Some(&general_axes),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });

    c.bench_function("var_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                var(
                    black_box(&a),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("var_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                var(
                    black_box(&a),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("var_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                var(
                    black_box(&sparse_nan_input),
                    Some(&axis_last),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&a_nonzero),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("prod_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                prod(
                    black_box(&product_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("prod_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                prod(
                    black_box(&product_input),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("prod_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                prod(
                    black_box(&product_nan_input),
                    Some(&axis_last),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&a_nonzero),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&a_nonzero),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&sparse_nan_input),
                    Some(&axis_last),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_last_first_nan_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&first_nan_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_last_first_nan_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&first_nan_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_last_sparse_nan_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&sparse_nan_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_last_sparse_nan_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&sparse_nan_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&a_nonzero),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_last_contiguous_i64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&integer_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_last_contiguous_i64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&integer_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_first_fixed_stride_i64", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&integer_input),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_first_fixed_stride_i64", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&integer_input),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_last_contiguous_bool", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&boolean_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_last_contiguous_bool", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&boolean_input),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("min_axis_first_fixed_stride_bool", |b| {
        b.iter(|| {
            black_box(
                reduce_min(
                    black_box(&boolean_input),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("max_axis_first_fixed_stride_bool", |b| {
        b.iter(|| {
            black_box(
                reduce_max(
                    black_box(&boolean_input),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("mean_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                mean(
                    black_box(&a_nonzero),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("mean_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                mean(
                    black_box(&a_nonzero),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("mean_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                mean(
                    black_box(&sparse_nan_input),
                    Some(&axis_last),
                    false,
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("std_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_std(
                    black_box(&a_nonzero),
                    Some(&axis_last),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("std_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                reduce_std(
                    black_box(&a_nonzero),
                    Some(&axis_first),
                    false,
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("argmax_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                argmax(
                    black_box(&a_nonzero),
                    Some(1),
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("argmin_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                argmin(
                    black_box(&a_nonzero),
                    Some(1),
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("argmin_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                argmin(
                    black_box(&sparse_nan_input),
                    Some(1),
                    NanPolicy::Ignore,
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("any_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                any(black_box(&a_nonzero), Some(&axis_last), false).unwrap(),
            )
        })
    });
    c.bench_function("any_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                any(black_box(&a_nonzero), Some(&axis_first), false).unwrap(),
            )
        })
    });
    c.bench_function("any_axis_last_contiguous_bool", |b| {
        b.iter(|| {
            black_box(
                any(black_box(&boolean_input), Some(&axis_last), false)
                    .unwrap(),
            )
        })
    });
    c.bench_function("all_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                all(black_box(&a_nonzero), Some(&axis_last), false).unwrap(),
            )
        })
    });
    c.bench_function("all_axis_first_fixed_stride_f64", |b| {
        b.iter(|| {
            black_box(
                all(black_box(&a_nonzero), Some(&axis_first), false).unwrap(),
            )
        })
    });
    c.bench_function("cumprod_axis_last_f64", |b| {
        b.iter(|| {
            black_box(
                cumprod(
                    black_box(&product_input),
                    Some(1),
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
}

fn bench_elementwise(c: &mut Criterion) {
    let left = matrix(0);
    let right = matrix(17);
    let small_left = shaped_f64(&[SMALL_SIDE, SMALL_SIDE], 0);
    let small_right = shaped_f64(&[SMALL_SIDE, SMALL_SIDE], 17);
    let medium_left = shaped_f64(&[MEDIUM_SIDE, MEDIUM_SIDE], 0);
    let medium_right = shaped_f64(&[MEDIUM_SIDE, MEDIUM_SIDE], 17);
    let one_d_left = shaped_f64(&[ELEMENTS], 0);
    let one_d_right = shaped_f64(&[ELEMENTS], 17);
    let three_d_left = shaped_f64(&[64, 128, 128], 0);
    let three_d_right = shaped_f64(&[64, 128, 128], 17);
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
    let column = Array::from_vec(
        (0..SIDE).map(|i| (i + 1) as f64).collect(),
        &[SIDE, 1],
    )
    .unwrap();
    let scalar = Array::from_vec(vec![2.0_f64], &[1, 1]).unwrap();
    let integer_left = Array::from_vec(
        (0..ELEMENTS).map(|i| (i % 4096) as i64).collect(),
        &[SIDE, SIDE],
    )
    .unwrap();
    let integer_denominator = Array::from_vec(
        (0..ELEMENTS).map(|i| (i % 31 + 1) as i64).collect(),
        &[SIDE, SIDE],
    )
    .unwrap();

    c.bench_function("add_contiguous_contiguous_32x32_f64", |b| {
        b.iter(|| {
            black_box(
                add(black_box(&small_left), black_box(&small_right)).unwrap(),
            )
        })
    });
    c.bench_function("add_contiguous_contiguous_256x256_f64", |b| {
        b.iter(|| {
            black_box(
                add(black_box(&medium_left), black_box(&medium_right)).unwrap(),
            )
        })
    });
    c.bench_function("add_contiguous_contiguous_f64", |b| {
        b.iter(|| black_box(add(black_box(&left), black_box(&right)).unwrap()))
    });
    c.bench_function("add_contiguous_contiguous_1m_1d_f64", |b| {
        b.iter(|| {
            black_box(
                add(black_box(&one_d_left), black_box(&one_d_right)).unwrap(),
            )
        })
    });
    c.bench_function("add_contiguous_contiguous_64x128x128_f64", |b| {
        b.iter(|| {
            black_box(
                add(black_box(&three_d_left), black_box(&three_d_right))
                    .unwrap(),
            )
        })
    });
    c.bench_function("subtract_contiguous_contiguous_f64", |b| {
        b.iter(|| {
            black_box(subtract(black_box(&left), black_box(&right)).unwrap())
        })
    });

    let left_t = left.transpose();
    let right_t = right.transpose();
    let integer_left_t = integer_left.transpose();
    let integer_denominator_t = integer_denominator.transpose();
    c.bench_function("add_strided_strided_f64", |b| {
        b.iter(|| {
            black_box(add(black_box(&left_t), black_box(&right_t)).unwrap())
        })
    });
    c.bench_function("add_contiguous_strided_f64", |b| {
        b.iter(|| {
            black_box(add(black_box(&left), black_box(&right_t)).unwrap())
        })
    });
    c.bench_function("subtract_strided_strided_f64", |b| {
        b.iter(|| {
            black_box(
                subtract(black_box(&left_t), black_box(&right_t)).unwrap(),
            )
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
    c.bench_function("multiply_column_broadcast_f64", |b| {
        b.iter(|| {
            black_box(multiply(black_box(&a), black_box(&column)).unwrap())
        })
    });
    c.bench_function("multiply_scalar_broadcast_f64", |b| {
        b.iter(|| {
            black_box(multiply(black_box(&a), black_box(&scalar)).unwrap())
        })
    });
    c.bench_function("add_i64_f64_contiguous", |b| {
        b.iter(|| {
            black_box(add(black_box(&integer_left), black_box(&a)).unwrap())
        })
    });
    c.bench_function("divide_i64_contiguous", |b| {
        b.iter(|| {
            black_box(
                divide(
                    black_box(&integer_left),
                    black_box(&integer_denominator),
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("divide_i64_strided", |b| {
        b.iter(|| {
            black_box(
                divide(
                    black_box(&integer_left_t),
                    black_box(&integer_denominator_t),
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("divide_contiguous_f64", |b| {
        b.iter(|| {
            black_box(divide(black_box(&a), black_box(&denominator)).unwrap())
        })
    });
    c.bench_function("trunc_divide_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                trunc_divide(black_box(&a), black_box(&denominator)).unwrap(),
            )
        })
    });
    c.bench_function("remainder_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                remainder(black_box(&a), black_box(&denominator)).unwrap(),
            )
        })
    });
    c.bench_function("power_contiguous_f64", |b| {
        b.iter(|| {
            black_box(power(black_box(&a), black_box(&denominator)).unwrap())
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
    let mut nan_data = a.to_vec();
    for index in (0..ELEMENTS).step_by(257) {
        nan_data[index] = f64::NAN;
    }
    let nan_input = Array::from_vec(nan_data, &[SIDE, SIDE]).unwrap();
    let product_input = Array::from_vec(
        (0..ELEMENTS)
            .map(|i| 1.0 + (i % 7) as f64 * 0.000_001)
            .collect(),
        &[SIDE, SIDE],
    )
    .unwrap();

    c.bench_function("cumsum_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                cumsum(black_box(&a), Some(1), sdnp::NanPolicy::Propagate)
                    .unwrap(),
            )
        })
    });

    c.bench_function("cumsum_axis_first_strided_f64", |b| {
        b.iter(|| {
            black_box(
                cumsum(black_box(&a), Some(0), sdnp::NanPolicy::Propagate)
                    .unwrap(),
            )
        })
    });
    c.bench_function("cumsum_ignore_axis_last_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                cumsum(black_box(&nan_input), Some(1), NanPolicy::Ignore)
                    .unwrap(),
            )
        })
    });
    c.bench_function("cumsum_ignore_axis_first_strided_f64", |b| {
        b.iter(|| {
            black_box(
                cumsum(black_box(&nan_input), Some(0), NanPolicy::Ignore)
                    .unwrap(),
            )
        })
    });
    c.bench_function("cumprod_axis_first_strided_f64", |b| {
        b.iter(|| {
            black_box(
                cumprod(
                    black_box(&product_input),
                    Some(0),
                    sdnp::NanPolicy::Propagate,
                )
                .unwrap(),
            )
        })
    });
}

fn bench_phase5(c: &mut Criterion) {
    let left = matrix(0);
    let right = matrix(17);
    let left_t = left.transpose();
    let right_t = right.transpose();
    let condition = Array::from_vec(
        (0..ELEMENTS).map(|i| i % 2 == 0).collect(),
        &[SIDE, SIDE],
    )
    .unwrap();
    let condition_t = condition.transpose();
    let scalar = Array::from_vec(vec![3.5_f64], &[1, 1]).unwrap();
    let x = Array::from_vec((0..SIDE).map(|i| i as f64).collect(), &[SIDE])
        .unwrap();
    let y =
        Array::from_vec((0..SIDE).map(|i| (i * 2) as f64).collect(), &[SIDE])
            .unwrap();

    c.bench_function("concatenate_axis0_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                concatenate(&[black_box(&left), black_box(&right)], 0).unwrap(),
            )
        })
    });
    c.bench_function("concatenate_axis0_strided_f64", |b| {
        b.iter(|| {
            black_box(
                concatenate(&[black_box(&left_t), black_box(&right_t)], 0)
                    .unwrap(),
            )
        })
    });
    c.bench_function("concatenate_axis1_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                concatenate(&[black_box(&left), black_box(&right)], 1).unwrap(),
            )
        })
    });
    c.bench_function("concatenate_axis1_strided_f64", |b| {
        b.iter(|| {
            black_box(
                concatenate(&[black_box(&left_t), black_box(&right_t)], 1)
                    .unwrap(),
            )
        })
    });
    c.bench_function("stack_axis0_contiguous_f64", |b| {
        b.iter(|| {
            black_box(stack(&[black_box(&left), black_box(&right)], 0).unwrap())
        })
    });
    c.bench_function("where_contiguous_f64", |b| {
        b.iter(|| {
            black_box(
                where_(
                    black_box(&condition),
                    black_box(&left),
                    black_box(&right),
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("where_strided_f64", |b| {
        b.iter(|| {
            black_box(
                where_(
                    black_box(&condition_t),
                    black_box(&left_t),
                    black_box(&right_t),
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("where_scalar_broadcast_f64", |b| {
        b.iter(|| {
            black_box(
                where_(
                    black_box(&condition),
                    black_box(&scalar),
                    black_box(&right),
                )
                .unwrap(),
            )
        })
    });
    c.bench_function("clip_contiguous_f64", |b| {
        b.iter(|| {
            black_box(clip(black_box(&left), Some(128.0), Some(896.0)).unwrap())
        })
    });
    c.bench_function("nonzero_bool_contiguous", |b| {
        b.iter(|| black_box(nonzero(black_box(&condition)).unwrap()))
    });
    c.bench_function("sort_axis_last_contiguous_f64", |b| {
        b.iter(|| black_box(sort_array(black_box(&left), Some(-1)).unwrap()))
    });
    c.bench_function("sort_axis_last_strided_f64", |b| {
        b.iter(|| black_box(sort_array(black_box(&left_t), Some(-1)).unwrap()))
    });
    c.bench_function("argsort_axis_last_contiguous_f64", |b| {
        b.iter(|| black_box(argsort(black_box(&left), Some(-1)).unwrap()))
    });
    c.bench_function("unique_flatten_f64", |b| {
        b.iter(|| black_box(unique(black_box(&left)).unwrap()))
    });
    c.bench_function("linspace_1m_f64", |b| {
        b.iter(|| {
            black_box(
                linspace(black_box(0.0), black_box(1.0), ELEMENTS, true)
                    .unwrap(),
            )
        })
    });
    c.bench_function("meshgrid_1024x1024_view_f64", |b| {
        b.iter(|| {
            black_box(
                meshgrid(&[black_box(&x), black_box(&y)], MeshgridIndexing::Xy)
                    .unwrap(),
            )
        })
    });
}

fn bench_phase6(c: &mut Criterion) {
    let vector_left = shaped_f64(&[ELEMENTS], 0);
    let vector_right = shaped_f64(&[ELEMENTS], 17);
    let outer_left = shaped_f64(&[SIDE], 0);
    let outer_right = shaped_f64(&[SIDE], 17);
    let small_left = shaped_f64(&[SMALL_SIDE, SMALL_SIDE], 0);
    let small_right = shaped_f64(&[SMALL_SIDE, SMALL_SIDE], 17);
    let linalg_left = shaped_f64(&[LINALG_SIDE, LINALG_SIDE], 0);
    let linalg_right = shaped_f64(&[LINALG_SIDE, LINALG_SIDE], 17);
    let linalg_left_t = linalg_left.transpose();
    let linalg_right_t = linalg_right.transpose();
    let medium_left = shaped_f64(&[MEDIUM_SIDE, MEDIUM_SIDE], 0);
    let medium_right = shaped_f64(&[MEDIUM_SIDE, MEDIUM_SIDE], 17);
    let batch_left = shaped_f64(&[8, 64, 64], 0);
    let batch_right = shaped_f64(&[8, 64, 64], 17);
    let large = matrix(0);
    let large_t = large.transpose();
    let diagonal_source = shaped_f64(&[SIDE], 0);

    c.bench_function("dot_1d_contiguous_1m_f64", |b| {
        b.iter(|| {
            black_box(
                dot(black_box(&vector_left), black_box(&vector_right)).unwrap(),
            )
        })
    });
    c.bench_function("outer_1024x1024_f64", |b| {
        b.iter(|| {
            black_box(
                outer(black_box(&outer_left), black_box(&outer_right)).unwrap(),
            )
        })
    });
    c.bench_function("matmul_contiguous_32x32_f64", |b| {
        b.iter(|| {
            black_box(
                matmul(black_box(&small_left), black_box(&small_right))
                    .unwrap(),
            )
        })
    });
    c.bench_function("matmul_contiguous_128x128_f64", |b| {
        b.iter(|| {
            black_box(
                matmul(black_box(&linalg_left), black_box(&linalg_right))
                    .unwrap(),
            )
        })
    });
    c.bench_function("matmul_contiguous_256x256_f64", |b| {
        b.iter(|| {
            black_box(
                matmul(black_box(&medium_left), black_box(&medium_right))
                    .unwrap(),
            )
        })
    });
    c.bench_function("matmul_strided_strided_128x128_f64", |b| {
        b.iter(|| {
            black_box(
                matmul(black_box(&linalg_left_t), black_box(&linalg_right_t))
                    .unwrap(),
            )
        })
    });
    c.bench_function("matmul_batched_8x64x64_f64", |b| {
        b.iter(|| {
            black_box(
                matmul(black_box(&batch_left), black_box(&batch_right))
                    .unwrap(),
            )
        })
    });
    c.bench_function("trace_1024x1024_f64", |b| {
        b.iter(|| black_box(trace(black_box(&large), 0, 0, 1).unwrap()))
    });
    c.bench_function("tri_1024x1024_f64", |b| {
        b.iter(|| black_box(tri_with::<f64>(SIDE, SIDE, 0).unwrap()))
    });
    c.bench_function("tril_1024x1024_contiguous_f64", |b| {
        b.iter(|| black_box(tril(black_box(&large), 0).unwrap()))
    });
    c.bench_function("tril_1024x1024_strided_f64", |b| {
        b.iter(|| black_box(tril(black_box(&large_t), 0).unwrap()))
    });
    c.bench_function("triu_1024x1024_contiguous_f64", |b| {
        b.iter(|| black_box(triu(black_box(&large), 0).unwrap()))
    });
    c.bench_function("diag_extract_1024x1024_f64", |b| {
        b.iter(|| black_box(diag(black_box(&large), 0).unwrap()))
    });
    c.bench_function("diag_build_1024_f64", |b| {
        b.iter(|| black_box(diag(black_box(&diagonal_source), 0).unwrap()))
    });
}

fn bench_phase7(c: &mut Criterion) {
    let contiguous = matrix(0);
    let strided = contiguous.transpose();
    let column = shaped_f64(&[SIDE, 1], 0);
    let row = shaped_f64(&[1, SIDE], 17);
    let shape = [SIDE, SIDE];

    c.bench_function("ndindex_1024x1024", |b| {
        b.iter(|| {
            let checksum = ndindex(black_box(&shape))
                .unwrap()
                .fold(0_usize, |sum, index| {
                    sum.wrapping_add(black_box(index[0] + index[1]))
                });
            black_box(checksum)
        })
    });
    c.bench_function("ndenumerate_contiguous_1m_f64", |b| {
        b.iter(|| {
            let checksum = ndenumerate(black_box(&contiguous)).fold(
                0.0_f64,
                |sum, (index, value)| {
                    sum + black_box(value + (index[0] + index[1]) as f64)
                },
            );
            black_box(checksum)
        })
    });
    c.bench_function("flat_contiguous_1m_f64", |b| {
        b.iter(|| {
            black_box(
                black_box(&contiguous)
                    .flat()
                    .fold(0.0_f64, |sum, value| sum + black_box(value)),
            )
        })
    });
    c.bench_function("flat_strided_1m_f64", |b| {
        b.iter(|| {
            black_box(
                black_box(&strided)
                    .flat()
                    .fold(0.0_f64, |sum, value| sum + black_box(value)),
            )
        })
    });
    c.bench_function("nditer_broadcast_1024x1024_f64", |b| {
        b.iter(|| {
            let checksum = nditer(black_box(&[&column, &row]))
                .unwrap()
                .fold(0.0_f64, |sum, values| {
                    sum + black_box(values[0] + values[1])
                });
            black_box(checksum)
        })
    });
    c.bench_function("axis0_iter_1024x1024_f64", |b| {
        b.iter(|| {
            let checksum = black_box(&contiguous)
                .iter_axis0()
                .fold(0_usize, |sum, row| sum + black_box(row.size()));
            black_box(checksum)
        })
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
    let strided_values =
        Array::from_vec(vec![3.5_f64; ELEMENTS / 2], &[SIDE, SIDE / 2])
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

    c.bench_function("scatter_array_strided_basic_f64", |b| {
        b.iter_batched(
            || a.copy(),
            |mut destination| {
                scatter_array(
                    black_box(&mut destination),
                    black_box(&strided_target),
                    black_box(&strided_values),
                )
                .unwrap();
                black_box(destination)
            },
            BatchSize::LargeInput,
        )
    });
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
        bench_phase5,
        bench_phase6,
        bench_phase7,
        bench_indexing
}
criterion_main!(paths);
