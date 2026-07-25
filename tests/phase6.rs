//! Phase 6: contractions, diagonals, traces, and triangular arrays.

use sdnp::{
    diag, diagonal, dot, matmul, outer, trace, tri, tri_with, tril, triu, vdot,
    Array, Complex64, Error,
};

#[test]
fn dot_supports_all_vector_and_matrix_combinations() {
    let vector = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    let other = Array::from_slice(&[4_i64, 5, 6], &[3]).unwrap();
    assert_eq!(dot(&vector, &other).unwrap().item().unwrap(), 32);

    let matrix = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    assert_eq!(dot(&matrix, &other).unwrap().to_vec(), vec![32, 77]);

    let row = Array::from_slice(&[10_i64, 20], &[2]).unwrap();
    assert_eq!(dot(&row, &matrix).unwrap().to_vec(), vec![90, 120, 150]);

    let right =
        Array::from_slice(&[10_i64, 20, 30, 40, 50, 60], &[3, 2]).unwrap();
    let product = dot(&matrix, &right).unwrap();
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.to_vec(), vec![220, 280, 490, 640]);
}

#[test]
fn dot_promotes_dtype_and_handles_strided_operands() {
    let left = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[3, 2])
        .unwrap()
        .transpose();
    let right = Array::from_slice(&[10.0_f64, 20.0, 30.0], &[3]).unwrap();
    assert_eq!(dot(&left, &right).unwrap().to_vec(), vec![220.0, 280.0]);

    let right_matrix =
        Array::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
            .unwrap()
            .transpose();
    let vector = Array::from_slice(&[10_i64, 20], &[2]).unwrap();
    assert_eq!(
        dot(&vector, &right_matrix).unwrap().to_vec(),
        vec![50.0, 110.0, 170.0]
    );

    let boolean_left = Array::from_slice(&[true, false, true], &[3]).unwrap();
    let boolean_right = Array::from_slice(&[false, true, true], &[3]).unwrap();
    assert!(dot(&boolean_left, &boolean_right).unwrap().item().unwrap());

    let empty_left = Array::from_vec(Vec::<f64>::new(), &[2, 0]).unwrap();
    let empty_right = Array::from_vec(Vec::<f64>::new(), &[0, 3]).unwrap();
    assert_eq!(
        dot(&empty_left, &empty_right).unwrap().to_vec(),
        vec![0.0; 6]
    );
}

#[test]
fn dot_validates_rank_and_inner_dimensions() {
    let short = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
    let long = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert!(matches!(dot(&short, &long), Err(Error::InvalidArgument(_))));

    let cube = Array::from_vec(vec![1_i64; 8], &[2, 2, 2]).unwrap();
    assert!(matches!(dot(&cube, &cube), Err(Error::InvalidArgument(_))));

    let scalar = Array::from_slice(&[2_i64], &[]).unwrap();
    assert!(matches!(
        dot(&scalar, &short),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn matmul_broadcasts_batches_and_vector_axes() {
    let left = Array::from_slice(
        &[
            1_i64, 2, 3, 4, 5, 6, //
            7, 8, 9, 10, 11, 12,
        ],
        &[2, 2, 3],
    )
    .unwrap();
    let right = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[3, 2]).unwrap();
    let product = matmul(&left, &right).unwrap();
    assert_eq!(product.shape(), &[2, 2, 2]);
    assert_eq!(product.to_vec(), vec![22, 28, 49, 64, 76, 100, 103, 136]);

    let vector = Array::from_slice(&[10_i64, 20, 30], &[3]).unwrap();
    let matrix_vector = matmul(&left, &vector).unwrap();
    assert_eq!(matrix_vector.shape(), &[2, 2]);
    assert_eq!(matrix_vector.to_vec(), vec![140, 320, 500, 680]);

    let left_vector = Array::from_slice(&[10_i64, 20], &[2]).unwrap();
    let batched_right = Array::from_slice(
        &[
            1_i64, 2, 3, 4, 5, 6, //
            7, 8, 9, 10, 11, 12,
        ],
        &[2, 2, 3],
    )
    .unwrap();
    let vector_matrix = matmul(&left_vector, &batched_right).unwrap();
    assert_eq!(vector_matrix.shape(), &[2, 3]);
    assert_eq!(vector_matrix.to_vec(), vec![90, 120, 150, 270, 300, 330]);
}

#[test]
fn matmul_broadcasts_both_leading_shapes() {
    let left = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6, 7, 8], &[2, 1, 2, 2])
        .unwrap();
    let right = Array::from_slice(
        &[
            10_i64, 20, 30, 40, //
            50, 60, 70, 80, //
            90, 100, 110, 120,
        ],
        &[1, 3, 2, 2],
    )
    .unwrap();
    let product = matmul(&left, &right).unwrap();
    assert_eq!(product.shape(), &[2, 3, 2, 2]);
    assert_eq!(
        product.to_vec(),
        vec![
            70, 100, 150, 220, 190, 220, 430, 500, 310, 340, 710, 780, 230,
            340, 310, 460, 670, 780, 910, 1060, 1110, 1220, 1510, 1660,
        ]
    );
}

#[test]
fn matmul_rejects_scalars_and_incompatible_batches() {
    let scalar = Array::from_slice(&[3_i64], &[]).unwrap();
    let matrix = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    assert!(matches!(
        matmul(&matrix, &scalar),
        Err(Error::InvalidArgument(_))
    ));

    let left = Array::from_vec(vec![1_i64; 8], &[2, 2, 2]).unwrap();
    let right = Array::from_vec(vec![1_i64; 12], &[3, 2, 2]).unwrap();
    assert!(matches!(
        matmul(&left, &right),
        Err(Error::Broadcast { .. })
    ));
}

#[test]
fn vdot_conjugates_first_and_outer_flattens_logically() {
    let left = Array::from_slice(
        &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        &[2],
    )
    .unwrap();
    let right = Array::from_slice(
        &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
        &[2],
    )
    .unwrap();
    assert_eq!(
        vdot(&left, &right).unwrap().item().unwrap(),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(
        dot(&left, &right).unwrap().item().unwrap(),
        Complex64::new(3.0, 4.0)
    );

    let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2])
        .unwrap()
        .transpose();
    let b = Array::from_slice(&[10_i64, 20], &[2]).unwrap();
    let result = outer(&a, &b).unwrap();
    assert_eq!(result.shape(), &[4, 2]);
    assert_eq!(result.to_vec(), vec![10, 20, 30, 60, 20, 40, 40, 80]);

    let wrong_size = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert!(matches!(
        vdot(&b, &wrong_size),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn diagonal_and_trace_support_nd_axes_and_offsets() {
    let matrix = Array::from_vec((0_i64..9).collect(), &[3, 3]).unwrap();
    assert_eq!(diagonal(&matrix, 0, 0, 1).unwrap().to_vec(), vec![0, 4, 8]);
    assert_eq!(diagonal(&matrix, 1, 0, 1).unwrap().to_vec(), vec![1, 5]);
    assert_eq!(trace(&matrix, 0, 0, 1).unwrap().item().unwrap(), 12);

    let cube = Array::from_vec((0_i64..8).collect(), &[2, 2, 2]).unwrap();
    let extracted = diagonal(&cube, 0, 0, 1).unwrap();
    assert_eq!(extracted.shape(), &[2, 2]);
    assert_eq!(extracted.to_vec(), vec![0, 6, 1, 7]);
    assert_eq!(trace(&cube, 0, 0, 1).unwrap().to_vec(), vec![6, 8]);

    let four_d = Array::from_vec((0_i64..24).collect(), &[2, 2, 2, 3]).unwrap();
    let traced = trace(&four_d, 0, -4, -3).unwrap();
    assert_eq!(traced.shape(), &[2, 3]);
    assert_eq!(traced.to_vec(), vec![18, 20, 22, 24, 26, 28]);

    assert!(matches!(
        diagonal(&matrix, 0, 0, 0),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn trace_uses_sum_accumulator_and_empty_identity() {
    let boolean =
        Array::from_slice(&[true, false, false, true], &[2, 2]).unwrap();
    let result = trace(&boolean, 0, 0, 1).unwrap();
    assert_eq!(result.item().unwrap(), 2_i64);

    let empty = Array::from_vec(Vec::<f64>::new(), &[0, 3]).unwrap();
    assert_eq!(trace(&empty, 0, 0, 1).unwrap().item().unwrap(), 0.0);
}

#[test]
fn tri_and_diag_cover_rectangular_offset_and_empty_shapes() {
    assert_eq!(
        tri::<i64>(3).unwrap().to_vec(),
        vec![1, 0, 0, 1, 1, 0, 1, 1, 1]
    );
    let rectangular = tri_with::<i64>(3, 5, -1).unwrap();
    assert_eq!(
        rectangular.to_vec(),
        vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    );

    let vector = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    let built = diag(&vector, 1).unwrap();
    assert_eq!(built.shape(), &[4, 4]);
    assert_eq!(
        built.to_vec(),
        vec![0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0]
    );
    assert_eq!(diag(&built, 1).unwrap().to_vec(), vec![1, 2, 3]);

    let empty = Array::from_vec(Vec::<i64>::new(), &[0]).unwrap();
    assert_eq!(diag(&empty, 0).unwrap().shape(), &[0, 0]);
}

#[test]
fn tril_and_triu_apply_to_vectors_and_trailing_axes() {
    let vector = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert_eq!(
        tril(&vector, 0).unwrap().to_vec(),
        vec![1, 0, 0, 1, 2, 0, 1, 2, 3]
    );
    assert_eq!(
        triu(&vector, 0).unwrap().to_vec(),
        vec![1, 2, 3, 0, 2, 3, 0, 0, 3]
    );

    let cube = Array::from_vec((0_i64..24).collect(), &[2, 3, 4]).unwrap();
    let lower = tril(&cube, 0).unwrap();
    assert_eq!(lower.get(&[0, 1, 2]).unwrap(), 0);
    assert_eq!(lower.get(&[0, 1, 1]).unwrap(), 5);
    assert_eq!(lower.get(&[1, 0, 0]).unwrap(), 12);
    assert!(!cube.shares_buffer_with(&lower));
}

#[test]
fn diagonal_family_handles_noncontiguous_inputs_and_rank_errors() {
    let matrix = Array::from_vec((0_i64..9).collect(), &[3, 3])
        .unwrap()
        .transpose();
    assert_eq!(diag(&matrix, 1).unwrap().to_vec(), vec![3, 7]);
    assert_eq!(
        tril(&matrix, 0).unwrap().to_vec(),
        vec![0, 0, 0, 1, 4, 0, 2, 5, 8]
    );

    let scalar = Array::from_slice(&[1_i64], &[]).unwrap();
    assert!(matches!(tril(&scalar, 0), Err(Error::InvalidArgument(_))));
    let cube = Array::from_vec(vec![1_i64; 8], &[2, 2, 2]).unwrap();
    assert!(matches!(diag(&cube, 0), Err(Error::InvalidArgument(_))));
}
