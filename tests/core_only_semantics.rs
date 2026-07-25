use sdnp::{
    add, concatenate, diag, diagonal, dot, matmul, outer, stack, trace, tril,
    vdot, Array,
};

#[test]
fn rust_core_boolean_arithmetic_is_explicitly_available() {
    let left = Array::from_slice(&[true, false, true], &[3]).unwrap();
    let right = Array::from_slice(&[false, false, true], &[3]).unwrap();
    assert_eq!(
        add(&left, &right).unwrap().to_vec(),
        vec![true, false, true]
    );
}

#[test]
fn boolean_matrix_multiplication_uses_or_and_semiring() {
    let left = Array::from_slice(&[true, false, false, true], &[2, 2]).unwrap();
    let right =
        Array::from_slice(&[false, true, true, false], &[2, 2]).unwrap();
    let result = matmul(&left, &right).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![false, true, true, false]);
}

#[test]
fn vector_contractions_return_core_zero_dimensional_arrays() {
    let left = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    let right = Array::from_slice(&[4_i64, 5, 6], &[3]).unwrap();

    let dot_result = dot(&left, &right).unwrap();
    let vdot_result = vdot(&left, &right).unwrap();
    assert_eq!(dot_result.shape(), &[]);
    assert_eq!(vdot_result.shape(), &[]);
    assert_eq!(dot_result.item().unwrap(), 32);
    assert_eq!(vdot_result.item().unwrap(), 32);
}

#[test]
fn batched_and_empty_contractions_preserve_geometry() {
    let left = Array::from_vec((0_i64..12).collect(), &[2, 2, 3]).unwrap();
    let right = Array::from_vec((0_i64..12).collect(), &[2, 3, 2]).unwrap();
    let result = matmul(&left, &right).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result.to_vec(), vec![10, 13, 28, 40, 172, 193, 244, 274]);

    let empty_left = Array::from_vec(Vec::<f64>::new(), &[2, 0]).unwrap();
    let empty_right = Array::from_vec(Vec::<f64>::new(), &[0, 3]).unwrap();
    let empty_result = matmul(&empty_left, &empty_right).unwrap();
    assert_eq!(empty_result.shape(), &[2, 3]);
    assert_eq!(empty_result.to_vec(), vec![0.0; 6]);
}

#[test]
fn diagonal_trace_and_diag_cover_strided_and_empty_geometry() {
    let matrix = Array::from_vec((0_i64..12).collect(), &[3, 4]).unwrap();
    let transposed = matrix.transpose();

    assert_eq!(diagonal(&transposed, 1, 0, 1).unwrap().to_vec(), vec![4, 9]);
    assert_eq!(trace(&transposed, 0, 0, 1).unwrap().item().unwrap(), 15);

    let vector = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    let diagonal_matrix = diag(&vector, -1).unwrap();
    assert_eq!(diagonal_matrix.shape(), &[4, 4]);
    assert_eq!(
        diagonal_matrix.to_vec(),
        vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0]
    );
}

#[test]
fn triangular_copy_handles_vector_and_transposed_inputs() {
    let vector = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert_eq!(
        tril(&vector, 0).unwrap().to_vec(),
        vec![1, 0, 0, 1, 2, 0, 1, 2, 3]
    );

    let matrix = Array::from_vec((1_i64..=9).collect(), &[3, 3]).unwrap();
    let result = tril(&matrix.transpose(), -1).unwrap();
    assert_eq!(result.to_vec(), vec![0, 0, 0, 2, 0, 0, 3, 6, 0]);
}

#[test]
fn joining_zero_dimensional_core_arrays_is_well_defined() {
    let first = Array::from_slice(&[1_i64], &[]).unwrap();
    let second = Array::from_slice(&[2_i64], &[]).unwrap();

    let stacked = stack(&[&first, &second], 0).unwrap();
    assert_eq!(stacked.shape(), &[2]);
    assert_eq!(stacked.to_vec(), vec![1, 2]);

    let left = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
    let right = Array::from_slice(&[3_i64], &[1]).unwrap();
    assert_eq!(
        concatenate(&[&left, &right], 0).unwrap().to_vec(),
        vec![1, 2, 3]
    );
}

#[test]
fn outer_flattens_noncontiguous_inputs_in_logical_order() {
    let left = Array::from_vec((1_i64..=6).collect(), &[2, 3])
        .unwrap()
        .transpose();
    let right = Array::from_slice(&[2_i64, 4], &[2]).unwrap();
    let result = outer(&left, &right).unwrap();
    assert_eq!(result.shape(), &[6, 2]);
    assert_eq!(
        result.to_vec(),
        vec![2, 4, 8, 16, 4, 8, 10, 20, 6, 12, 12, 24]
    );
}
