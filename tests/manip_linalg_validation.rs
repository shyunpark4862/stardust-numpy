use sdnp::{
    concatenate, diagonal, dot, hstack, matmul, stack, vdot, vstack, Array,
    Error,
};

#[test]
fn concatenate_and_stack_reject_invalid_operand_geometry() {
    assert!(matches!(
        concatenate::<i64>(&[], 0),
        Err(Error::EmptyOperands { op: "concatenate" })
    ));
    assert!(matches!(
        stack::<i64>(&[], 0),
        Err(Error::EmptyOperands { op: "stack" })
    ));

    let scalar = Array::from_slice(&[1_i64], &[]).unwrap();
    assert!(matches!(
        concatenate(&[&scalar], 0),
        Err(Error::InvalidRank { .. })
    ));

    let matrix = Array::from_vec(vec![1_i64; 6], &[2, 3]).unwrap();
    let vector = Array::from_vec(vec![1_i64; 3], &[3]).unwrap();
    assert!(matches!(
        concatenate(&[&matrix, &vector], 0),
        Err(Error::RankMismatch { .. })
    ));

    let wrong_width = Array::from_vec(vec![1_i64; 8], &[2, 4]).unwrap();
    assert!(matches!(
        concatenate(&[&matrix, &wrong_width], 0),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        stack(&[&matrix, &wrong_width], 0),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn promoted_stack_variants_validate_every_operand() {
    assert!(matches!(
        vstack::<i64>(&[]),
        Err(Error::EmptyOperands { op: "vstack" })
    ));
    assert!(matches!(
        hstack::<i64>(&[]),
        Err(Error::EmptyOperands { op: "hstack" })
    ));

    let vector = Array::from_vec(vec![1_i64; 3], &[3]).unwrap();
    let wrong_vector = Array::from_vec(vec![1_i64; 4], &[4]).unwrap();
    assert!(matches!(
        vstack(&[&vector, &wrong_vector]),
        Err(Error::ShapeMismatch { .. })
    ));

    let matrix = Array::from_vec(vec![1_i64; 6], &[2, 3]).unwrap();
    let wrong_height = Array::from_vec(vec![1_i64; 9], &[3, 3]).unwrap();
    assert!(matches!(
        hstack(&[&matrix, &wrong_height]),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn contraction_plans_reject_invalid_geometry() {
    let scalar = Array::from_slice(&[1_i64], &[]).unwrap();
    let vector = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
    assert!(matches!(
        matmul(&scalar, &vector),
        Err(Error::InvalidRank { op: "matmul", .. })
    ));

    let left = Array::from_vec(vec![1_i64; 6], &[2, 3]).unwrap();
    let right = Array::from_vec(vec![1_i64; 8], &[4, 2]).unwrap();
    assert!(matches!(
        matmul(&left, &right),
        Err(Error::ContractionMismatch { left: 3, right: 4 })
    ));

    let batched_left = Array::from_vec(vec![1_i64; 24], &[2, 3, 4]).unwrap();
    let batched_right = Array::from_vec(vec![1_i64; 40], &[5, 4, 2]).unwrap();
    assert!(matches!(
        matmul(&batched_left, &batched_right),
        Err(Error::BatchBroadcastMismatch { .. })
    ));

    let tensor = Array::from_vec(vec![1_i64; 6], &[1, 2, 3]).unwrap();
    assert!(matches!(
        dot(&tensor, &left),
        Err(Error::InvalidRank { op: "dot", .. })
    ));
}

#[test]
fn vdot_and_diagonal_reject_invalid_geometry() {
    let left = Array::from_vec(vec![1_i64; 6], &[2, 3]).unwrap();
    let right = Array::from_vec(vec![1_i64; 5], &[5]).unwrap();
    assert!(matches!(
        vdot(&left, &right),
        Err(Error::FlattenedSizeMismatch { left: 6, right: 5 })
    ));

    let vector = Array::from_vec(vec![1_i64; 3], &[3]).unwrap();
    assert!(matches!(
        diagonal(&vector, 0, 0, 1),
        Err(Error::InvalidRank {
            op: "diagonal and trace",
            ..
        })
    ));
}
