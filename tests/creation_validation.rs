use sdnp::{
    arange, diag, geomspace, linspace, logspace, meshgrid, tril, triu, Array,
    Error, MeshgridIndexing,
};

#[test]
fn range_constructors_reject_invalid_semantics_in_core() {
    assert!(matches!(
        arange(0, 3, 0),
        Err(Error::InvalidArgument(message))
            if message == "arange step must not be zero"
    ));
    assert!(matches!(
        linspace(f64::NAN, 1.0, 3, true),
        Err(Error::InvalidArgument(message))
            if message == "linspace bounds must be finite"
    ));
    assert!(matches!(
        logspace(0.0, f64::INFINITY, 3, true, 10.0),
        Err(Error::InvalidArgument(message))
            if message == "logspace bounds must be finite"
    ));
    assert!(matches!(
        logspace(0.0, 1.0, 3, true, 0.0),
        Err(Error::InvalidArgument(message))
            if message
                == "logspace base must be finite and greater than zero"
    ));
    assert!(matches!(
        geomspace(0.0, 1.0, 3, true),
        Err(Error::InvalidArgument(message))
            if message == "geomspace bounds must not be zero"
    ));
    assert!(matches!(
        geomspace(-1.0, 1.0, 3, true),
        Err(Error::InvalidArgument(message))
            if message == "geomspace bounds must have the same sign"
    ));
}

#[test]
fn triangular_and_diag_reject_invalid_rank_in_core() {
    let scalar = Array::from_slice(&[1_i64], &[]).unwrap();
    assert!(matches!(
        tril(&scalar, 0),
        Err(Error::InvalidRank {
            op: "tril",
            actual: 0,
            ..
        })
    ));
    assert!(matches!(
        triu(&scalar, 0),
        Err(Error::InvalidRank {
            op: "triu",
            actual: 0,
            ..
        })
    ));

    let tensor = Array::from_vec(vec![1_i64; 8], &[2, 2, 2]).unwrap();
    assert!(matches!(
        diag(&tensor, 0),
        Err(Error::InvalidRank {
            op: "diag",
            actual: 3,
            ..
        })
    ));
}

#[test]
fn meshgrid_rejects_non_vector_inputs_in_core() {
    let matrix = Array::from_vec(vec![1_i64; 4], &[2, 2]).unwrap();
    assert!(matches!(
        meshgrid(&[&matrix], MeshgridIndexing::Ij),
        Err(Error::InvalidRank {
            op: "meshgrid",
            actual: 2,
            ..
        })
    ));
}
