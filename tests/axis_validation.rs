use sdnp::{
    argmax, argmin, concatenate, diagonal, max, mean, min, sort, stack, std,
    sum, var, Array, Error, NanPolicy,
};

#[test]
fn single_axis_operations_return_axis_bounds_errors() {
    let array = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();

    assert!(matches!(
        sort(&array, Some(2)),
        Err(Error::AxisOutOfBounds { axis: 2, ndim: 2 })
    ));
    assert!(matches!(
        concatenate(&[&array], -3),
        Err(Error::AxisOutOfBounds { axis: -3, ndim: 2 })
    ));
}

#[test]
fn reduction_axes_must_be_unique() {
    let array = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();

    assert!(matches!(
        sum(&array, Some(&[0, -2]), false, NanPolicy::Propagate),
        Err(Error::DuplicateAxes)
    ));
}

#[test]
fn permutations_must_be_complete_and_unique() {
    let array = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();

    assert!(matches!(
        array.permute_axes(&[0]),
        Err(Error::NotPermutation)
    ));
    assert!(matches!(
        array.permute_axes(&[0, 0]),
        Err(Error::NotPermutation)
    ));
    assert!(matches!(
        array.permute_axes(&[0, 2]),
        Err(Error::AxisOutOfBounds { axis: 2, ndim: 2 })
    ));
}

#[test]
fn insert_and_diagonal_axes_are_semantically_validated() {
    let array = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();

    assert!(matches!(
        stack(&[&array], 3),
        Err(Error::AxisOutOfBounds { axis: 3, ndim: 3 })
    ));
    assert!(matches!(
        diagonal(&array, 0, 0, -2),
        Err(Error::AxesMustDiffer)
    ));
    assert!(matches!(
        diagonal(&array, 0, 0, 2),
        Err(Error::AxisOutOfBounds { axis: 2, ndim: 2 })
    ));
}

#[test]
fn squeeze_rejects_duplicates_and_non_unit_axes() {
    let array = Array::from_vec((0_i64..3).collect(), &[1, 3]).unwrap();

    assert!(matches!(
        array.squeeze(Some(&[0, -2])),
        Err(Error::DuplicateAxes)
    ));
    assert!(matches!(
        array.squeeze(Some(&[1])),
        Err(Error::CannotSqueezeAxis {
            axis: 1,
            axis_len: 3,
        })
    ));
    assert!(matches!(
        array.squeeze(Some(&[])),
        Err(Error::InvalidArgument(message))
            if message == "squeeze axes must be a non-empty sequence"
    ));
}

#[test]
fn reshape_validation_is_owned_by_core() {
    let array = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();

    assert!(matches!(
        array.reshape(&[]),
        Err(Error::InvalidArgument(message))
            if message == "reshape target shape must be non-empty"
    ));
    assert!(matches!(
        array.reshape(&[-1, -1]),
        Err(Error::InvalidArgument(message))
            if message == "only one reshape dimension may be -1"
    ));
    assert!(matches!(
        array.reshape(&[4]),
        Err(Error::InvalidArgument(message))
            if message.contains("cannot reshape array of size 6")
    ));
}

#[test]
fn non_identity_reductions_reject_nonempty_empty_slices() {
    let array = Array::<f64>::from_vec(Vec::new(), &[2, 0]).unwrap();
    let axis = [1];

    for result in [
        min(&array, Some(&axis), false, NanPolicy::Propagate),
        max(&array, Some(&axis), false, NanPolicy::Propagate),
    ] {
        assert!(matches!(result, Err(Error::EmptyReduction { .. })));
    }
    assert!(matches!(
        mean(&array, Some(&axis), false, NanPolicy::Propagate),
        Err(Error::EmptyReduction { op: "mean" })
    ));
    assert!(matches!(
        var(&array, Some(&axis), false, NanPolicy::Propagate),
        Err(Error::EmptyReduction { op: "var" })
    ));
    assert!(matches!(
        std(&array, Some(&axis), false, NanPolicy::Propagate),
        Err(Error::EmptyReduction { op: "std" })
    ));
    assert!(matches!(
        argmin(&array, Some(1), NanPolicy::Propagate),
        Err(Error::EmptyReduction { op: "argmin" })
    ));
    assert!(matches!(
        argmax(&array, Some(1), NanPolicy::Propagate),
        Err(Error::EmptyReduction { op: "argmax" })
    ));
}

#[test]
fn nan_ignoring_arg_reductions_reject_all_nan_slices_in_core() {
    let array =
        Array::from_vec(vec![1.0, f64::NAN, 2.0, f64::NAN], &[2, 2]).unwrap();

    assert!(matches!(
        argmin(&array, Some(0), NanPolicy::Ignore),
        Err(Error::AllNanSlice { op: "argmin" })
    ));
    assert!(matches!(
        argmax(&array, Some(0), NanPolicy::Ignore),
        Err(Error::AllNanSlice { op: "argmax" })
    ));
}
