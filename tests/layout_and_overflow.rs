use sdnp::{
    broadcast_shape, c_order_strides, size_of_shape, zeros, Array, Error,
};

#[test]
fn structured_construction_errors_remain_distinguishable() {
    assert_eq!(
        Array::from_vec(vec![1_i64, 2], &[2, 2]).unwrap_err(),
        Error::BufferSizeMismatch {
            buffer_len: 2,
            size: 4,
        }
    );

    assert!(matches!(
        Array::<i64>::from_vec(
            Vec::new(),
            &[isize::MAX as usize, 0, 2],
        ),
        Err(Error::InvalidArgument(message))
            if message.contains("shape geometry")
    ));
}

#[test]
fn public_shape_helpers_report_overflow_instead_of_wrapping() {
    assert!(size_of_shape(&[usize::MAX, 2]).is_err());
    assert!(c_order_strides(&[0, isize::MAX as usize, 2]).is_err());
    assert_eq!(size_of_shape(&[]).unwrap(), 1);
    assert_eq!(c_order_strides(&[2, 3]).unwrap(), vec![3, 1]);
}

#[test]
fn core_keeps_zero_dimensional_internal_representation() {
    let scalar = Array::from_slice(&[42_i64], &[]).unwrap();
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.strides(), &[]);
    assert_eq!(scalar.size(), 1);
    assert_eq!(scalar.item().unwrap(), 42);
    assert_eq!(scalar.get(&[]).unwrap(), 42);
}

#[test]
fn empty_layouts_preserve_shape_and_safe_strides() {
    let array = zeros::<i64>(&[2, 0, 3]).unwrap();
    assert_eq!(array.shape(), &[2, 0, 3]);
    assert_eq!(array.strides(), &[0, 3, 1]);
    assert_eq!(array.size(), 0);
    assert!(array.to_vec().is_empty());
}

#[test]
fn views_detach_on_write_but_broadcast_views_are_read_only() {
    let source = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();
    let mut transposed = source.transpose();
    transposed.set(&[0, 0], 99).unwrap();
    assert_eq!(source.get(&[0, 0]).unwrap(), 0);
    assert_eq!(transposed.get(&[0, 0]).unwrap(), 99);

    let mut broadcast = Array::from_slice(&[1_i64, 2, 3], &[1, 3])
        .unwrap()
        .broadcast_to(&[4, 3])
        .unwrap();
    assert_eq!(broadcast.set(&[1, 1], 7), Err(Error::ReadOnly));
}

#[test]
fn reshape_checks_products_and_supports_inference() {
    let array = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();
    let inferred = array.reshape(&[3, -1]).unwrap();
    assert_eq!(inferred.shape(), &[3, 2]);
    assert_eq!(inferred.to_vec(), vec![0, 1, 2, 3, 4, 5]);

    assert!(array.reshape(&[-1, -1]).is_err());
    assert!(array.reshape(&[4, 2]).is_err());
    assert!(array.reshape(&[isize::MAX, 2]).is_err());
}

#[test]
fn broadcast_shape_reports_incompatible_inputs() {
    assert_eq!(
        broadcast_shape(&[2, 1, 3], &[1, 4, 3]).unwrap(),
        vec![2, 4, 3]
    );
    assert!(matches!(
        broadcast_shape(&[2, 3], &[4, 3]),
        Err(Error::Broadcast { .. })
    ));
}
