//! Phase 1 smoke: creation + broadcasting.

use sdnp::{
    arange, broadcast_arrays, broadcast_shape, eye, ones, zeros, Array,
};

#[test]
fn create_and_broadcast() {
    let z = zeros::<i64>(&[2, 1]).unwrap();
    let o = ones::<i64>(&[3]).unwrap();
    let out = broadcast_arrays(&[&z, &o]).unwrap();
    assert_eq!(out[0].shape(), &[2, 3]);
    assert_eq!(out[1].shape(), &[2, 3]);
    assert_eq!(out[1].get(&[1, 2]).unwrap(), 1);

    let a = arange(0_i64, 4, 1).unwrap();
    assert_eq!(a.size(), 4);

    let e = eye::<f64>(2).unwrap();
    assert_eq!(e.get(&[1, 1]).unwrap(), 1.0);

    assert_eq!(broadcast_shape(&[2, 1], &[3]).unwrap(), vec![2, 3]);

    let row = Array::from_slice(&[10_i64, 20, 30], &[1, 3]).unwrap();
    let wide = row.broadcast_to(&[4, 3]).unwrap();
    assert!(row.shares_buffer_with(&wide));
    assert_eq!(wide.get(&[3, 1]).unwrap(), 20);
}
