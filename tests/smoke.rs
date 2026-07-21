//! Smoke tests for Phase 0 core.

use sdnp::{c_order_strides, gather, Array, IndexSpec};

#[test]
fn scaffold_array_get_set() {
    let mut a = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2]).unwrap();
    assert_eq!(a.get(&[0, 1]).unwrap(), 20);
    a.set(&[1, 0], 99).unwrap();
    assert_eq!(a.get(&[1, 0]).unwrap(), 99);
    assert_eq!(c_order_strides(a.shape()), a.strides());
}

#[test]
fn transpose_and_basic_slice_share_buffer() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let t = a.t();
    let s = gather(
        &a,
        &[
            IndexSpec::slice(Some(1), Some(2), None),
            IndexSpec::slice(None, None, None),
        ],
    )
    .unwrap();
    assert!(a.shares_buffer_with(&t));
    assert!(a.shares_buffer_with(&s));
    assert_eq!(s.shape(), &[1, 3]);
    assert_eq!(s.get(&[0, 2]).unwrap(), 6);
}
