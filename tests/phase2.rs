//! Phase 2: ufuncs, promotion, NaN/Inf, complex helpers.

use num_complex::Complex;
use sdnp::{
    absolute, add, conj, divide, equal, imag, isnan, logical_and, multiply,
    real, trunc_divide, Array, Error,
};

#[test]
fn promote_add_and_broadcast_mul() {
    let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    let b = Array::from_slice(&[10.0_f64], &[1]).unwrap();
    let s = add(&a, &b).unwrap();
    assert_eq!(s.get(&[2]).unwrap(), 13.0);

    let row = Array::from_slice(&[1_i64, 2], &[1, 2]).unwrap();
    let col = Array::from_slice(&[10_i64, 20, 30], &[3, 1]).unwrap();
    let m = multiply(&row, &col).unwrap();
    assert_eq!(m.shape(), &[3, 2]);
    assert_eq!(m.get(&[2, 1]).unwrap(), 60);
}

#[test]
fn divide_rust_semantics() {
    let a = Array::from_slice(&[7_i64, -7], &[2]).unwrap();
    let b = Array::from_slice(&[2_i64, 2], &[2]).unwrap();
    let q = divide(&a, &b).unwrap();
    assert_eq!(q.get(&[0]).unwrap(), 3);
    assert_eq!(q.get(&[1]).unwrap(), -3);

    let z = Array::from_slice(&[0_i64], &[1]).unwrap();
    assert_eq!(
        divide(&Array::from_slice(&[1_i64], &[1]).unwrap(), &z).unwrap_err(),
        Error::DivideByZero
    );

    let fa = Array::from_slice(&[-7.0], &[1]).unwrap();
    let fb = Array::from_slice(&[2.0], &[1]).unwrap();
    assert_eq!(divide(&fa, &fb).unwrap().get(&[0]).unwrap(), -3.5);
    assert_eq!(trunc_divide(&fa, &fb).unwrap().get(&[0]).unwrap(), -3.0);
}

#[test]
fn bool_add_stays_bool() {
    let a = Array::from_slice(&[true, false], &[2]).unwrap();
    let b = Array::from_slice(&[true, true], &[2]).unwrap();
    let c = add(&a, &b).unwrap();
    assert!(c.get(&[0]).unwrap());
    assert!(c.get(&[1]).unwrap());
}

#[test]
fn compare_logical_isnan() {
    let a = Array::from_slice(&[1_i64, 0, 3], &[3]).unwrap();
    let b = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert!(equal(&a, &b).unwrap().get(&[0]).unwrap());
    assert!(!logical_and(&a, &b).unwrap().get(&[1]).unwrap());

    let f = Array::from_slice(&[f64::NAN, 1.0], &[2]).unwrap();
    assert!(isnan(&f).unwrap().get(&[0]).unwrap());
}

#[test]
fn complex_helpers() {
    let z = Array::from_slice(&[Complex::new(3.0, 4.0)], &[1]).unwrap();
    assert_eq!(absolute(&z).unwrap().get(&[0]).unwrap(), 5.0);
    assert_eq!(real(&z).unwrap().get(&[0]).unwrap(), 3.0);
    assert_eq!(imag(&z).unwrap().get(&[0]).unwrap(), 4.0);
    assert_eq!(
        conj(&z).unwrap().get(&[0]).unwrap(),
        Complex::new(3.0, -4.0)
    );
}

#[test]
fn noncontiguous_add() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let t = a.transpose();
    let ones = Array::from_vec(vec![1_i64; 6], &[3, 2]).unwrap();
    let c = add(&t, &ones).unwrap();
    assert_eq!(c.get(&[0, 1]).unwrap(), 5);
}
