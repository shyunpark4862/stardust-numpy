//! Phase 4: reductions and cumulative ops.

use sdnp::{
    all, any, argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum,
    var, Array, Error,
};

#[test]
fn sum_all_and_axes() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    assert_eq!(sum(&a, None, false).unwrap().item().unwrap(), 21);
    assert_eq!(sum(&a, Some(&[0]), false).unwrap().to_vec(), vec![5, 7, 9]);
    assert_eq!(sum(&a, Some(&[1]), false).unwrap().to_vec(), vec![6, 15]);
    assert_eq!(sum(&a, Some(&[-1]), false).unwrap().to_vec(), vec![6, 15]);
}

#[test]
fn sum_keepdims_and_multi_axis() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let k = sum(&a, Some(&[0]), true).unwrap();
    assert_eq!(k.shape(), &[1, 3]);
    assert_eq!(k.to_vec(), vec![5, 7, 9]);

    let b = Array::from_slice(
        &(0..24).map(|x| x as i64).collect::<Vec<_>>(),
        &[2, 3, 4],
    )
    .unwrap();
    // sum over axes 0 and 2 → shape (3,)
    let s = sum(&b, Some(&[0, 2]), false).unwrap();
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.to_vec(), vec![60, 92, 124]);
}

#[test]
fn prod_min_max() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    assert_eq!(prod(&a, None, false).unwrap().item().unwrap(), 24);
    assert_eq!(prod(&a, Some(&[0]), false).unwrap().to_vec(), vec![3, 8]);

    let b = Array::from_slice(&[-2_i64, 8, 3, 1], &[2, 2]).unwrap();
    assert_eq!(min(&b, None, false).unwrap().item().unwrap(), -2);
    assert_eq!(max(&b, None, false).unwrap().item().unwrap(), 8);
    assert_eq!(min(&b, Some(&[0]), false).unwrap().to_vec(), vec![-2, 1]);
}

#[test]
fn empty_sum_prod_ok_min_err() {
    let a = Array::from_slice(&[] as &[i64], &[0]).unwrap();
    assert_eq!(sum(&a, None, false).unwrap().item().unwrap(), 0);
    assert_eq!(prod(&a, None, false).unwrap().item().unwrap(), 1);
    assert!(matches!(
        min(&a, None, false),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        mean(&a, None, false),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn mean_var_std() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    assert!(
        (mean(&a, None, false).unwrap().item().unwrap() - 3.5).abs() < 1e-12
    );
    assert_eq!(
        mean(&a, Some(&[0]), false).unwrap().to_vec(),
        vec![2.5, 3.5, 4.5]
    );

    let b = Array::from_slice(&[1_i64, 2, 3, 4], &[4]).unwrap();
    assert!(
        (var(&b, None, false).unwrap().item().unwrap() - 1.25).abs() < 1e-12
    );
    assert!(
        (std(&b, None, false).unwrap().item().unwrap() - 1.25_f64.sqrt()).abs()
            < 1e-12
    );
}

#[test]
fn argmin_argmax() {
    let a = Array::from_slice(&[3_i64, 8, 1, 2, 5, 9], &[2, 3]).unwrap();
    assert_eq!(argmin(&a, None).unwrap().item().unwrap(), 2);
    assert_eq!(argmax(&a, None).unwrap().item().unwrap(), 5);
    assert_eq!(argmin(&a, Some(0)).unwrap().to_vec(), vec![1, 1, 0]);
    assert_eq!(argmax(&a, Some(1)).unwrap().to_vec(), vec![1, 2]);

    let ties = Array::from_slice(&[3_i64, 1, 1, 2], &[4]).unwrap();
    assert_eq!(argmin(&ties, None).unwrap().item().unwrap(), 1);
}

#[test]
fn cumsum_cumprod() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    assert_eq!(cumsum(&a, Some(1)).unwrap().to_vec(), vec![1, 3, 3, 7]);
    assert_eq!(cumsum(&a, Some(0)).unwrap().to_vec(), vec![1, 2, 4, 6]);
    assert_eq!(cumsum(&a, None).unwrap().to_vec(), vec![1, 3, 6, 10]);
    assert_eq!(cumprod(&a, Some(1)).unwrap().to_vec(), vec![1, 2, 3, 12]);
}

#[test]
fn any_all() {
    let a = Array::from_slice(&[0_i64, 0, 0, 3], &[2, 2]).unwrap();
    assert!(any(&a, None, false).unwrap().item().unwrap());
    assert!(!all(&a, None, false).unwrap().item().unwrap());
    assert_eq!(
        any(&a, Some(&[0]), false).unwrap().to_vec(),
        vec![false, true]
    );

    let empty = Array::from_slice(&[] as &[i64], &[0]).unwrap();
    assert!(!any(&empty, None, false).unwrap().item().unwrap());
    assert!(all(&empty, None, false).unwrap().item().unwrap());
}

#[test]
fn duplicate_axis_err() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    assert!(sum(&a, Some(&[0, 0]), false).is_err());
}

#[test]
fn empty_outer_vs_empty_inner() {
    // shape (0, 3): reducing axis 1 → empty outer → empty result
    let a = Array::from_slice(&[] as &[i64], &[0, 3]).unwrap();
    let s = sum(&a, Some(&[1]), false).unwrap();
    assert_eq!(s.shape(), &[0]);
    assert_eq!(s.to_vec(), Vec::<i64>::new());

    let m = min(&a, Some(&[1]), false).unwrap();
    assert_eq!(m.shape(), &[0]);

    let mean_out = mean(&a, Some(&[1]), false).unwrap();
    assert_eq!(mean_out.shape(), &[0]);

    // reducing axis 0 → empty inner → identity / error
    assert_eq!(sum(&a, Some(&[0]), false).unwrap().to_vec(), vec![0, 0, 0]);
    assert!(matches!(
        min(&a, Some(&[0]), false),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        mean(&a, Some(&[0]), false),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn bool_cumsum_cumprod_promote() {
    let a = Array::from_slice(&[true, false, true], &[3]).unwrap();
    assert_eq!(cumsum(&a, None).unwrap().to_vec(), vec![1_i64, 1, 2]);
    assert_eq!(cumprod(&a, None).unwrap().to_vec(), vec![1_i64, 0, 0]);
}

#[test]
fn noncontiguous_and_suffix_paths() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    // last-axis / suffix contiguous path
    assert_eq!(sum(&a, Some(&[1]), false).unwrap().to_vec(), vec![6, 15]);
    // single non-suffix axis on a transpose → fixed-stride path
    let t = a.transpose();
    assert_eq!(sum(&t, Some(&[0]), false).unwrap().to_vec(), vec![6, 15]);
    assert_eq!(sum(&t, Some(&[1]), false).unwrap().to_vec(), vec![5, 7, 9]);
}
