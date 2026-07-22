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

    let c = Array::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .unwrap();
    assert_eq!(
        sum(&c, Some(&[1]), false).unwrap().to_vec(),
        vec![6.0, 15.0]
    );
    assert_eq!(
        mean(&c, Some(&[1]), false).unwrap().to_vec(),
        vec![2.0, 5.0]
    );
    for actual in var(&c, Some(&[1]), false).unwrap().to_vec() {
        assert!((actual - 2.0 / 3.0).abs() < 1e-12);
    }

    let transposed = c.transpose();
    for actual in var(&transposed, Some(&[1]), false).unwrap().to_vec() {
        assert!((actual - 2.25).abs() < 1e-12);
    }

    let cube = Array::from_slice(
        &(0..24).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 4],
    )
    .unwrap();
    let general = var(&cube, Some(&[0, 2]), false).unwrap().to_vec();
    for actual in general {
        assert!((actual - 37.25).abs() < 1e-12);
    }
}

#[test]
fn general_strided_reduction_on_permuted_array() {
    // Permute so the reduced axes are neither a trailing block nor a single
    // axis, and the underlying strides are genuinely non-contiguous.
    let cube = Array::from_slice(
        &(0..24).map(|x| x as i64).collect::<Vec<_>>(),
        &[2, 3, 4],
    )
    .unwrap();
    let permuted = cube.permute_axes(&[1, 0, 2]).unwrap(); // shape [3,2,4]
    assert!(!permuted.is_c_contiguous());

    // Reduce axes 0 and 2 (outer axis 1 survives) — matches direct
    // computation from the original cube for cross-checking.
    let s = sum(&permuted, Some(&[0, 2]), false).unwrap();
    assert_eq!(s.shape(), &[2]);
    let expected: Vec<i64> = (0..2)
        .map(|i| {
            (0..3)
                .flat_map(|j| (0..4).map(move |k| (i, j, k)))
                .map(|(i, j, k)| cube.get(&[i, j, k]).unwrap())
                .sum()
        })
        .collect();
    assert_eq!(s.to_vec(), expected);

    let mn = min(&permuted, Some(&[0, 2]), false).unwrap();
    let mx = max(&permuted, Some(&[0, 2]), false).unwrap();
    assert_eq!(mn.to_vec(), vec![0, 12]);
    assert_eq!(mx.to_vec(), vec![11, 23]);
}

#[test]
fn general_strided_reduction_reversed_axis() {
    // Negative-stride (reversed) axis mixed into a multi-axis reduction.
    let a = Array::from_slice(
        &(0..12).map(|x| x as i64).collect::<Vec<_>>(),
        &[3, 4],
    )
    .unwrap();
    let reversed_cols = sdnp::gather(
        &a,
        &[
            sdnp::IndexSpec::full(),
            sdnp::IndexSpec::slice(None, None, Some(-1)),
        ],
    )
    .unwrap();
    assert!(!reversed_cols.is_c_contiguous());
    // Sum ignoring column order must match the un-reversed sum.
    assert_eq!(
        sum(&reversed_cols, Some(&[1]), false).unwrap().to_vec(),
        sum(&a, Some(&[1]), false).unwrap().to_vec()
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
fn flat_argmin_argmax_on_strided_layouts() {
    // Contiguous [2,3] → transpose is shape [3,2], strides [1,3].
    // Flat C-order: 3,2, 8,5, 1,9 → argmin=4, argmax=5.
    let a = Array::from_slice(&[3_i64, 8, 1, 2, 5, 9], &[2, 3]).unwrap();
    let t = a.transpose();
    assert!(!t.is_c_contiguous());
    assert_eq!(argmin(&t, None).unwrap().item().unwrap(), 4);
    assert_eq!(argmax(&t, None).unwrap().item().unwrap(), 5);

    // Column step [::, ::2]: shape [2,2], strides [3,2] — partially coalescible.
    // Flat C-order: 3,1, 2,9 → argmin=1, argmax=3.
    let cols = sdnp::gather(
        &a,
        &[
            sdnp::IndexSpec::full(),
            sdnp::IndexSpec::slice(None, None, Some(2)),
        ],
    )
    .unwrap();
    assert!(!cols.is_c_contiguous());
    assert_eq!(argmin(&cols, None).unwrap().item().unwrap(), 1);
    assert_eq!(argmax(&cols, None).unwrap().item().unwrap(), 3);

    // Reverse rows: negative stride, flat C-order 2,5,9, 3,8,1 → argmin=5, argmax=2.
    let rev = sdnp::gather(
        &a,
        &[
            sdnp::IndexSpec::slice(None, None, Some(-1)),
            sdnp::IndexSpec::full(),
        ],
    )
    .unwrap();
    assert!(!rev.is_c_contiguous());
    assert_eq!(argmin(&rev, None).unwrap().item().unwrap(), 5);
    assert_eq!(argmax(&rev, None).unwrap().item().unwrap(), 2);
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
fn flat_cumsum_cumprod_on_strided_layouts() {
    // Contiguous [2,3] → transpose [3,2]: flat C-order 1,4, 2,5, 3,6.
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let t = a.transpose();
    assert!(!t.is_c_contiguous());
    assert_eq!(
        cumsum(&t, None).unwrap().to_vec(),
        vec![1, 5, 7, 12, 15, 21]
    );
    assert_eq!(
        cumprod(&t, None).unwrap().to_vec(),
        vec![1, 4, 8, 40, 120, 720]
    );

    // Column step: flat C-order 1,3, 4,6.
    let cols = sdnp::gather(
        &a,
        &[
            sdnp::IndexSpec::full(),
            sdnp::IndexSpec::slice(None, None, Some(2)),
        ],
    )
    .unwrap();
    assert_eq!(cumsum(&cols, None).unwrap().to_vec(), vec![1, 4, 8, 14]);

    // Negative row step: flat C-order 4,5,6, 1,2,3.
    let rev = sdnp::gather(
        &a,
        &[
            sdnp::IndexSpec::slice(None, None, Some(-1)),
            sdnp::IndexSpec::full(),
        ],
    )
    .unwrap();
    assert_eq!(
        cumsum(&rev, None).unwrap().to_vec(),
        vec![4, 9, 15, 16, 18, 21]
    );
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
