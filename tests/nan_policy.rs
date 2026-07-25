use sdnp::{
    argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum, var,
    Array, Complex64, NanPolicy,
};

const P: NanPolicy = NanPolicy::Propagate;
const I: NanPolicy = NanPolicy::Ignore;

#[test]
fn propagate_preserves_nan_behavior() {
    let a = Array::from_slice(&[1.0, f64::NAN, 3.0], &[3]).unwrap();
    assert!(sum(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert!(prod(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert!(mean(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert!(var(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert!(std(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert!(min(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert!(max(&a, None, false, P).unwrap().item().unwrap().is_nan());
    assert_eq!(argmin(&a, None, P).unwrap().item().unwrap(), 1);
    assert_eq!(argmax(&a, None, P).unwrap().item().unwrap(), 1);
    let cumulative = cumsum(&a, None, P).unwrap().to_vec();
    assert_eq!(cumulative[0], 1.0);
    assert!(cumulative[1].is_nan() && cumulative[2].is_nan());
}

#[test]
fn ignore_suffix_partial_and_all_nan() {
    let a = Array::from_slice(
        &[1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, f64::NAN],
        &[2, 3],
    )
    .unwrap();
    let axes = [1];
    let sums = sum(&a, Some(&axes), false, I).unwrap().to_vec();
    assert_eq!(sums[0], 4.0);
    assert!(sums[1].is_nan());
    let products = prod(&a, Some(&axes), false, I).unwrap().to_vec();
    assert_eq!(products[0], 3.0);
    assert!(products[1].is_nan());
    let means = mean(&a, Some(&axes), false, I).unwrap().to_vec();
    assert_eq!(means[0], 2.0);
    assert!(means[1].is_nan());
    let variances = var(&a, Some(&axes), false, I).unwrap().to_vec();
    assert_eq!(variances[0], 1.0);
    assert!(variances[1].is_nan());
    let deviations = std(&a, Some(&axes), false, I).unwrap().to_vec();
    assert_eq!(deviations[0], 1.0);
    assert!(deviations[1].is_nan());
    let minima = min(&a, Some(&axes), false, I).unwrap().to_vec();
    assert_eq!(minima[0], 1.0);
    assert!(minima[1].is_nan());
    assert_eq!(argmin(&a, Some(1), I).unwrap().shape()[0], 2);
    assert_eq!(argmax(&a, Some(1), I).unwrap().shape()[0], 2);
}

#[test]
fn ignore_prefix_and_general_strided() {
    let a = Array::from_slice(
        &[
            1.0,
            f64::NAN,
            3.0,
            f64::NAN,
            5.0,
            7.0,
            9.0,
            11.0,
            f64::NAN,
            13.0,
            15.0,
            17.0,
        ],
        &[3, 2, 2],
    )
    .unwrap();
    assert_eq!(
        sum(&a, Some(&[0]), false, I).unwrap().to_vec(),
        vec![6.0, 20.0, 27.0, 28.0]
    );

    let permuted = a.permute_axes(&[1, 0, 2]).unwrap();
    assert!(!permuted.is_c_contiguous());
    assert_eq!(
        sum(&permuted, Some(&[0, 2]), false, I).unwrap().to_vec(),
        vec![4.0, 32.0, 45.0]
    );
    assert_eq!(
        mean(&permuted, Some(&[0, 2]), false, I).unwrap().to_vec(),
        vec![2.0, 8.0, 15.0]
    );
}

#[test]
fn ignore_arg_indices_skip_nans_and_keep_first_tie() {
    let a =
        Array::from_slice(&[f64::NAN, 3.0, 1.0, 1.0, f64::NAN], &[5]).unwrap();
    assert_eq!(argmin(&a, None, I).unwrap().item().unwrap(), 2);
    assert_eq!(argmax(&a, None, I).unwrap().item().unwrap(), 1);
    let all_nan = Array::from_slice(&[f64::NAN, f64::NAN], &[2]).unwrap();
    assert_eq!(argmin(&all_nan, None, I).unwrap().item().unwrap(), 0);
}

#[test]
fn ignore_cumulative_contiguous_and_strided() {
    let a = Array::from_slice(
        &[f64::NAN, 2.0, f64::NAN, 3.0, f64::NAN, f64::NAN],
        &[2, 3],
    )
    .unwrap();
    let sum_axis = cumsum(&a, Some(1), I).unwrap().to_vec();
    assert!(sum_axis[0].is_nan());
    assert_eq!(&sum_axis[1..3], &[2.0, 2.0]);
    assert_eq!(sum_axis[3], 3.0);
    assert_eq!(sum_axis[4], 3.0);
    assert_eq!(sum_axis[5], 3.0);

    let product_axis = cumprod(&a, Some(1), I).unwrap().to_vec();
    assert!(product_axis[0].is_nan());
    assert_eq!(&product_axis[1..3], &[2.0, 2.0]);
    assert_eq!(&product_axis[3..], &[3.0, 3.0, 3.0]);

    let transposed = a.transpose();
    let flat = cumsum(&transposed, None, I).unwrap().to_vec();
    assert!(flat[0].is_nan());
    assert_eq!(&flat[1..], &[3.0, 5.0, 5.0, 5.0, 5.0]);
    let axis = cumsum(&transposed, Some(0), I).unwrap().to_vec();
    assert!(axis[0].is_nan());
    assert_eq!(axis[1], 3.0);
    assert_eq!(axis[2], 2.0);
    assert_eq!(axis[3], 3.0);
    assert_eq!(axis[4], 2.0);
    assert_eq!(axis[5], 3.0);

    let all_nan = Array::from_slice(&[f64::NAN, f64::NAN], &[2]).unwrap();
    assert!(cumsum(&all_nan, None, I)
        .unwrap()
        .to_vec()
        .iter()
        .all(|value| value.is_nan()));
}

#[test]
fn complex_ignore_skips_whole_component_nan_element() {
    let a = Array::from_slice(
        &[
            Complex64::new(1.0, 2.0),
            Complex64::new(f64::NAN, 4.0),
            Complex64::new(3.0, f64::NAN),
            Complex64::new(5.0, 6.0),
        ],
        &[4],
    )
    .unwrap();
    assert_eq!(
        sum(&a, None, false, I).unwrap().item().unwrap(),
        Complex64::new(6.0, 8.0)
    );
    assert_eq!(
        mean(&a, None, false, I).unwrap().item().unwrap(),
        Complex64::new(3.0, 4.0)
    );
    assert_eq!(
        prod(&a, None, false, I).unwrap().item().unwrap(),
        Complex64::new(-7.0, 16.0)
    );
    let cumulative = cumsum(&a, None, I).unwrap().to_vec();
    assert_eq!(cumulative[0], Complex64::new(1.0, 2.0));
    assert_eq!(cumulative[1], cumulative[0]);
    assert_eq!(cumulative[2], cumulative[0]);
    assert_eq!(cumulative[3], Complex64::new(6.0, 8.0));
    let products = cumprod(&a, None, I).unwrap().to_vec();
    assert_eq!(products[1], products[0]);
    assert_eq!(products[2], products[0]);
    assert_eq!(products[3], Complex64::new(-7.0, 16.0));
}

#[test]
fn ignore_empty_zero_dim_and_non_nan_types() {
    let empty = Array::from_slice(&[] as &[f64], &[0]).unwrap();
    assert_eq!(sum(&empty, None, false, I).unwrap().item().unwrap(), 0.0);
    assert_eq!(prod(&empty, None, false, I).unwrap().item().unwrap(), 1.0);
    let _ = mean(&empty, None, false, I).unwrap();
    let _ = min(&empty, None, false, I).unwrap();

    let scalar = Array::from_slice(&[f64::NAN], &[]).unwrap();
    assert!(sum(&scalar, None, false, I)
        .unwrap()
        .item()
        .unwrap()
        .is_nan());
    assert!(mean(&scalar, None, false, I)
        .unwrap()
        .item()
        .unwrap()
        .is_nan());
    assert!(cumsum(&scalar, None, I).unwrap().to_vec()[0].is_nan());

    let integers = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert_eq!(
        sum(&integers, None, false, I).unwrap().to_vec(),
        sum(&integers, None, false, P).unwrap().to_vec()
    );
    assert_eq!(
        cumsum(&integers, None, I).unwrap().to_vec(),
        cumsum(&integers, None, P).unwrap().to_vec()
    );
    assert_eq!(
        min(&integers, None, false, I).unwrap().to_vec(),
        min(&integers, None, false, P).unwrap().to_vec()
    );
    assert_eq!(
        argmax(&integers, None, I).unwrap().to_vec(),
        argmax(&integers, None, P).unwrap().to_vec()
    );

    let booleans = Array::from_slice(&[true, false, true], &[3]).unwrap();
    assert_eq!(
        prod(&booleans, None, false, I).unwrap().to_vec(),
        prod(&booleans, None, false, P).unwrap().to_vec()
    );
    assert_eq!(
        mean(&booleans, None, false, I).unwrap().to_vec(),
        mean(&booleans, None, false, P).unwrap().to_vec()
    );
    assert_eq!(
        var(&booleans, None, false, I).unwrap().to_vec(),
        var(&booleans, None, false, P).unwrap().to_vec()
    );
}
