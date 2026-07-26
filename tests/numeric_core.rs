use sdnp::{
    absolute, add, argmax, argmin, cumsum, divide, max, mean, min, multiply,
    negative, power, prod, remainder, sort, std, sum, unique, var, Array,
    Error, NanPolicy,
};

#[test]
fn integer_kernels_have_explicit_wrapping_semantics() {
    let maximum = Array::from_slice(&[i64::MAX], &[1]).unwrap();
    let minimum = Array::from_slice(&[i64::MIN], &[1]).unwrap();
    let one = Array::from_slice(&[1_i64], &[1]).unwrap();
    let two = Array::from_slice(&[2_i64], &[1]).unwrap();

    assert_eq!(add(&maximum, &one).unwrap().to_vec(), vec![i64::MIN]);
    assert_eq!(multiply(&maximum, &two).unwrap().to_vec(), vec![-2]);
    assert_eq!(negative(&minimum).unwrap().to_vec(), vec![i64::MIN]);
    assert_eq!(absolute(&minimum).unwrap().to_vec(), vec![i64::MIN]);
    assert_eq!(power(&maximum, &two).unwrap().to_vec(), vec![1]);
}

#[test]
fn integer_domain_errors_do_not_panic() {
    let one = Array::from_slice(&[1_i64], &[1]).unwrap();
    let zero = Array::from_slice(&[0_i64], &[1]).unwrap();
    let minus_one = Array::from_slice(&[-1_i64], &[1]).unwrap();
    let minimum = Array::from_slice(&[i64::MIN], &[1]).unwrap();

    assert!(matches!(divide(&one, &zero), Err(Error::DivideByZero)));
    assert!(matches!(remainder(&one, &zero), Err(Error::DivideByZero)));
    assert!(matches!(
        divide(&minimum, &minus_one),
        Err(Error::InvalidArgument(message)) if message.contains("overflow")
    ));
    assert!(matches!(
        power(&one, &minus_one),
        Err(Error::InvalidArgument(message)) if message.contains("exponent")
    ));
}

#[test]
fn reductions_cover_prefix_suffix_and_general_strided_layouts() {
    let base = Array::from_vec((1_i64..=24).collect(), &[2, 3, 4]).unwrap();
    let transposed = base.permute_axes(&[2, 0, 1]).unwrap();

    let suffix = sum(&base, Some(&[-1]), false, NanPolicy::Propagate).unwrap();
    assert_eq!(suffix.shape(), &[2, 3]);
    assert_eq!(suffix.to_vec(), vec![10, 26, 42, 58, 74, 90]);

    let prefix = sum(&base, Some(&[0]), false, NanPolicy::Propagate).unwrap();
    assert_eq!(prefix.shape(), &[3, 4]);
    assert_eq!(
        prefix.to_vec(),
        vec![14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    );

    let general =
        sum(&transposed, Some(&[0, 2]), false, NanPolicy::Propagate).unwrap();
    assert_eq!(general.shape(), &[2]);
    assert_eq!(general.to_vec(), vec![78, 222]);
}

#[test]
fn nan_policy_is_consistent_across_reduction_families() {
    let values = Array::from_slice(&[1.0_f64, f64::NAN, 3.0], &[3]).unwrap();

    assert!(sum(&values, None, false, NanPolicy::Propagate)
        .unwrap()
        .item()
        .unwrap()
        .is_nan());
    assert_eq!(
        sum(&values, None, false, NanPolicy::Ignore)
            .unwrap()
            .item()
            .unwrap(),
        4.0
    );
    assert_eq!(
        mean(&values, None, false, NanPolicy::Ignore)
            .unwrap()
            .item()
            .unwrap(),
        2.0
    );
    assert_eq!(
        min(&values, None, false, NanPolicy::Ignore)
            .unwrap()
            .item()
            .unwrap(),
        1.0
    );
    assert_eq!(
        max(&values, None, false, NanPolicy::Ignore)
            .unwrap()
            .item()
            .unwrap(),
        3.0
    );
}

#[test]
fn empty_reduction_identities_and_shapes_are_preserved() {
    let empty = Array::from_vec(Vec::<i64>::new(), &[2, 0, 3]).unwrap();
    let sums = sum(&empty, Some(&[1]), false, NanPolicy::Propagate).unwrap();
    let products =
        prod(&empty, Some(&[1]), false, NanPolicy::Propagate).unwrap();
    assert_eq!(sums.shape(), &[2, 3]);
    assert_eq!(sums.to_vec(), vec![0; 6]);
    assert_eq!(products.to_vec(), vec![1; 6]);
}

#[test]
fn cumulative_and_arg_reductions_handle_noncontiguous_arrays() {
    let base = Array::from_vec(vec![3_i64, 1, 4, 2, 8, 5], &[2, 3]).unwrap();
    let transposed = base.transpose();

    let cumulative =
        cumsum(&transposed, Some(-1), NanPolicy::Propagate).unwrap();
    assert_eq!(cumulative.shape(), &[3, 2]);
    assert_eq!(cumulative.to_vec(), vec![3, 5, 1, 9, 4, 9]);

    let indices = argmax(&transposed, Some(-1), NanPolicy::Propagate).unwrap();
    assert_eq!(indices.to_vec(), vec![0, 1, 1]);
}

#[test]
fn boolean_arg_reductions_handle_terminal_values_and_strides() {
    // Early-exit bool arg kernels must agree on flat, axis, and strided paths.
    let values = Array::from_vec(
        vec![true, true, false, false, false, true, false, true],
        &[2, 4],
    )
    .unwrap();

    assert_eq!(
        argmin(&values, None, NanPolicy::Propagate)
            .unwrap()
            .item()
            .unwrap(),
        2
    );
    assert_eq!(
        argmax(&values, None, NanPolicy::Propagate)
            .unwrap()
            .item()
            .unwrap(),
        0
    );
    assert_eq!(
        argmin(&values, Some(-1), NanPolicy::Propagate)
            .unwrap()
            .to_vec(),
        vec![2, 0]
    );
    assert_eq!(
        argmax(&values, Some(-1), NanPolicy::Propagate)
            .unwrap()
            .to_vec(),
        vec![0, 1]
    );

    let transposed = values.transpose();
    assert_eq!(
        argmin(&transposed, Some(-1), NanPolicy::Propagate)
            .unwrap()
            .to_vec(),
        vec![1, 0, 0, 0]
    );
    assert_eq!(
        argmax(&transposed, Some(-1), NanPolicy::Propagate)
            .unwrap()
            .to_vec(),
        vec![0, 0, 0, 1]
    );
}

#[test]
fn floating_statistics_return_expected_population_values() {
    let values = Array::from_slice(&[1_i64, 2, 3, 4], &[4]).unwrap();
    assert_eq!(
        mean(&values, None, false, NanPolicy::Propagate)
            .unwrap()
            .item()
            .unwrap(),
        2.5
    );
    assert_eq!(
        var(&values, None, false, NanPolicy::Propagate)
            .unwrap()
            .item()
            .unwrap(),
        1.25
    );
    assert_eq!(
        std(&values, None, false, NanPolicy::Propagate)
            .unwrap()
            .item()
            .unwrap(),
        1.25_f64.sqrt()
    );
}

#[test]
fn sorting_and_unique_materialize_logical_c_order() {
    let base = Array::from_vec(vec![3_i64, 1, 2, 3, 2, 1], &[2, 3]).unwrap();
    let transposed = base.transpose();
    let sorted = sort(&transposed, Some(-1)).unwrap();
    assert_eq!(sorted.shape(), &[3, 2]);
    assert_eq!(sorted.to_vec(), vec![3, 3, 1, 2, 1, 2]);

    let values = unique(&transposed).unwrap();
    assert_eq!(values.to_vec(), vec![1, 2, 3]);
}
