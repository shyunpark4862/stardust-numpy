//! Phase 5: joining, selection, sorting, unique, and spaces.

use sdnp::{
    argsort, clip, concatenate, geomspace, hstack, linspace, logspace,
    meshgrid, nonzero, sort, stack, unique, unique_with, vstack, where_, Array,
    Complex64, MeshgridIndexing, UniqueOptions,
};

#[test]
fn join_contiguous_and_noncontiguous_inputs() {
    let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    let b = Array::from_slice(&[5_i64, 6], &[1, 2]).unwrap();
    let joined = concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(joined.shape(), &[3, 2]);
    assert_eq!(joined.to_vec(), vec![1, 2, 3, 4, 5, 6]);

    let transposed = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[3, 2])
        .unwrap()
        .transpose();
    let row = Array::from_slice(&[7_i64, 8, 9], &[1, 3]).unwrap();
    let joined = concatenate(&[&transposed, &row], 0).unwrap();
    assert_eq!(joined.to_vec(), vec![1, 3, 5, 2, 4, 6, 7, 8, 9]);
}

#[test]
fn stack_variants() {
    let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
    let b = Array::from_slice(&[3_i64, 4], &[2]).unwrap();

    assert_eq!(stack(&[&a, &b], 1).unwrap().to_vec(), vec![1, 3, 2, 4]);
    assert_eq!(vstack(&[&a, &b]).unwrap().shape(), &[2, 2]);
    assert_eq!(hstack(&[&a, &b]).unwrap().shape(), &[4]);
}

#[test]
fn selection_broadcast_nonzero_and_clip() {
    let condition =
        Array::from_slice(&[true, false, false, true], &[2, 2]).unwrap();
    let x = Array::from_slice(&[1_i64, 2], &[1, 2]).unwrap();
    let y = Array::from_slice(&[10.0_f64], &[]).unwrap();
    let selected = where_(&condition, &x, &y).unwrap();
    assert_eq!(selected.to_vec(), vec![1.0, 10.0, 10.0, 2.0]);

    let nz = nonzero(&selected).unwrap();
    assert_eq!(nz[0].to_vec(), vec![0, 0, 1, 1]);
    assert_eq!(nz[1].to_vec(), vec![0, 1, 0, 1]);

    let source = Array::from_slice(&[-2_i64, 0, 3, 8], &[4]).unwrap();
    let clipped = clip(&source, Some(0), Some(5)).unwrap();
    assert_eq!(clipped.to_vec(), vec![0, 0, 3, 5]);
    assert!(!source.shares_buffer_with(&clipped));
}

#[test]
fn sorting_axes_and_flatten() {
    let a = Array::from_slice(&[3_i64, 1, 2, 6, 4, 5], &[2, 3]).unwrap();
    assert_eq!(sort(&a, Some(-1)).unwrap().to_vec(), vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(argsort(&a, None).unwrap().to_vec(), vec![1, 2, 0, 4, 5, 3]);
}

#[test]
fn squeeze_and_astype_cover_views_and_explicit_conversions() {
    let source = Array::from_slice(
        &[1.9_f64, -2.1, f64::NAN, f64::INFINITY],
        &[1, 2, 2],
    )
    .unwrap();
    let squeezed = source.squeeze(None).unwrap();
    assert_eq!(squeezed.shape(), &[2, 2]);
    assert!(squeezed.shares_buffer_with(&source));

    let integers = squeezed.astype::<i64>().unwrap();
    assert_eq!(integers.to_vec(), [1, -2, 0, i64::MAX]);
    assert!(integers.is_c_contiguous());
    assert!(!integers.shares_buffer_with(&squeezed.astype::<i64>().unwrap()));

    let transposed = squeezed.transpose();
    let booleans = transposed.astype::<bool>().unwrap();
    assert_eq!(booleans.shape(), &[2, 2]);
    assert_eq!(booleans.to_vec(), [true, true, true, true]);
    assert!(booleans.is_c_contiguous());
}

#[test]
fn astype_supports_all_sixteen_dtype_pairs() {
    let booleans = Array::from_slice(&[false, true], &[2]).unwrap();
    assert_eq!(booleans.astype::<bool>().unwrap().to_vec(), [false, true]);
    assert_eq!(booleans.astype::<i64>().unwrap().to_vec(), [0, 1]);
    assert_eq!(booleans.astype::<f64>().unwrap().to_vec(), [0.0, 1.0]);
    assert_eq!(
        booleans.astype::<Complex64>().unwrap().to_vec(),
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    );

    let integers = Array::from_slice(&[0_i64, -2], &[2]).unwrap();
    assert_eq!(integers.astype::<bool>().unwrap().to_vec(), [false, true]);
    assert_eq!(integers.astype::<i64>().unwrap().to_vec(), [0, -2]);
    assert_eq!(integers.astype::<f64>().unwrap().to_vec(), [0.0, -2.0]);
    assert_eq!(
        integers.astype::<Complex64>().unwrap().to_vec(),
        [Complex64::new(0.0, 0.0), Complex64::new(-2.0, 0.0)]
    );

    let floats = Array::from_slice(&[0.0_f64, -2.9], &[2]).unwrap();
    assert_eq!(floats.astype::<bool>().unwrap().to_vec(), [false, true]);
    assert_eq!(floats.astype::<i64>().unwrap().to_vec(), [0, -2]);
    assert_eq!(floats.astype::<f64>().unwrap().to_vec(), [0.0, -2.9]);
    assert_eq!(
        floats.astype::<Complex64>().unwrap().to_vec(),
        [Complex64::new(0.0, 0.0), Complex64::new(-2.9, 0.0)]
    );

    let complex = Array::from_slice(
        &[Complex64::new(0.0, 0.0), Complex64::new(-2.9, 7.0)],
        &[2],
    )
    .unwrap();
    assert_eq!(complex.astype::<bool>().unwrap().to_vec(), [false, true]);
    assert_eq!(complex.astype::<i64>().unwrap().to_vec(), [0, -2]);
    assert_eq!(complex.astype::<f64>().unwrap().to_vec(), [0.0, -2.9]);
    assert_eq!(
        complex.astype::<Complex64>().unwrap().to_vec(),
        complex.to_vec()
    );
    assert!(
        !complex.shares_buffer_with(&complex.astype::<Complex64>().unwrap())
    );
}

#[test]
fn unique_values_and_metadata() {
    let a = Array::from_slice(&[3_i64, 1, 3, 2], &[4]).unwrap();
    assert_eq!(unique(&a).unwrap().to_vec(), vec![1, 2, 3]);

    let result = unique_with(
        &a,
        UniqueOptions {
            return_index: true,
            return_inverse: true,
            return_counts: true,
        },
    )
    .unwrap();
    assert_eq!(result.values.to_vec(), vec![1, 2, 3]);
    assert_eq!(result.indices.unwrap().to_vec(), vec![1, 3, 0]);
    assert_eq!(result.inverse_indices.unwrap().to_vec(), vec![2, 0, 2, 1]);
    assert_eq!(result.counts.unwrap().to_vec(), vec![1, 1, 2]);

    let nan =
        Array::from_slice(&[3.0_f64, f64::NAN, 1.0, f64::NAN], &[4]).unwrap();
    let values = unique(&nan).unwrap().to_vec();
    assert_eq!(&values[..2], &[1.0, 3.0]);
    assert!(values[2].is_nan());
}

#[test]
fn spaces_and_meshgrid() {
    assert_eq!(
        linspace(0.0, 1.0, 5, true).unwrap().to_vec(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
    assert_eq!(
        logspace(0.0, 2.0, 3, true, 10.0).unwrap().to_vec(),
        vec![1.0, 10.0, 100.0]
    );
    let geometric = geomspace(1.0, 16.0, 5, true).unwrap().to_vec();
    for (actual, expected) in geometric.iter().zip([1.0, 2.0, 4.0, 8.0, 16.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }

    let x = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    let y = Array::from_slice(&[10_i64, 20], &[2]).unwrap();
    let xy = meshgrid(&[&x, &y], MeshgridIndexing::Xy).unwrap();
    assert_eq!(xy[0].shape(), &[2, 3]);
    assert_eq!(xy[0].to_vec(), vec![1, 2, 3, 1, 2, 3]);
    assert_eq!(xy[1].to_vec(), vec![10, 10, 10, 20, 20, 20]);
}
