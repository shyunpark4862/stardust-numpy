//! Phase 5: joining, selection, sorting, unique, and spaces.

use sdnp::{
    argsort, clip, concatenate, geomspace, hstack, linspace, logspace,
    meshgrid, nonzero, sort, sort_in_place, stack, unique, unique_with, vstack,
    where_, Array, MeshgridIndexing, UniqueOptions,
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
fn sorting_axes_flatten_and_cow() {
    let a = Array::from_slice(&[3_i64, 1, 2, 6, 4, 5], &[2, 3]).unwrap();
    assert_eq!(sort(&a, Some(-1)).unwrap().to_vec(), vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(argsort(&a, None).unwrap().to_vec(), vec![1, 2, 0, 4, 5, 3]);

    let mut shared = a.clone();
    sort_in_place(&mut shared, None).unwrap();
    assert_eq!(shared.shape(), &[2, 3]);
    assert_eq!(shared.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a.to_vec(), vec![3, 1, 2, 6, 4, 5]);
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
