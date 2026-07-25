use sdnp::{
    gather, ndenumerate, ndindex, nditer, scatter, scatter_array, Array, Error,
    IndexSpec,
};

#[test]
fn normalized_basic_indexing_handles_reverse_ellipsis_and_new_axis() {
    let array = Array::from_vec((0_i64..12).collect(), &[3, 4]).unwrap();
    let reversed = gather(
        &array,
        &[IndexSpec::slice(None, None, Some(-1)), IndexSpec::Ellipsis],
    )
    .unwrap();
    assert_eq!(reversed.shape(), &[3, 4]);
    assert_eq!(
        reversed.to_vec(),
        vec![8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    );

    let expanded = gather(
        &array,
        &[IndexSpec::NewAxis, IndexSpec::index(-1), IndexSpec::full()],
    )
    .unwrap();
    assert_eq!(expanded.shape(), &[1, 4]);
    assert_eq!(expanded.to_vec(), vec![8, 9, 10, 11]);
}

#[test]
fn fancy_and_boolean_indexing_follow_c_order() {
    let array = Array::from_vec((0_i64..12).collect(), &[3, 4]).unwrap();
    let rows = Array::from_slice(&[2_i64, 0], &[2]).unwrap();
    let fancy = gather(
        &array,
        &[IndexSpec::IntegerArray(rows), IndexSpec::index(1)],
    )
    .unwrap();
    assert_eq!(fancy.shape(), &[2]);
    assert_eq!(fancy.to_vec(), vec![9, 1]);

    let mask = Array::from_slice(
        &[
            true, false, false, true, false, false, false, false, true, false,
            false, false,
        ],
        &[3, 4],
    )
    .unwrap();
    let selected = gather(&array, &[IndexSpec::BoolArray(mask)]).unwrap();
    assert_eq!(selected.to_vec(), vec![0, 3, 8]);
}

#[test]
fn scatter_scalar_and_array_detach_shared_storage() {
    let source = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();
    let mut target = source.clone();
    scatter(&mut target, &[IndexSpec::index(0), IndexSpec::full()], 9).unwrap();
    assert_eq!(target.to_vec(), vec![9, 9, 9, 3, 4, 5]);
    assert_eq!(source.to_vec(), vec![0, 1, 2, 3, 4, 5]);

    let values = Array::from_slice(&[7_i64, 8, 9], &[3]).unwrap();
    scatter_array(
        &mut target,
        &[IndexSpec::index(1), IndexSpec::full()],
        &values,
    )
    .unwrap();
    assert_eq!(target.to_vec(), vec![9, 9, 9, 7, 8, 9]);
}

#[test]
fn indexing_errors_are_structured() {
    let array = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert!(matches!(
        gather(&array, &[IndexSpec::index(3)]),
        Err(Error::IndexOutOfBounds { .. })
    ));
    assert!(
        gather(&array, &[IndexSpec::Ellipsis, IndexSpec::Ellipsis]).is_err()
    );
    assert!(
        gather(&array, &[IndexSpec::index(0), IndexSpec::index(0)]).is_err()
    );
}

#[test]
fn core_iterators_cover_empty_strided_and_broadcast_inputs() {
    assert_eq!(
        ndindex(&[2, 2]).unwrap().collect::<Vec<_>>(),
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );

    let base = Array::from_vec((0_i64..6).collect(), &[2, 3]).unwrap();
    let transposed = base.transpose();
    let enumerated = ndenumerate(&transposed).collect::<Vec<_>>();
    assert_eq!(enumerated[0], (vec![0, 0], 0));
    assert_eq!(enumerated[1], (vec![0, 1], 3));
    assert_eq!(enumerated.len(), 6);

    let column = Array::from_slice(&[10_i64, 20], &[2, 1]).unwrap();
    let row = Array::from_slice(&[1_i64, 2, 3], &[1, 3]).unwrap();
    let iter = nditer(&[&column, &row]).unwrap().collect::<Vec<_>>();
    assert_eq!(
        iter,
        vec![
            vec![10, 1],
            vec![10, 2],
            vec![10, 3],
            vec![20, 1],
            vec![20, 2],
            vec![20, 3],
        ]
    );
}
