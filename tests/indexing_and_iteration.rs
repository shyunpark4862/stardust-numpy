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
fn direct_basic_indexing_handles_omitted_axes_and_reverse_slices() {
    let array = Array::from_vec((0_i64..12).collect(), &[3, 4]).unwrap();
    let direct =
        gather(&array, &[IndexSpec::slice(None, None, Some(-1))]).unwrap();
    let general = gather(
        &array,
        &[IndexSpec::slice(None, None, Some(-1)), IndexSpec::Ellipsis],
    )
    .unwrap();

    assert_eq!(direct.shape(), &[3, 4]);
    assert_eq!(direct.to_vec(), vec![8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]);
    assert_eq!(direct.to_vec(), general.to_vec());

    let direct_last = gather(&array, &[IndexSpec::index(-1)]).unwrap();
    let general_last =
        gather(&array, &[IndexSpec::index(-1), IndexSpec::Ellipsis]).unwrap();
    assert_eq!(direct_last.to_vec(), general_last.to_vec());

    let empty = gather(&array, &[IndexSpec::slice(Some(0), Some(0), Some(-1))])
        .unwrap();
    assert_eq!(empty.shape(), &[0, 4]);
    assert!(empty.to_vec().is_empty());
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
fn direct_basic_scatter_handles_slices_arrays_and_omitted_axes() {
    let source = Array::from_vec((0_i64..12).collect(), &[3, 4]).unwrap();
    let mut direct = source.clone();
    let mut general = source.clone();
    let reversed_rows = IndexSpec::slice(None, None, Some(-2));
    scatter(&mut direct, std::slice::from_ref(&reversed_rows), 9).unwrap();
    scatter(&mut general, &[reversed_rows, IndexSpec::Ellipsis], 9).unwrap();
    assert_eq!(direct.to_vec(), vec![9, 9, 9, 9, 4, 5, 6, 7, 9, 9, 9, 9]);
    assert_eq!(direct.to_vec(), general.to_vec());
    assert_eq!(source.to_vec(), (0_i64..12).collect::<Vec<_>>());

    let values = Array::from_slice(&[1_i64, 2, 3, 4], &[4]).unwrap();
    scatter_array(&mut direct, &[IndexSpec::index(1)], &values).unwrap();
    scatter_array(
        &mut general,
        &[IndexSpec::index(1), IndexSpec::Ellipsis],
        &values,
    )
    .unwrap();
    assert_eq!(direct.to_vec(), vec![9, 9, 9, 9, 1, 2, 3, 4, 9, 9, 9, 9]);
    assert_eq!(direct.to_vec(), general.to_vec());
}

#[test]
fn indexing_errors_are_structured() {
    let array = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert!(matches!(
        gather(&array, &[IndexSpec::index(3)]),
        Err(Error::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        gather(&array, &[IndexSpec::Ellipsis, IndexSpec::Ellipsis]),
        Err(Error::InvalidIndex(_))
    ));
    assert!(matches!(
        gather(&array, &[IndexSpec::index(0), IndexSpec::index(0)]),
        Err(Error::InvalidIndex(_))
    ));
    assert!(matches!(
        gather(&array, &[IndexSpec::slice(None, None, Some(0))]),
        Err(Error::InvalidArgument(_))
    ));

    let bad_mask = Array::from_slice(&[true, false], &[2]).unwrap();
    assert!(matches!(
        gather(&array, &[IndexSpec::BoolArray(bad_mask)]),
        Err(Error::InvalidIndex(_))
    ));
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
