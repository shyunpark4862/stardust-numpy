//! Phase 7: NumPy-style shape and array iteration.

use sdnp::{
    gather, ndenumerate, ndindex, nditer, Array, Error, IndexSpec, Result,
};

#[test]
fn ndindex_walks_c_order_and_tracks_remaining_length() -> Result<()> {
    let mut indices = ndindex(&[2, 3])?;
    assert_eq!(indices.len(), 6);
    assert_eq!(indices.next(), Some(vec![0, 0]));
    assert_eq!(indices.len(), 5);
    assert_eq!(
        indices.collect::<Vec<_>>(),
        vec![vec![0, 1], vec![0, 2], vec![1, 0], vec![1, 1], vec![1, 2],]
    );

    assert_eq!(ndindex(&[])?.collect::<Vec<_>>(), vec![Vec::new()]);
    assert_eq!(ndindex(&[2, 0, 3])?.count(), 0);
    Ok(())
}

#[test]
fn flat_iter_uses_logical_c_order_for_all_layouts() {
    let matrix = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    assert_eq!(matrix.flat().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);

    let transposed = matrix.transpose();
    assert_eq!(
        transposed.flat().collect::<Vec<_>>(),
        vec![1, 4, 2, 5, 3, 6]
    );

    let reversed = gather(
        &matrix,
        &[IndexSpec::full(), IndexSpec::slice(None, None, Some(-1))],
    )
    .unwrap();
    assert_eq!(reversed.flat().collect::<Vec<_>>(), vec![3, 2, 1, 6, 5, 4]);
}

#[test]
fn flat_iter_handles_empty_and_zero_dimensional_arrays() {
    let empty = Array::from_vec(Vec::<i64>::new(), &[0, 3]).unwrap();
    assert_eq!(empty.flat().len(), 0);
    assert_eq!(empty.flat().collect::<Vec<_>>(), Vec::<i64>::new());

    let scalar = Array::from_slice(&[7_i64], &[]).unwrap();
    assert_eq!(scalar.flat().collect::<Vec<_>>(), vec![7]);
}

#[test]
fn ndenumerate_pairs_coordinates_with_logical_values() {
    let matrix = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2]).unwrap();
    assert_eq!(
        ndenumerate(&matrix).collect::<Vec<_>>(),
        vec![
            (vec![0, 0], 10),
            (vec![0, 1], 20),
            (vec![1, 0], 30),
            (vec![1, 1], 40),
        ]
    );

    let transposed = matrix.transpose();
    assert_eq!(
        ndenumerate(&transposed).collect::<Vec<_>>(),
        vec![
            (vec![0, 0], 10),
            (vec![0, 1], 30),
            (vec![1, 0], 20),
            (vec![1, 1], 40),
        ]
    );

    let scalar = Array::from_slice(&[5_i64], &[]).unwrap();
    assert_eq!(ndenumerate(&scalar).collect::<Vec<_>>(), vec![(vec![], 5)]);
}

#[test]
fn nditer_walks_single_and_multiple_operands() -> Result<()> {
    let left = Array::from_slice(&[1_i64, 2, 3], &[3])?;
    let right = Array::from_slice(&[10_i64, 20, 30], &[3])?;

    assert_eq!(
        nditer(&[&left])?.collect::<Vec<_>>(),
        vec![vec![1], vec![2], vec![3]]
    );
    assert_eq!(
        nditer(&[&left, &right])?.collect::<Vec<_>>(),
        vec![vec![1, 10], vec![2, 20], vec![3, 30]]
    );
    Ok(())
}

#[test]
fn nditer_broadcasts_and_handles_strided_operands() -> Result<()> {
    let left = Array::from_slice(&[1_i64, 2, 3], &[3, 1])?;
    let right = Array::from_slice(&[10_i64, 20, 30], &[3])?;
    assert_eq!(
        nditer(&[&left, &right])?.collect::<Vec<_>>(),
        vec![
            vec![1, 10],
            vec![1, 20],
            vec![1, 30],
            vec![2, 10],
            vec![2, 20],
            vec![2, 30],
            vec![3, 10],
            vec![3, 20],
            vec![3, 30],
        ]
    );

    let matrix = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let transposed = matrix.transpose();
    let tens = Array::from_slice(&[10_i64; 6], &[3, 2])?;
    assert_eq!(
        nditer(&[&transposed, &tens])?.collect::<Vec<_>>(),
        vec![
            vec![1, 10],
            vec![4, 10],
            vec![2, 10],
            vec![5, 10],
            vec![3, 10],
            vec![6, 10],
        ]
    );
    Ok(())
}

#[test]
fn nditer_validates_operands_and_empty_shapes() {
    let left = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
    let right = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
    assert!(matches!(nditer::<i64>(&[]), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        nditer(&[&left, &right]),
        Err(Error::Broadcast { .. })
    ));

    let empty = Array::from_vec(Vec::<i64>::new(), &[0, 3]).unwrap();
    assert_eq!(nditer(&[&empty]).unwrap().count(), 0);

    let scalar = Array::from_slice(&[9_i64], &[]).unwrap();
    assert_eq!(
        nditer(&[&scalar]).unwrap().collect::<Vec<_>>(),
        vec![vec![9]]
    );
}

#[test]
fn axis0_iteration_returns_cached_shared_views() -> Result<()> {
    let matrix = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    assert_eq!(matrix.axis0_len()?, 2);

    let mut rows = matrix.iter_axis0()?;
    assert_eq!(rows.len(), 2);
    let first = rows.next().unwrap();
    let second = rows.next().unwrap();
    assert_eq!(first.shape(), &[3]);
    assert_eq!(first.to_vec(), vec![1, 2, 3]);
    assert_eq!(second.to_vec(), vec![4, 5, 6]);
    assert!(first.shares_buffer_with(&matrix));
    assert!(second.shares_buffer_with(&matrix));

    let vector = Array::from_slice(&[10_i64, 20], &[2])?;
    let scalars = vector.iter_axis0()?.collect::<Vec<_>>();
    assert_eq!(scalars[0].shape(), &[] as &[usize]);
    assert_eq!(scalars[0].item()?, 10);
    assert_eq!(scalars[1].item()?, 20);
    Ok(())
}

#[test]
fn axis0_iteration_preserves_layout_and_copy_on_write() -> Result<()> {
    let matrix = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let reversed = gather(
        &matrix,
        &[IndexSpec::slice(None, None, Some(-1)), IndexSpec::full()],
    )?;
    let rows = reversed.iter_axis0()?.collect::<Vec<_>>();
    assert_eq!(rows[0].to_vec(), vec![4, 5, 6]);
    assert_eq!(rows[1].to_vec(), vec![1, 2, 3]);

    let mut row = matrix.iter_axis0()?.next().unwrap();
    assert!(row.shares_buffer_with(&matrix));
    row.set(&[0], 99)?;
    assert_eq!(matrix.get(&[0, 0])?, 1);
    assert_eq!(row.get(&[0])?, 99);
    assert!(!row.shares_buffer_with(&matrix));
    Ok(())
}

#[test]
fn axis0_iteration_handles_empty_axes_and_rejects_zero_dimensional_arrays() {
    let no_rows = Array::from_vec(Vec::<i64>::new(), &[0, 3]).unwrap();
    assert_eq!(no_rows.iter_axis0().unwrap().count(), 0);

    let empty_rows = Array::from_vec(Vec::<i64>::new(), &[3, 0]).unwrap();
    let rows = empty_rows.iter_axis0().unwrap().collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.shape() == [0]));

    let scalar = Array::from_slice(&[1_i64], &[]).unwrap();
    assert!(matches!(scalar.axis0_len(), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        scalar.iter_axis0(),
        Err(Error::InvalidArgument(_))
    ));
}
