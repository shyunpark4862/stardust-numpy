//! Phase 3: basic / fancy / boolean indexing and assignment.

use sdnp::{gather, scatter, scatter_array, Array, IndexSpec, Result};

fn ix(i: i64) -> IndexSpec {
    IndexSpec::index(i)
}

fn sl(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> IndexSpec {
    IndexSpec::slice(start, stop, step)
}

fn full() -> IndexSpec {
    IndexSpec::full()
}

fn int_arr(data: &[i64]) -> IndexSpec {
    IndexSpec::IntegerArray(Array::from_slice(data, &[data.len()]).unwrap())
}

fn bool_arr(data: &[bool]) -> IndexSpec {
    IndexSpec::BoolArray(Array::from_slice(data, &[data.len()]).unwrap())
}

#[test]
fn negative_integer_index() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30], &[3])?;
    let s = gather(&a, &[ix(-1)])?;
    assert_eq!(s.shape(), &[] as &[usize]);
    assert_eq!(s.item()?, 30);
    assert!(s.shares_buffer_with(&a));
    Ok(())
}

#[test]
fn integer_out_of_range() {
    let a = Array::from_slice(&[10_i64, 20, 30], &[3]).unwrap();
    assert!(gather(&a, &[ix(3)]).is_err());
    assert!(gather(&a, &[ix(-4)]).is_err());
}

#[test]
fn basic_slice_is_view() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40, 50], &[5])?;
    let mut b = gather(&a, &[sl(Some(1), Some(4), None)])?;
    assert_eq!(b.shape(), &[3]);
    assert_eq!(b.get(&[0])?, 20);
    assert!(b.shares_buffer_with(&a));
    b.set(&[0], 999)?;
    // CoW: writing through the view detaches; original stays unless sole owner.
    assert_eq!(a.get(&[1])?, 20);
    assert_eq!(b.get(&[0])?, 999);
    assert!(!b.shares_buffer_with(&a));
    assert_eq!(b.as_buffer().len(), b.size());
    assert_eq!(b.offset(), 0);
    assert!(b.is_c_contiguous());
    Ok(())
}

#[test]
fn slice_with_step() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40, 50, 60], &[6])?;
    let b = gather(&a, &[sl(Some(1), Some(6), Some(2))])?;
    assert_eq!(b.to_vec(), vec![20, 40, 60]);
    assert_eq!(b.strides(), &[2]);
    Ok(())
}

#[test]
fn reverse_slice() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40], &[4])?;
    let b = gather(&a, &[sl(None, None, Some(-1))])?;
    assert_eq!(b.to_vec(), vec![40, 30, 20, 10]);
    assert_eq!(b.strides(), &[-1]);
    Ok(())
}

#[test]
fn partial_integer_row_view() -> Result<()> {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let row = gather(&a, &[ix(1)])?;
    assert_eq!(row.shape(), &[3]);
    assert_eq!(row.to_vec(), vec![4, 5, 6]);
    assert!(row.shares_buffer_with(&a));
    Ok(())
}

#[test]
fn newaxis_and_ellipsis() -> Result<()> {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let b = gather(&a, &[IndexSpec::Ellipsis, ix(1)])?;
    assert_eq!(b.shape(), &[2]);
    assert_eq!(b.to_vec(), vec![2, 5]);

    let c = gather(&a, &[IndexSpec::NewAxis, full(), full()])?;
    assert_eq!(c.shape(), &[1, 2, 3]);
    Ok(())
}

#[test]
fn fancy_1d() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40], &[4])?;
    let b = gather(&a, &[int_arr(&[2, 0, -1])])?;
    assert_eq!(b.to_vec(), vec![30, 10, 40]);
    assert!(!b.shares_buffer_with(&a));
    Ok(())
}

#[test]
fn fancy_duplicates_and_copy() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30], &[3])?;
    let mut b = gather(&a, &[int_arr(&[1, 1, 0])])?;
    assert_eq!(b.to_vec(), vec![20, 20, 10]);
    b.set(&[0], 999)?;
    assert_eq!(a.to_vec(), vec![10, 20, 30]);
    Ok(())
}

#[test]
fn boolean_mask_1d() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40], &[4])?;
    let b = gather(&a, &[bool_arr(&[true, false, true, false])])?;
    assert_eq!(b.to_vec(), vec![10, 30]);
    Ok(())
}

#[test]
fn boolean_mask_length_mismatch() {
    let a = Array::from_slice(&[10_i64, 20, 30], &[3]).unwrap();
    assert!(gather(&a, &[bool_arr(&[true, false])]).is_err());
}

#[test]
fn boolean_mask_first_axis_2d() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40, 50, 60], &[3, 2])?;
    let b = gather(&a, &[bool_arr(&[true, false, true])])?;
    assert_eq!(b.shape(), &[2, 2]);
    assert_eq!(b.to_vec(), vec![10, 20, 50, 60]);
    Ok(())
}

#[test]
fn boolean_elementwise_2d() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2])?;
    let mask = Array::from_slice(&[false, true, true, false], &[2, 2])?;
    let b = gather(&a, &[IndexSpec::BoolArray(mask)])?;
    assert_eq!(b.shape(), &[2]);
    assert_eq!(b.to_vec(), vec![20, 30]);
    Ok(())
}

#[test]
fn fancy_pairwise_2d() -> Result<()> {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    // a[[0,1], [2,0]] → [3, 4]
    let b = gather(&a, &[int_arr(&[0, 1]), int_arr(&[2, 0])])?;
    assert_eq!(b.shape(), &[2]);
    assert_eq!(b.to_vec(), vec![3, 4]);
    Ok(())
}

#[test]
fn fancy_with_slice_column() -> Result<()> {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let b = gather(&a, &[full(), int_arr(&[2, 0])])?;
    assert_eq!(b.shape(), &[2, 2]);
    assert_eq!(b.get(&[0, 0])?, 3);
    assert_eq!(b.get(&[0, 1])?, 1);
    assert_eq!(b.get(&[1, 0])?, 6);
    assert_eq!(b.get(&[1, 1])?, 4);
    Ok(())
}

#[test]
fn separated_fancy_prepends_shape() -> Result<()> {
    // a[[0,1], :, [0,1]] on (2,2,2) — non-contiguous fancy → fancy shape first
    let data: Vec<i64> = (0..8).collect();
    let a = Array::from_vec(data, &[2, 2, 2])?;
    let b = gather(&a, &[int_arr(&[0, 1]), full(), int_arr(&[0, 1])])?;
    assert_eq!(b.shape(), &[2, 2]);
    // result[f, mid] = a[rows[f], mid, cols[f]]
    assert_eq!(b.get(&[0, 0])?, 0); // a[0,0,0]
    assert_eq!(b.get(&[0, 1])?, 2); // a[0,1,0]
    assert_eq!(b.get(&[1, 0])?, 5); // a[1,0,1]
    assert_eq!(b.get(&[1, 1])?, 7); // a[1,1,1]
    Ok(())
}

#[test]
fn separated_fancy_with_leading_slice() -> Result<()> {
    // a[:, [0,1], :, [0,1]] — fancy slots non-contiguous and not starting at 0
    // result layout: fancy | before | after → (2, 2, 2)
    let data: Vec<i64> = (0..16).collect();
    let a = Array::from_vec(data, &[2, 2, 2, 2])?;
    let b = gather(&a, &[full(), int_arr(&[0, 1]), full(), int_arr(&[0, 1])])?;
    assert_eq!(b.shape(), &[2, 2, 2]);
    // result[f, i, k] = a[i, rows[f], k, cols[f]]
    assert_eq!(b.get(&[0, 0, 0])?, 0); // a[0,0,0,0]
    assert_eq!(b.get(&[0, 0, 1])?, 2); // a[0,0,1,0]
    assert_eq!(b.get(&[0, 1, 0])?, 8); // a[1,0,0,0]
    assert_eq!(b.get(&[1, 0, 0])?, 5); // a[0,1,0,1]
    assert_eq!(b.get(&[1, 1, 1])?, 15); // a[1,1,1,1]
    Ok(())
}

#[test]
fn scatter_scalar_slice() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    scatter(&mut a, &[full(), ix(1)], 99)?;
    assert_eq!(a.to_vec(), vec![1, 99, 3, 4, 99, 6]);
    Ok(())
}

#[test]
fn scatter_on_shared_view_materializes_logical_storage() -> Result<()> {
    let a = Array::from_slice(&[10_i64, 20, 30, 40, 50, 60], &[6])?;
    let mut view = gather(&a, &[sl(Some(1), Some(6), Some(2))])?;
    assert_eq!(view.to_vec(), vec![20, 40, 60]);
    assert!(view.shares_buffer_with(&a));

    scatter(&mut view, &[full()], 7)?;

    assert_eq!(a.to_vec(), vec![10, 20, 30, 40, 50, 60]);
    assert_eq!(view.to_vec(), vec![7, 7, 7]);
    assert!(!view.shares_buffer_with(&a));
    assert_eq!(view.as_buffer().len(), view.size());
    assert_eq!(view.offset(), 0);
    assert!(view.is_c_contiguous());
    Ok(())
}

#[test]
fn scatter_array_slice() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let vals = Array::from_slice(&[7_i64, 8], &[2])?;
    scatter_array(&mut a, &[ix(0), sl(Some(1), None, None)], &vals)?;
    assert_eq!(a.to_vec(), vec![1, 7, 8, 4, 5, 6]);
    Ok(())
}

#[test]
fn scatter_array_broadcast() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let vals = Array::from_slice(&[99_i64], &[1])?;
    scatter_array(&mut a, &[full(), ix(1)], &vals)?;
    assert_eq!(a.to_vec(), vec![1, 99, 3, 4, 99, 6]);
    Ok(())
}

#[test]
fn scatter_scalar_column_step() -> Result<()> {
    // [:, ::2] on a 3x4 array: column stride 2, row gap between axes.
    let mut a = Array::from_vec((0_i64..12).collect(), &[3, 4])?;
    scatter(&mut a, &[full(), sl(None, None, Some(2))], 0)?;
    assert_eq!(a.to_vec(), vec![0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11]);
    Ok(())
}

#[test]
fn scatter_scalar_row_gap_slice() -> Result<()> {
    // Select every other row out of a 4x3 array: rows have a stride-6 gap.
    let mut a = Array::from_vec((0_i64..12).collect(), &[4, 3])?;
    scatter(&mut a, &[sl(Some(0), None, Some(2)), full()], 99)?;
    assert_eq!(a.to_vec(), vec![99, 99, 99, 3, 4, 5, 99, 99, 99, 9, 10, 11]);
    Ok(())
}

#[test]
fn scatter_scalar_multi_axis_strided_slice() -> Result<()> {
    // Step on both axes of a 4x6 array.
    let mut a = Array::from_vec((0_i64..24).collect(), &[4, 6])?;
    scatter(
        &mut a,
        &[sl(Some(1), None, Some(2)), sl(Some(0), None, Some(3))],
        0,
    )?;
    let expected: Vec<i64> = (0..24)
        .enumerate()
        .map(|(i, x)| {
            let row = i / 6;
            let col = i % 6;
            if row % 2 == 1 && row >= 1 && col % 3 == 0 {
                0
            } else {
                x
            }
        })
        .collect();
    assert_eq!(a.to_vec(), expected);
    Ok(())
}

#[test]
fn scatter_scalar_negative_step() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4, 5], &[5])?;
    scatter(&mut a, &[sl(Some(3), None, Some(-2))], 0)?;
    // indices 3, 1 (step -2 from 3) get zeroed.
    assert_eq!(a.to_vec(), vec![1, 0, 3, 0, 5]);
    Ok(())
}

#[test]
fn scatter_scalar_empty_slice_is_noop() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4], &[4])?;
    scatter(&mut a, &[sl(Some(2), Some(2), None)], 0)?;
    assert_eq!(a.to_vec(), vec![1, 2, 3, 4]);
    Ok(())
}

#[test]
fn scatter_scalar_zero_d_integer_selection() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3], &[3])?;
    scatter(&mut a, &[ix(1)], 42)?;
    assert_eq!(a.to_vec(), vec![1, 42, 3]);
    Ok(())
}

#[test]
fn scatter_scalar_newaxis_basic() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    scatter(&mut a, &[IndexSpec::NewAxis, full(), full()], 0)?;
    assert_eq!(a.to_vec(), vec![0; 6]);
    Ok(())
}

#[test]
fn scatter_array_row_gap_destination_strided_source() -> Result<()> {
    let mut a = Array::from_vec((0_i64..12).collect(), &[4, 3])?;
    // Source is itself a non-contiguous view (reversed columns).
    let src_base = Array::from_slice(&[7_i64, 8, 9], &[3])?;
    let src = gather(&src_base, &[sl(None, None, Some(-1))])?;
    scatter_array(&mut a, &[sl(Some(0), None, Some(2)), full()], &src)?;
    assert_eq!(a.to_vec(), vec![9, 8, 7, 3, 4, 5, 9, 8, 7, 9, 10, 11]);
    Ok(())
}

#[test]
fn scatter_scalar_nonzero_base_offset() -> Result<()> {
    let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3])?;
    let mut view = gather(&a, &[ix(1), full()])?;
    assert_eq!(view.offset(), 3);
    scatter(&mut view, &[sl(None, None, Some(2))], 0)?;
    assert_eq!(view.to_vec(), vec![0, 5, 0]);
    Ok(())
}

#[test]
fn scatter_fancy_scalar() -> Result<()> {
    let mut a = Array::from_slice(&[10_i64, 20, 30, 40], &[4])?;
    scatter(&mut a, &[int_arr(&[0, 2])], 7)?;
    assert_eq!(a.to_vec(), vec![7, 20, 7, 40]);
    Ok(())
}

#[test]
fn scatter_fancy_overlap_safe() -> Result<()> {
    let mut a = Array::from_slice(&[10_i64, 20, 30, 40], &[4])?;
    let src = gather(&a, &[int_arr(&[1, 0])])?;
    scatter_array(&mut a, &[int_arr(&[0, 1])], &src)?;
    assert_eq!(a.to_vec(), vec![20, 10, 30, 40]);
    Ok(())
}

#[test]
fn scatter_basic_shared_source_overlap_safe() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3, 4], &[4])?;
    let src = gather(&a, &[sl(Some(0), Some(3), None)])?;
    assert!(src.shares_buffer_with(&a));

    scatter_array(&mut a, &[sl(Some(1), Some(4), None)], &src)?;

    assert_eq!(a.to_vec(), vec![1, 1, 2, 3]);
    Ok(())
}

#[test]
fn scatter_boolean() -> Result<()> {
    let mut a = Array::from_slice(&[10_i64, 20, 30, 40], &[4])?;
    scatter(&mut a, &[bool_arr(&[true, false, true, false])], 0)?;
    assert_eq!(a.to_vec(), vec![0, 20, 0, 40]);
    Ok(())
}

#[test]
fn method_wrappers() -> Result<()> {
    let mut a = Array::from_slice(&[1_i64, 2, 3], &[3])?;
    let b = a.gather(&[ix(-1)])?;
    assert_eq!(b.item()?, 3);
    a.scatter(&[ix(0)], 9)?;
    assert_eq!(a.get(&[0])?, 9);
    Ok(())
}
