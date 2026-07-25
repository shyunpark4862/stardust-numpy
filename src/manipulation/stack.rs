use crate::array::{insert_axis_view, Array};
use crate::axis::normalize_insert_axis;
use crate::dtype::Scalar;
use crate::error::Result;

use super::concatenate;

/// Join arrays along a newly inserted `axis`.
pub fn stack<T: Scalar>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>> {
    debug_assert!(!arrays.is_empty(), "stack requires at least one array");
    let first = arrays[0];
    let axis = normalize_insert_axis(axis, first.ndim());

    let views: Vec<_> = arrays
        .iter()
        .map(|array| insert_axis_view(array, axis as isize))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, axis as isize)
}

/// Stack arrays vertically (row-wise).
pub fn vstack<T: Scalar>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    debug_assert!(!arrays.is_empty(), "vstack requires at least one array");
    let views: Vec<_> = arrays
        .iter()
        .map(|array| promote_at_least_2d(array))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, 0)
}

/// Stack arrays horizontally (column-wise).
pub fn hstack<T: Scalar>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    debug_assert!(!arrays.is_empty(), "hstack requires at least one array");
    let views: Vec<_> = arrays
        .iter()
        .map(|array| promote_at_least_1d(array))
        .collect::<Result<_>>()?;
    let axis = if views[0].ndim() == 1 { 0 } else { 1 };
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, axis)
}

fn promote_at_least_1d<T: Scalar>(array: &Array<T>) -> Result<Array<T>> {
    if array.ndim() == 0 {
        insert_axis_view(array, 0)
    } else {
        Ok(array.view())
    }
}

fn promote_at_least_2d<T: Scalar>(array: &Array<T>) -> Result<Array<T>> {
    match array.ndim() {
        0 => {
            let vector = insert_axis_view(array, 0)?;
            insert_axis_view(&vector, 0)
        }
        1 => insert_axis_view(array, 0),
        _ => Ok(array.view()),
    }
}
