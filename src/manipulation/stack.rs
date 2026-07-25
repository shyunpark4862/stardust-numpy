use crate::array::{insert_axis_view, Array};
use crate::axis::normalize_insert_axis;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

use super::concatenate;

/// Join arrays along a newly inserted `axis`.
///
/// Every input must have exactly the same shape. Negative axes count backward
/// from the result rank. The result is newly allocated and C-contiguous.
pub fn stack<T: Scalar>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>> {
    let first = require_arrays(arrays, "stack")?;
    let axis = normalize_insert_axis(axis, first.ndim())?;
    for (index, array) in arrays.iter().enumerate().skip(1) {
        if array.shape() != first.shape() {
            return Err(Error::InvalidArgument(format!(
                "all arrays must have the same shape; array 0 has shape {:?}, \
                 array {index} has shape {:?}",
                first.shape(),
                array.shape()
            )));
        }
    }

    let views: Vec<_> = arrays
        .iter()
        .map(|array| insert_axis_view(array, axis as isize))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, axis as isize)
}

/// Stack arrays vertically (row-wise).
///
/// Zero-dimensional inputs are promoted to shape `(1, 1)` and one-dimensional
/// inputs to `(1, N)` before concatenation along axis 0. Higher-dimensional
/// inputs are unchanged.
pub fn vstack<T: Scalar>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    require_arrays(arrays, "vstack")?;
    let views: Vec<_> = arrays
        .iter()
        .map(|array| promote_at_least_2d(array))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, 0)
}

/// Stack arrays horizontally (column-wise).
///
/// Zero-dimensional inputs are promoted to length-one vectors. If the
/// promoted inputs are one-dimensional they are concatenated along axis 0;
/// otherwise they are concatenated along axis 1, matching NumPy's `hstack`
/// dimensional promotion.
pub fn hstack<T: Scalar>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    require_arrays(arrays, "hstack")?;
    let views: Vec<_> = arrays
        .iter()
        .map(|array| promote_at_least_1d(array))
        .collect::<Result<_>>()?;
    let axis = if views[0].ndim() == 1 { 0 } else { 1 };
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, axis)
}

fn require_arrays<'a, T: Scalar>(
    arrays: &'a [&Array<T>],
    operation: &str,
) -> Result<&'a Array<T>> {
    arrays.first().copied().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{operation} requires at least one array"
        ))
    })
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
