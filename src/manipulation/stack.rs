//! Stack arrays by inserting a new axis (`np.stack`, `vstack`, `hstack`).
//!
//! [`stack`] inserts `axis` then concatenates views. [`vstack`] and
//! [`hstack`] promote 0-D/1-D inputs to the ranks NumPy expects before joining.

use crate::array::{insert_axis_view, Array};
use crate::axis::resolve_insert_axis;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

use super::concatenate;

/// Join arrays along a newly inserted `axis`.
///
/// Like NumPy's `np.stack`: each input receives a length-1 axis inserted at
/// `axis`, then the expanded arrays are concatenated along that same axis.
/// All inputs must share the same shape. Negative `axis` counts from the end
/// of the *output* rank (`ndim + 1`).
///
/// **Axis rules:** `axis` must lie in `[-(ndim + 1), ndim + 1)` where
/// `ndim` is the input rank. Output rank is `ndim + 1`; the stacked axis
/// has length equal to the number of inputs.
///
/// # Arguments
///
/// * `arrays` - Non-empty slice of same-shaped arrays.
/// * `axis` - Position at which to insert the new axis (may be negative).
///
/// # Returns
///
/// A new C-contiguous [`Array`] with rank one greater than the inputs.
///
/// # Errors
///
/// * [`Error::AxisOutOfBounds`](crate::Error::AxisOutOfBounds) - `axis` is
///   outside the output rank.
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   or offset overflow during concatenation.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{stack, Array};
///
/// let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
/// let b = Array::from_slice(&[3_i64, 4], &[2]).unwrap();
/// let s = stack(&[&a, &b], 0).unwrap();
/// assert_eq!(s.shape(), &[2, 2]);
/// ```
pub fn stack<T: Scalar>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>> {
    let first = arrays
        .first()
        .copied()
        .ok_or(Error::EmptyOperands { op: "stack" })?;
    for (index, array) in arrays.iter().enumerate().skip(1) {
        if array.shape() != first.shape() {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: first.shape().to_vec(),
                index,
                actual: array.shape().to_vec(),
            });
        }
    }
    let axis = resolve_insert_axis(axis, first.ndim())?;

    let views: Vec<_> = arrays
        .iter()
        .map(|array| insert_axis_view(array, axis as isize))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, axis as isize)
}

/// Stack arrays vertically (row-wise), like NumPy's `np.vstack`.
///
/// Scalar (0-D) inputs are promoted to `(1, 1)` and 1-D inputs to `(1, n)`
/// before concatenation along axis 0. Higher-rank inputs are stacked as-is
/// on axis 0.
///
/// # Arguments
///
/// * `arrays` - Arrays to join row-wise after rank promotion.
///
/// # Returns
///
/// A new C-contiguous [`Array`] with one more row per input along axis 0.
///
/// # Errors
///
/// Same as [`concatenate`](crate::concatenate): shape mismatch or allocation
/// failures surface as [`Error::InvalidArgument`](crate::Error::InvalidArgument).
///
/// # Examples
///
/// ```rust
/// use sdnp::{vstack, Array};
///
/// let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
/// let b = Array::from_slice(&[3_i64, 4], &[2]).unwrap();
/// let v = vstack(&[&a, &b]).unwrap();
/// assert_eq!(v.shape(), &[2, 2]);
/// ```
pub fn vstack<T: Scalar>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(Error::EmptyOperands { op: "vstack" });
    }
    let views: Vec<_> = arrays
        .iter()
        .map(|array| promote_at_least_2d(array))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, 0)
}

/// Stack arrays horizontally (column-wise), like NumPy's `np.hstack`.
///
/// Scalar (0-D) inputs are promoted to 1-D before joining. When all promoted
/// inputs are 1-D, concatenation uses axis 0; for higher rank, axis 1 is
/// used (columns grow horizontally).
///
/// # Arguments
///
/// * `arrays` - Arrays to join column-wise after rank promotion.
///
/// # Returns
///
/// A new C-contiguous [`Array`] widened along axis 0 (1-D) or axis 1
/// (higher rank).
///
/// # Errors
///
/// Same as [`concatenate`](crate::concatenate): shape mismatch or allocation
/// failures surface as [`Error::InvalidArgument`](crate::Error::InvalidArgument).
///
/// # Examples
///
/// ```rust
/// use sdnp::{hstack, Array};
///
/// let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
/// let b = Array::from_slice(&[3_i64, 4], &[2]).unwrap();
/// assert_eq!(hstack(&[&a, &b]).unwrap().to_vec(), vec![1, 2, 3, 4]);
/// ```
pub fn hstack<T: Scalar>(arrays: &[&Array<T>]) -> Result<Array<T>> {
    if arrays.is_empty() {
        return Err(Error::EmptyOperands { op: "hstack" });
    }
    let views: Vec<_> = arrays
        .iter()
        .map(|array| promote_at_least_1d(array))
        .collect::<Result<_>>()?;
    // 1-D inputs concatenate along axis 0; higher rank uses axis 1.
    let axis = if views[0].ndim() == 1 { 0 } else { 1 };
    let refs: Vec<_> = views.iter().collect();
    concatenate(&refs, axis)
}

/// Promote a 0-D array to 1-D via [`insert_axis_view`]; pass through others.
///
/// Used by [`hstack`] so scalars concatenate like NumPy 1-D vectors.
///
/// # Arguments
///
/// * `array` — input array, possibly rank 0
///
/// # Returns
///
/// A view with rank at least 1; unchanged layout when already ≥ 1-D.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — axis insertion or layout validation fails
fn promote_at_least_1d<T: Scalar>(array: &Array<T>) -> Result<Array<T>> {
    if array.ndim() == 0 {
        insert_axis_view(array, 0)
    } else {
        Ok(array.view())
    }
}

/// Promote inputs to rank at least 2 for vertical stacking.
///
/// 0-D scalars become `(1, 1)`; 1-D vectors become `(1, n)`; higher-rank
/// arrays are returned as cheap views.
///
/// # Arguments
///
/// * `array` — input array of any rank
///
/// # Returns
///
/// An array with rank ≥ 2 suitable for [`concatenate`] along axis 0.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — axis insertion or layout validation fails
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
