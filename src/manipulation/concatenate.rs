//! Join arrays along an existing axis (`np.concatenate`).
//!
//! All inputs must match on every axis except the concatenation axis. Output
//! is a new C-contiguous array built by copying each operand's slabs in order.

use crate::array::Array;
use crate::axis::resolve_axis;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::shape::{checked_allocation_len, checked_size_of_shape};
use crate::traversal::{extend_unary, RunPlan};

/// Join arrays along an existing `axis`.
///
/// Like NumPy's `np.concatenate`: all inputs must share the same rank and
/// agree on every axis length except `axis`, whose output length is the sum
/// of the input lengths along that axis. Negative `axis` indices count from
/// the end (`-1` is the last axis).
///
/// **Axis rules:** `axis` must lie in `[-ndim, ndim)`. All arrays must have
/// identical shapes except at the concatenation axis. The first array's
/// shape (except on `axis`) defines the expected layout for every operand.
///
/// # Arguments
///
/// * `arrays` - Non-empty slice of arrays with matching dtype and compatible
///   shapes.
/// * `axis` - Existing axis along which to join (may be negative).
///
/// # Returns
///
/// A new C-contiguous [`Array`] whose shape matches the inputs except on
/// `axis`.
///
/// # Errors
///
/// * [`Error::AxisOutOfBounds`](crate::Error::AxisOutOfBounds) - `axis` is
///   outside the input rank.
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Concatenated
///   axis length, slab size, or offset overflows; allocation exceeds limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{concatenate, Array};
///
/// let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
/// let b = Array::from_slice(&[3_i64, 4, 5], &[3]).unwrap();
/// assert_eq!(concatenate(&[&a, &b], 0).unwrap().to_vec(), vec![1, 2, 3, 4, 5]);
/// ```
pub fn concatenate<T: Scalar>(
    arrays: &[&Array<T>],
    axis: isize,
) -> Result<Array<T>> {
    let first = arrays
        .first()
        .copied()
        .ok_or(Error::EmptyOperands { op: "concatenate" })?;
    if first.ndim() == 0 {
        return Err(Error::InvalidRank {
            op: "concatenate",
            expected: "arrays of at least one dimension",
            actual: 0,
        });
    }
    let axis = resolve_axis(axis, first.ndim())?;
    for (index, array) in arrays.iter().enumerate().skip(1) {
        if array.ndim() != first.ndim() {
            return Err(Error::RankMismatch {
                op: "concatenate",
                expected: first.ndim(),
                index,
                actual: array.ndim(),
            });
        }
        if array.shape().iter().zip(first.shape()).enumerate().any(
            |(dimension, (actual, expected))| {
                dimension != axis && actual != expected
            },
        ) {
            return Err(Error::ShapeMismatch {
                op: "concatenate",
                expected: first.shape().to_vec(),
                index,
                actual: array.shape().to_vec(),
            });
        }
    }

    let mut output_shape = first.shape().to_vec();
    let mut axis_len = 0usize;
    for array in arrays {
        axis_len =
            axis_len.checked_add(array.shape()[axis]).ok_or_else(|| {
                Error::InvalidArgument(
                    "concatenated axis length overflows usize".into(),
                )
            })?;
    }
    output_shape[axis] = axis_len;

    let capacity = checked_size_of_shape(&output_shape)?;
    checked_allocation_len::<T>(capacity)?;
    let leading_count = checked_size_of_shape(&first.shape()[..axis])?;
    let mut output = Vec::with_capacity(capacity);

    // Outer loop: fixed prefix indices before the concat axis.
    for leading_index in 0..leading_count {
        for array in arrays {
            append_axis_slab(&mut output, array, axis, leading_index)?;
        }
    }
    Array::from_vec(output, &output_shape)
}

/// Copy one slab of `array` along `axis` into `output`.
///
/// A slab is the contiguous memory span for fixed prefix indices before
/// `axis`. Contiguous operands use direct slice extension; strided operands
/// resolve a base offset and walk a [`RunPlan`] over trailing axes.
///
/// # Arguments
///
/// * `output` — destination buffer being extended in C-order
/// * `array` — source array whose slab is appended
/// * `axis` — concatenation axis index
/// * `leading_index` — flat index over axes `[0..axis)`
///
/// # Returns
///
/// `Ok(())` after appending the slab (no-op when slab length is zero).
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — slab or offset arithmetic overflows
fn append_axis_slab<T: Scalar>(
    output: &mut Vec<T>,
    array: &Array<T>,
    axis: usize,
    leading_index: usize,
) -> Result<()> {
    let trailing_len = checked_size_of_shape(&array.shape()[axis + 1..])?;
    let slab_len =
        array.shape()[axis]
            .checked_mul(trailing_len)
            .ok_or_else(|| {
                Error::InvalidArgument("array slab size overflows usize".into())
            })?;
    if slab_len == 0 {
        return Ok(());
    }

    if let Some(slice) = array.as_c_contiguous_slice() {
        let start = leading_index.checked_mul(slab_len).ok_or_else(|| {
            Error::InvalidArgument("array offset overflows usize".into())
        })?;
        let end = start.checked_add(slab_len).ok_or_else(|| {
            Error::InvalidArgument("array offset overflows usize".into())
        })?;
        output.extend_from_slice(&slice[start..end]);
        return Ok(());
    }

    // Map flat leading_index back to a prefix multi-index, then copy the
    // contiguous-in-memory slab along axis..ndim via a run plan.
    let leading_shape = &array.shape()[..axis];
    let leading_strides = &array.strides()[..axis];
    let mut remainder = leading_index;
    let mut base = array.offset() as isize;
    for dimension in (0..axis).rev() {
        let coordinate = remainder % leading_shape[dimension];
        remainder /= leading_shape[dimension];
        let delta = isize::try_from(coordinate)
            .ok()
            .and_then(|coordinate| {
                coordinate.checked_mul(leading_strides[dimension])
            })
            .ok_or_else(|| {
                Error::InvalidArgument("array offset overflows isize".into())
            })?;
        base = base.checked_add(delta).ok_or_else(|| {
            Error::InvalidArgument("array offset overflows isize".into())
        })?;
    }
    let slab_shape = &array.shape()[axis..];
    let slab_strides = &array.strides()[axis..];
    let plan = RunPlan::new(slab_shape, [slab_strides]);
    extend_unary(&plan, &array.data, base as usize, output, |value| value);
    Ok(())
}
