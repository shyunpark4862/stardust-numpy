use crate::array::Array;
use crate::axis::normalize_axis;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::shape::checked_size_of_shape;
use crate::traversal::{extend_unary, RunPlan};

/// Join arrays along an existing `axis`.
///
/// All arrays must have the same rank and equal dimensions except along
/// `axis`. Negative axes count backward from the rank. The result is a newly
/// allocated C-contiguous array.
pub fn concatenate<T: Scalar>(
    arrays: &[&Array<T>],
    axis: isize,
) -> Result<Array<T>> {
    let first = require_arrays(arrays, "concatenate")?;
    if first.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "cannot concatenate 0-D arrays".into(),
        ));
    }
    let axis = normalize_axis(axis, first.ndim())?;

    let mut output_shape = first.shape().to_vec();
    let mut axis_len = 0usize;
    for (index, array) in arrays.iter().enumerate() {
        if array.ndim() != first.ndim() {
            return Err(Error::InvalidArgument(format!(
                "all arrays must have the same rank; array 0 has rank {}, \
                 array {index} has rank {}",
                first.ndim(),
                array.ndim()
            )));
        }
        for dimension in 0..first.ndim() {
            if dimension != axis
                && array.shape()[dimension] != first.shape()[dimension]
            {
                return Err(Error::InvalidArgument(format!(
                    "array dimensions must match except along axis {axis}; \
                     array 0 has shape {:?}, array {index} has shape {:?}",
                    first.shape(),
                    array.shape()
                )));
            }
        }
        axis_len =
            axis_len.checked_add(array.shape()[axis]).ok_or_else(|| {
                Error::InvalidArgument(
                    "concatenated axis length overflows usize".into(),
                )
            })?;
    }
    output_shape[axis] = axis_len;

    let capacity = checked_size_of_shape(&output_shape)?;
    let leading_count = checked_size_of_shape(&first.shape()[..axis])?;
    let mut output = Vec::with_capacity(capacity);

    for leading_index in 0..leading_count {
        for array in arrays {
            append_axis_slab(&mut output, array, axis, leading_index)?;
        }
    }
    debug_assert_eq!(output.len(), capacity);
    Array::from_vec(output, &output_shape)
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
        output.extend_from_slice(&slice[start..start + slab_len]);
        return Ok(());
    }

    let leading_shape = &array.shape()[..axis];
    let leading_strides = &array.strides()[..axis];
    let mut remainder = leading_index;
    let mut base = array.offset() as isize;
    for dimension in (0..axis).rev() {
        let coordinate = remainder % leading_shape[dimension];
        remainder /= leading_shape[dimension];
        base += coordinate as isize * leading_strides[dimension];
    }
    debug_assert!(base >= 0);
    let slab_shape = &array.shape()[axis..];
    let slab_strides = &array.strides()[axis..];
    let plan = RunPlan::new(slab_shape, [slab_strides]);
    extend_unary(&plan, &array.data, base as usize, output, |value| value);
    Ok(())
}
