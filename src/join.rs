//! Array joining and dimension-insertion operations.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::stride_iter::StrideIter;

fn insert_axis_view<T: Scalar>(a: &Array<T>, axis: isize) -> Result<Array<T>> {
    let axis = normalize_insert_axis(axis, a.ndim())?;
    let mut shape = a.shape().to_vec();
    let mut strides = a.strides().to_vec();
    shape.insert(axis, 1);
    strides.insert(axis, 0);
    Array::from_arc_raw_parts(
        Arc::clone(&a.data),
        shape,
        strides,
        a.offset(),
        a.is_writable(),
    )
}

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

    let mut out_shape = first.shape().to_vec();
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
    out_shape[axis] = axis_len;

    let capacity = checked_shape_size(&out_shape)?;
    let outer_len = checked_shape_size(&first.shape()[..axis])?;
    let mut output = Vec::with_capacity(capacity);

    for outer_index in 0..outer_len {
        for array in arrays {
            append_axis_slab(&mut output, array, axis, outer_index)?;
        }
    }
    debug_assert_eq!(output.len(), capacity);
    Array::from_vec(output, &out_shape)
}

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

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(Error::InvalidArgument(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
}

fn normalize_insert_axis(axis: isize, ndim: usize) -> Result<usize> {
    let result_ndim = ndim.checked_add(1).ok_or_else(|| {
        Error::InvalidArgument("array rank overflows usize".into())
    })?;
    normalize_axis(axis, result_ndim)
}

fn checked_shape_size(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |size, &dimension| {
        size.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidArgument("array shape size overflows usize".into())
        })
    })
}

fn append_axis_slab<T: Scalar>(
    output: &mut Vec<T>,
    array: &Array<T>,
    axis: usize,
    outer_index: usize,
) -> Result<()> {
    let inner_len = checked_shape_size(&array.shape()[axis + 1..])?;
    let slab_len =
        array.shape()[axis].checked_mul(inner_len).ok_or_else(|| {
            Error::InvalidArgument("array slab size overflows usize".into())
        })?;
    if slab_len == 0 {
        return Ok(());
    }

    if let Some(slice) = array.as_c_contiguous_slice() {
        let start = outer_index.checked_mul(slab_len).ok_or_else(|| {
            Error::InvalidArgument("array offset overflows usize".into())
        })?;
        output.extend_from_slice(&slice[start..start + slab_len]);
        return Ok(());
    }

    let prefix_shape = &array.shape()[..axis];
    let prefix_strides = &array.strides()[..axis];
    let mut remainder = outer_index;
    let mut base = array.offset() as isize;
    for dimension in (0..axis).rev() {
        let coordinate = remainder % prefix_shape[dimension];
        remainder /= prefix_shape[dimension];
        base += coordinate as isize * prefix_strides[dimension];
    }
    debug_assert!(base >= 0);
    let slab_shape = &array.shape()[axis..];
    let slab_strides = &array.strides()[axis..];
    output.extend(
        StrideIter::new(slab_shape, slab_strides, base as usize)
            .map(|physical| array.data[physical]),
    );
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenate_noncontiguous_on_middle_axis() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let transposed = a.transpose();
        let joined = concatenate(&[&transposed, &transposed], 1).unwrap();
        assert_eq!(joined.shape(), &[3, 4]);
        assert_eq!(joined.to_vec(), vec![1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6]);
        assert!(joined.is_c_contiguous());
    }

    #[test]
    fn stack_supports_scalars_and_negative_axis() {
        let a = Array::from_slice(&[1_i64], &[]).unwrap();
        let b = Array::from_slice(&[2_i64], &[]).unwrap();
        let joined = stack(&[&a, &b], -1).unwrap();
        assert_eq!(joined.shape(), &[2]);
        assert_eq!(joined.to_vec(), vec![1, 2]);
    }

    #[test]
    fn vertical_and_horizontal_promotion() {
        let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
        let b = Array::from_slice(&[3_i64, 4], &[2]).unwrap();
        let vertical = vstack(&[&a, &b]).unwrap();
        assert_eq!(vertical.shape(), &[2, 2]);
        assert_eq!(vertical.to_vec(), vec![1, 2, 3, 4]);

        let horizontal = hstack(&[&a, &b]).unwrap();
        assert_eq!(horizontal.shape(), &[4]);
        assert_eq!(horizontal.to_vec(), vec![1, 2, 3, 4]);
    }
}
