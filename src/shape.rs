//! Shape and stride helpers (element-unit strides).
//!
//! Empty shape (`[]`) is a valid **0-D** array: [`size_of_shape`] is `1`,
//! and [`c_order_strides`] returns an empty stride list.
//!
//! Axis index / slice resolution lives in [`crate::index`] (`bounds`).

use crate::error::{Error, Result};

/// Number of elements in a shape (product of dimensions).
///
/// The empty product is `1`, so a 0-D shape has size 1.
#[inline]
pub fn size_of_shape(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

/// Checked number of elements in a shape.
///
/// Allocation boundaries use this helper to report shape overflow instead
/// of wrapping a `usize` product.
pub(crate) fn checked_size_of_shape(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |size, &dimension| {
        size.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidArgument("array shape size overflows usize".into())
        })
    })
}

/// C-order (row-major) strides in **element** units.
///
/// # Examples
///
/// ```
/// use sdnp::shape::c_order_strides;
/// assert_eq!(c_order_strides(&[2, 3]), vec![3, 1]);
/// assert_eq!(c_order_strides(&[]), Vec::<isize>::new());
/// ```
pub fn c_order_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    let mut strides = vec![0_isize; ndim];
    if ndim == 0 {
        return strides;
    }
    let mut acc = 1_isize;
    for i in (0..ndim).rev() {
        strides[i] = acc;
        acc *= shape[i] as isize;
    }
    strides
}

/// Whether layout is C-contiguous (row-major, unit stride on the last axis).
///
/// The array may sit at a non-zero buffer offset; callers that scan memory
/// must start at that offset (see ufunc contiguous fast-path). Strides on
/// singleton axes are ignored because their only valid coordinate is zero.
pub(crate) fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1_isize;
    for i in (0..shape.len()).rev() {
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected *= shape[i] as isize;
    }
    true
}

/// Convert multi-dimensional indices to a backing-buffer offset.
#[inline]
pub(crate) fn offset_at(
    indices: &[usize],
    strides: &[isize],
    offset: usize,
) -> usize {
    debug_assert_eq!(indices.len(), strides.len());
    let mut idx = offset as isize;
    for (&i, &s) in indices.iter().zip(strides.iter()) {
        idx += i as isize * s;
    }
    idx as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_order_2d() {
        assert_eq!(c_order_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(size_of_shape(&[2, 3]), 6);
    }

    #[test]
    fn zero_dim_shape() {
        assert_eq!(size_of_shape(&[]), 1);
        assert_eq!(c_order_strides(&[]), Vec::<isize>::new());
        assert!(is_c_contiguous(&[], &[]));
        assert_eq!(offset_at(&[], &[], 0), 0);
    }

    #[test]
    fn c_contiguous_ignores_offset() {
        // Contiguity is about strides; offset is the caller's base index.
        assert!(is_c_contiguous(&[2, 3], &[3, 1]));
        assert!(!is_c_contiguous(&[2, 3], &[1, 3]));
    }

    #[test]
    fn c_contiguous_ignores_singleton_axis_strides() {
        assert!(is_c_contiguous(&[1, 2, 3], &[0, 3, 1]));
        assert!(is_c_contiguous(&[2, 1, 3], &[3, -99, 1]));
        assert!(is_c_contiguous(&[2, 3, 1], &[3, 1, 42]));
    }

    #[test]
    fn c_contiguous_rejects_expanded_zero_stride_axis() {
        assert!(!is_c_contiguous(&[2, 3], &[0, 1]));
    }
}
