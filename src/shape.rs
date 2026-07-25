//! Shape products, C-order stride computation, and layout validation.
//!
//! Strides are stored in **element** units (not bytes), matching NumPy's
//! logical addressing model. An empty shape `[]` denotes a valid 0-D array:
//! its size is 1 and its stride list is empty. Axis index and slice
//! resolution lives in [`crate::index`].

use crate::error::{Error, Result};

/// Compute the number of logical elements described by `shape`.
///
/// Returns `Ok(0)` when any dimension is zero. The empty product is `1`,
/// so a 0-D shape has size 1.
///
/// # Arguments
///
/// * `shape` — axis lengths to multiply
///
/// # Returns
///
/// The element count, or `0` if any axis length is zero.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — the product overflows `usize`
#[inline]
pub fn size_of_shape(shape: &[usize]) -> Result<usize> {
    checked_size_of_shape(shape)
}

/// Checked element count; used internally at allocation boundaries.
///
/// Short-circuits to zero when any dimension is zero before multiplying.
///
/// # Arguments
///
/// * `shape` — axis lengths to multiply
///
/// # Returns
///
/// The element count, or `0` if any axis length is zero.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — the product overflows `usize`
pub(crate) fn checked_size_of_shape(shape: &[usize]) -> Result<usize> {
    // Any zero dimension collapses the whole array to length zero.
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |size, &dimension| {
        size.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidArgument("array shape size overflows usize".into())
        })
    })
}

/// Reject allocations whose byte size would overflow `usize` or `isize`.
///
/// NumPy-compatible pointer arithmetic assumes buffer byte lengths fit in
/// `isize` so signed offsets remain representable.
///
/// # Arguments
///
/// * `len` — number of elements to allocate
///
/// # Returns
///
/// `Ok(())` when `len * size_of::<T>()` fits in both `usize` and `isize`.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — byte size overflows `usize` or `isize`
pub(crate) fn checked_allocation_len<T>(len: usize) -> Result<()> {
    let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        Error::InvalidArgument("array allocation size overflows usize".into())
    })?;
    // NumPy-compatible pointer arithmetic assumes offsets fit in isize.
    if bytes > isize::MAX as usize {
        return Err(Error::InvalidArgument(
            "array allocation exceeds isize address range".into(),
        ));
    }
    Ok(())
}

/// Validate that every non-zero dimension and their product fit in `isize`.
///
/// Rejects shapes whose nonzero axis lengths or running products exceed
/// addressable range. Zero-length axes are skipped (they do not contribute
/// to the product).
///
/// # Arguments
///
/// * `shape` — array shape to validate
///
/// # Returns
///
/// `Ok(())` when geometry is safe for signed offset arithmetic.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — a dimension or product exceeds `isize`
pub(crate) fn validate_shape_geometry(shape: &[usize]) -> Result<()> {
    let mut nonzero_product = 1usize;
    for &dimension in shape {
        if dimension == 0 {
            continue;
        }
        if dimension > isize::MAX as usize {
            return Err(Error::InvalidArgument(
                "array dimension exceeds isize address range".into(),
            ));
        }
        nonzero_product =
            nonzero_product.checked_mul(dimension).ok_or_else(|| {
                Error::InvalidArgument(
                    "array shape geometry overflows usize".into(),
                )
            })?;
        if nonzero_product > isize::MAX as usize {
            return Err(Error::InvalidArgument(
                "array shape geometry exceeds isize address range".into(),
            ));
        }
    }
    Ok(())
}

/// Unchecked element count for shapes already validated at a boundary.
///
/// # Arguments
///
/// * `shape` — validated axis lengths
///
/// # Returns
///
/// Product of all axis lengths (0 when any axis length is 0).
#[inline]
pub(crate) fn size_of_shape_unchecked(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

/// Compute C-order (row-major) strides in element units.
///
/// The last axis has stride 1; each earlier axis stride equals the product
/// of all later dimension lengths. A 0-D shape yields an empty stride list.
///
/// # Arguments
///
/// * `shape` — axis lengths for which to compute strides
///
/// # Returns
///
/// A vector of strides with the same length as `shape`.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — a dimension or stride product exceeds
///   `isize`
///
/// # Examples
///
/// ```
/// use sdnp::shape::c_order_strides;
///
/// assert_eq!(c_order_strides(&[2, 3]).unwrap(), vec![3, 1]);
/// assert_eq!(c_order_strides(&[]).unwrap(), Vec::<isize>::new());
/// ```
pub fn c_order_strides(shape: &[usize]) -> Result<Vec<isize>> {
    checked_c_order_strides(shape)
}

/// Checked C-order stride builder used at array construction time.
///
/// Walks axes from last to first, accumulating running stride products with
/// overflow checks at each step.
///
/// # Arguments
///
/// * `shape` — axis lengths for which to compute strides
///
/// # Returns
///
/// C-order strides in element units.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — dimension or stride product exceeds
///   `isize`
pub(crate) fn checked_c_order_strides(shape: &[usize]) -> Result<Vec<isize>> {
    let ndim = shape.len();
    let mut strides = vec![0_isize; ndim];
    if ndim == 0 {
        return Ok(strides);
    }
    // Walk axes from last to first, accumulating running stride products.
    let mut acc = 1_isize;
    for i in (0..ndim).rev() {
        strides[i] = acc;
        let dimension = isize::try_from(shape[i]).map_err(|_| {
            Error::InvalidArgument(
                "array dimension exceeds isize address range".into(),
            )
        })?;
        acc = acc.checked_mul(dimension).ok_or_else(|| {
            Error::InvalidArgument("C-order stride overflows isize".into())
        })?;
    }
    Ok(strides)
}

/// Stride helper for shapes validated earlier in the same call chain.
///
/// # Arguments
///
/// * `shape` — shape already validated by the caller
///
/// # Returns
///
/// C-order strides; panics only if validation was skipped incorrectly.
///
/// # Panics
///
/// Panics when stride computation fails (caller must validate first).
#[inline]
pub(crate) fn c_order_strides_unchecked(shape: &[usize]) -> Vec<isize> {
    checked_c_order_strides(shape).expect("validated array shape")
}

/// Return whether `(shape, strides)` describe a C-contiguous layout.
///
/// The logical origin may sit at a non-zero buffer offset; ufunc fast-paths
/// must honor that offset. Strides on length-one axes are ignored because
/// their only valid coordinate is zero.
///
/// # Arguments
///
/// * `shape` — axis lengths
/// * `strides` — element strides aligned with `shape`
///
/// # Returns
///
/// `true` when memory order matches C-order strides for `shape`.
pub(crate) fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1_isize;
    for i in (0..shape.len()).rev() {
        // Singleton axes do not constrain memory layout.
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        let Ok(dimension) = isize::try_from(shape[i]) else {
            return false;
        };
        let Some(next) = expected.checked_mul(dimension) else {
            return false;
        };
        expected = next;
    }
    true
}

/// Map multi-dimensional indices to a flat offset in the backing buffer.
///
/// Computes `offset + Σ indices[k] * strides[k]` using signed intermediate
/// arithmetic, then casts back to `usize`.
///
/// # Arguments
///
/// * `indices` — coordinate along each axis
/// * `strides` — element strides aligned with `indices`
/// * `offset` — base buffer index of the array view
///
/// # Returns
///
/// Linear buffer index for the coordinate.
#[inline]
pub(crate) fn offset_at(
    indices: &[usize],
    strides: &[isize],
    offset: usize,
) -> usize {
    let mut idx = offset as isize;
    for (&i, &s) in indices.iter().zip(strides.iter()) {
        idx += i as isize * s;
    }
    idx as usize
}
