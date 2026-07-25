//! Axis-number normalization shared across reduction, manipulation, and
//! view code.
//!
//! NumPy accepts negative axis indices counting backward from the last
//! dimension. Callers are expected to validate axis lists before calling
//! these helpers; invalid axes map to `usize::MAX` as a sentinel value.

/// Convert one possibly-negative axis index into the range `0..ndim`.
///
/// Negative values count backward from `ndim` (e.g. `-1` is the last axis).
/// Out-of-range inputs return `usize::MAX` so callers can detect failure
/// without panicking.
///
/// # Arguments
///
/// * `axis` — raw axis index (may be negative)
/// * `ndim` — number of dimensions in the array being addressed
///
/// # Returns
///
/// Normalized axis in `0..ndim`, or `usize::MAX` when out of range.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> usize {
    let normalized = if axis < 0 {
        // Negative index: count backward from ndim.
        ndim.checked_sub(axis.unsigned_abs())
    } else {
        // Non-negative index must fit in usize and be strictly less than ndim.
        usize::try_from(axis).ok().filter(|&axis| axis < ndim)
    };
    normalized.unwrap_or(usize::MAX)
}

/// Normalize every axis in `axes`, preserving the original order.
///
/// Applies [`normalize_axis`] element-wise. Duplicate or invalid entries are
/// preserved as `usize::MAX` for the caller to reject.
///
/// # Arguments
///
/// * `axes` — raw axis indices (may be negative)
/// * `ndim` — number of dimensions in the array being addressed
///
/// # Returns
///
/// Normalized axis indices in the same order as `axes`.
pub(crate) fn normalize_axis_list(axes: &[isize], ndim: usize) -> Vec<usize> {
    axes.iter()
        .map(|&axis| normalize_axis(axis, ndim))
        .collect()
}

/// Normalize an axis index for a rank that will gain one inserted dimension.
///
/// Delegates to [`normalize_axis`] with `ndim + 1`, matching NumPy rules for
/// `expand_dims` / `newaxis` insertion bounds.
///
/// # Arguments
///
/// * `axis` — insertion position (may be negative)
/// * `ndim` — current rank before the new axis is inserted
///
/// # Returns
///
/// Normalized insertion axis in `0..=ndim`, or `usize::MAX` when invalid.
pub(crate) fn normalize_insert_axis(axis: isize, ndim: usize) -> usize {
    normalize_axis(axis, ndim + 1)
}
