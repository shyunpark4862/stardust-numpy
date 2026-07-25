//! Shared axis-number normalization (caller must supply valid axes).

/// Normalize one possibly-negative axis into `0..ndim`.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> usize {
    debug_assert!(ndim > 0, "normalize_axis requires ndim > 0");
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };
    debug_assert!(
        normalized >= 0 && (normalized as usize) < ndim,
        "axis {axis} out of bounds for ndim {ndim}"
    );
    normalized as usize
}

/// Normalize an axis list while preserving order.
pub(crate) fn normalize_axis_list(axes: &[isize], ndim: usize) -> Vec<usize> {
    axes.iter()
        .map(|&axis| normalize_axis(axis, ndim))
        .collect()
}

/// Normalize an axis that addresses a rank after inserting one dimension.
pub(crate) fn normalize_insert_axis(axis: isize, ndim: usize) -> usize {
    normalize_axis(axis, ndim + 1)
}
