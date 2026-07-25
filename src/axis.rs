//! Shared axis-number normalization.

use crate::error::{Error, Result};

/// Normalize one possibly-negative axis into `0..ndim`.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(Error::AxisOutOfBounds { axis, ndim });
    }
    Ok(normalized as usize)
}

/// Normalize an axis list while preserving order and rejecting duplicates.
pub(crate) fn normalize_axis_list(
    axes: &[isize],
    ndim: usize,
) -> Result<Vec<usize>> {
    let mut normalized = Vec::with_capacity(axes.len());
    let mut seen = vec![false; ndim];
    for &axis in axes {
        let axis = normalize_axis(axis, ndim)?;
        if seen[axis] {
            return Err(Error::InvalidArgument(
                "axes must not contain duplicates".into(),
            ));
        }
        seen[axis] = true;
        normalized.push(axis);
    }
    Ok(normalized)
}

/// Normalize an axis that addresses a rank after inserting one dimension.
pub(crate) fn normalize_insert_axis(axis: isize, ndim: usize) -> Result<usize> {
    let result_ndim = ndim.checked_add(1).ok_or_else(|| {
        Error::InvalidArgument("array rank overflows usize".into())
    })?;
    normalize_axis(axis, result_ndim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_negative_axes() {
        assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
        assert_eq!(normalize_axis_list(&[-1, 0], 3).unwrap(), vec![2, 0]);
    }

    #[test]
    fn rejects_invalid_and_duplicate_axes() {
        assert_eq!(
            normalize_axis(0, 0),
            Err(Error::AxisOutOfBounds { axis: 0, ndim: 0 })
        );
        assert!(normalize_axis_list(&[0, -2], 2).is_err());
    }
}
