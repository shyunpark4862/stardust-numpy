//! Fallible axis resolution shared across reduction, manipulation, linear
//! algebra, and view code.
//!
//! NumPy accepts negative axis indices counting backward from the last
//! dimension. The core owns semantic axis validation: these helpers either
//! return canonical unsigned axes or a structured [`Error`](crate::Error).

use crate::error::{Error, Result};

/// Duplicate-checked set of canonical axes.
///
/// Common ranks fit inline; unusually large ranks use a heap bitset.
pub(crate) enum ResolvedAxisMask {
    Inline(u128),
    Heap(Vec<bool>),
}

impl ResolvedAxisMask {
    #[inline]
    pub(crate) fn contains(&self, axis: usize) -> bool {
        match self {
            Self::Inline(mask) => mask & (1_u128 << axis) != 0,
            Self::Heap(mask) => mask[axis],
        }
    }
}

/// Resolve one possibly-negative axis into `0..ndim`.
pub(crate) fn resolve_axis(axis: isize, ndim: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        ndim.checked_sub(axis.unsigned_abs())
    } else {
        usize::try_from(axis).ok().filter(|&axis| axis < ndim)
    };
    normalized.ok_or(Error::AxisOutOfBounds { axis, ndim })
}

/// Resolve a nonduplicated axis list while preserving its order.
pub(crate) fn resolve_axis_list(
    axes: &[isize],
    ndim: usize,
) -> Result<Vec<usize>> {
    let mut resolved = Vec::with_capacity(axes.len());
    let mut seen = vec![false; ndim];
    for &axis in axes {
        let axis = resolve_axis(axis, ndim)?;
        if seen[axis] {
            return Err(Error::DuplicateAxes);
        }
        seen[axis] = true;
        resolved.push(axis);
    }
    Ok(resolved)
}

/// Resolve a nonduplicated axis list into a membership mask.
///
/// Ranks up to 128 require no heap allocation. The result intentionally does
/// not preserve input order and is intended for operations that only need axis
/// membership.
pub(crate) fn resolve_axis_mask(
    axes: &[isize],
    ndim: usize,
) -> Result<ResolvedAxisMask> {
    if ndim <= u128::BITS as usize {
        let mut mask = 0_u128;
        for &axis in axes {
            let axis = resolve_axis(axis, ndim)?;
            let bit = 1_u128 << axis;
            if mask & bit != 0 {
                return Err(Error::DuplicateAxes);
            }
            mask |= bit;
        }
        Ok(ResolvedAxisMask::Inline(mask))
    } else {
        let mut mask = vec![false; ndim];
        for &axis in axes {
            let axis = resolve_axis(axis, ndim)?;
            if mask[axis] {
                return Err(Error::DuplicateAxes);
            }
            mask[axis] = true;
        }
        Ok(ResolvedAxisMask::Heap(mask))
    }
}

/// Resolve an insertion position in `0..=ndim`.
pub(crate) fn resolve_insert_axis(axis: isize, ndim: usize) -> Result<usize> {
    let output_ndim = ndim
        .checked_add(1)
        .ok_or(Error::AxisOutOfBounds { axis, ndim })?;
    resolve_axis(axis, output_ndim)
}

/// Resolve two distinct axes used to define a diagonal plane.
pub(crate) fn resolve_diagonal_axes(
    axis1: isize,
    axis2: isize,
    ndim: usize,
) -> Result<(usize, usize)> {
    let axis1 = resolve_axis(axis1, ndim)?;
    let axis2 = resolve_axis(axis2, ndim)?;
    if axis1 == axis2 {
        return Err(Error::AxesMustDiffer);
    }
    Ok((axis1, axis2))
}

/// Resolve and validate a complete permutation of `0..ndim`, visiting each
/// canonical axis exactly once in input order.
///
/// The visitor lets callers consume resolved axes directly without first
/// allocating a temporary `Vec`. Ranks up to 128 use an inline bitset for
/// duplicate detection; unusually large ranks fall back to heap storage.
pub(crate) fn visit_resolved_permutation(
    axes: &[isize],
    ndim: usize,
    mut visit: impl FnMut(usize),
) -> Result<()> {
    if axes.len() != ndim {
        return Err(Error::NotPermutation);
    }

    if ndim <= u128::BITS as usize {
        let mut seen = 0_u128;
        for &axis in axes {
            let axis = resolve_axis(axis, ndim)?;
            let bit = 1_u128 << axis;
            if seen & bit != 0 {
                return Err(Error::NotPermutation);
            }
            seen |= bit;
            visit(axis);
        }
    } else {
        let mut seen = vec![false; ndim];
        for &axis in axes {
            let axis = resolve_axis(axis, ndim)?;
            if seen[axis] {
                return Err(Error::NotPermutation);
            }
            seen[axis] = true;
            visit(axis);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_negative_axis_and_rejects_bounds() {
        assert_eq!(resolve_axis(-1, 3), Ok(2));
        assert_eq!(
            resolve_axis(3, 3),
            Err(Error::AxisOutOfBounds { axis: 3, ndim: 3 })
        );
        assert_eq!(
            resolve_axis(-4, 3),
            Err(Error::AxisOutOfBounds { axis: -4, ndim: 3 })
        );
    }

    #[test]
    fn rejects_duplicate_axis_lists_after_resolution() {
        assert_eq!(resolve_axis_list(&[0, -2], 2), Err(Error::DuplicateAxes));
        assert!(matches!(
            resolve_axis_mask(&[0, -2], 2),
            Err(Error::DuplicateAxes)
        ));
        let mask = resolve_axis_mask(&[-1, 0], 3).unwrap();
        assert!(mask.contains(0));
        assert!(mask.contains(2));
        assert!(!mask.contains(1));
    }

    #[test]
    fn validates_insert_diagonal_and_permutation_axes() {
        assert_eq!(resolve_insert_axis(-1, 2), Ok(2));
        assert!(matches!(
            resolve_insert_axis(3, 2),
            Err(Error::AxisOutOfBounds { .. })
        ));
        assert_eq!(resolve_diagonal_axes(0, -2, 2), Err(Error::AxesMustDiffer));
        assert_eq!(
            visit_resolved_permutation(&[0, 0], 2, |_| {}),
            Err(Error::NotPermutation)
        );
        assert_eq!(
            visit_resolved_permutation(&[0], 2, |_| {}),
            Err(Error::NotPermutation)
        );

        let mut resolved = Vec::new();
        visit_resolved_permutation(&[-1, 0], 2, |axis| resolved.push(axis))
            .unwrap();
        assert_eq!(resolved, [1, 0]);

        let large: Vec<isize> = (0..129).collect();
        let mut visited = 0;
        visit_resolved_permutation(&large, large.len(), |_| visited += 1)
            .unwrap();
        assert_eq!(visited, large.len());
    }
}
