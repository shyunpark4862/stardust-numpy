//! Reduction-axis geometry helpers.

use crate::axis::normalize_axis_list;
use crate::error::Result;
use crate::shape::size_of_shape;

/// Normalize one or more axes. Returns sorted unique axes.
fn normalize_reduction_axes(axes: &[isize], ndim: usize) -> Vec<usize> {
    let mut out = normalize_axis_list(axes, ndim);
    out.sort_unstable();
    out.dedup();
    out
}

/// Axes to reduce: `None` → all axes; otherwise normalized and sorted.
pub(crate) fn resolve_reduced_axes(
    ndim: usize,
    axes: Option<&[isize]>,
) -> Result<Vec<usize>> {
    Ok(match axes {
        None => (0..ndim).collect(),
        Some([]) => vec![],
        Some(axes) => normalize_reduction_axes(axes, ndim),
    })
}

/// Shape after removing `reduced` axes (ascending).
pub(crate) fn kept_shape(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(shape.len().saturating_sub(reduced.len()));
    let mut r = 0usize;
    for (axis, &dim) in shape.iter().enumerate() {
        if r < reduced.len() && reduced[r] == axis {
            r += 1;
        } else {
            out.push(dim);
        }
    }
    out
}

/// Complementary axis list (ascending) for non-reduced axes.
pub(crate) fn kept_axes(ndim: usize, reduced: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(ndim.saturating_sub(reduced.len()));
    let mut r = 0usize;
    for axis in 0..ndim {
        if r < reduced.len() && reduced[r] == axis {
            r += 1;
        } else {
            out.push(axis);
        }
    }
    out
}

/// Shape of the reduced block.
pub(crate) fn reduced_shape(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
    reduced.iter().map(|&ax| shape[ax]).collect()
}

/// `keepdims=True` output shape.
pub(crate) fn keepdims_shape(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
    let mut out = shape.to_vec();
    for &ax in reduced {
        out[ax] = 1;
    }
    out
}

/// Output shape for a reduction (`keepdims` aware).
pub(crate) fn output_shape(
    shape: &[usize],
    reduced: &[usize],
    keepdims: bool,
) -> Vec<usize> {
    if keepdims {
        keepdims_shape(shape, reduced)
    } else {
        kept_shape(shape, reduced)
    }
}

/// Shared geometry for operations that traverse one axis while preserving
/// independent output slots.
#[derive(Clone, Debug)]
pub(crate) struct AxisTraversalPlan {
    pub(crate) axis: usize,
    pub(crate) axis_len: usize,
    pub(crate) kept_axes: Vec<usize>,
    pub(crate) kept_shape: Vec<usize>,
    pub(crate) output_len: usize,
}

impl AxisTraversalPlan {
    pub(crate) fn new(shape: &[usize], axis: usize) -> Self {
        debug_assert!(axis < shape.len());
        let axis_len = shape[axis];
        let reduced_axes = [axis];
        let axes = kept_axes(shape.len(), &reduced_axes);
        let kept_shape = kept_shape(shape, &reduced_axes);
        Self {
            axis,
            axis_len,
            output_len: size_of_shape(&kept_shape),
            kept_axes: axes,
            kept_shape,
        }
    }

    /// Strides for the non-traversed axes in outer C-order.
    pub(crate) fn kept_strides(&self, strides: &[isize]) -> Vec<isize> {
        self.kept_axes.iter().map(|&axis| strides[axis]).collect()
    }

    /// Whether the traversed axis is the last logical axis.
    #[inline]
    pub(crate) fn is_last_axis(&self, ndim: usize) -> bool {
        self.axis + 1 == ndim
    }
}

/// Shared geometry for axis reductions (computed once per call).
#[derive(Clone, Debug)]
pub(crate) struct ReducePlan {
    pub(crate) reduced_axes: Vec<usize>,
    pub(crate) kept_axes: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) kept_shape: Vec<usize>,
    pub(crate) reduced_shape: Vec<usize>,
    /// Number of independent reduction outputs (product of kept dimensions).
    /// `0` means an empty result (no slots to fill).
    pub(crate) output_len: usize,
    /// Elements folded per slot (product of reduced dims).
    /// `0` means the reduced block is empty.
    pub(crate) reduction_len: usize,
}

/// Physical traversal selected from reduction geometry and input layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraversalSchedule {
    /// Every output slot owns one contiguous suffix chunk.
    SuffixContiguous,
    /// Reduced prefix rows are scanned in memory order while all output slots
    /// are accumulated together.
    PrefixContiguous {
        reduced_len: usize,
        output_len: usize,
    },
    /// Arbitrary layout handled by `RunPlan` and reduced-axis cursors.
    GeneralStrided,
}

impl ReducePlan {
    pub(crate) fn new(
        shape: &[usize],
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Self> {
        let reduced_axes = resolve_reduced_axes(shape.len(), axes)?;
        let kept_axes = kept_axes(shape.len(), &reduced_axes);
        let kept_shape = kept_shape(shape, &reduced_axes);
        let reduced_shape = reduced_shape(shape, &reduced_axes);
        let output_shape = output_shape(shape, &reduced_axes, keepdims);
        Ok(Self {
            reduced_axes,
            kept_axes,
            output_shape,
            output_len: size_of_shape(&kept_shape),
            reduction_len: size_of_shape(&reduced_shape),
            kept_shape,
            reduced_shape,
        })
    }

    /// True when the reduced block has no elements (`reduction_len == 0`).
    #[inline]
    pub(crate) fn reduction_is_empty(&self) -> bool {
        self.reduction_len == 0
    }

    /// True when reduced axes form a trailing contiguous block
    /// (`[ndim-k, …, ndim-1]`), so each outer slot is one C-order chunk.
    ///
    /// Does **not** check memory layout; callers combine this with
    /// [`Array::as_c_contiguous_slice`](crate::array::Array::as_c_contiguous_slice).
    #[inline]
    pub(crate) fn is_suffix_reduction(&self, ndim: usize) -> bool {
        let k = self.reduced_axes.len();
        if k == 0 {
            return true;
        }
        if k > ndim {
            return false;
        }
        let start = ndim - k;
        self.reduced_axes
            .iter()
            .enumerate()
            .all(|(i, &ax)| ax == start + i)
    }

    /// True when reduced axes form a leading block (`[0, …, k-1]`).
    #[inline]
    pub(crate) fn is_prefix_reduction(&self) -> bool {
        self.reduced_axes
            .iter()
            .enumerate()
            .all(|(axis, &reduced)| axis == reduced)
    }

    /// Choose a physical traversal without embedding operation semantics.
    #[inline]
    pub(crate) fn traversal_schedule(
        &self,
        ndim: usize,
        is_c_contiguous: bool,
    ) -> TraversalSchedule {
        if !is_c_contiguous {
            return TraversalSchedule::GeneralStrided;
        }
        if self.is_suffix_reduction(ndim) {
            return TraversalSchedule::SuffixContiguous;
        }
        if self.is_prefix_reduction() {
            return TraversalSchedule::PrefixContiguous {
                reduced_len: self.reduction_len,
                output_len: self.output_len,
            };
        }
        TraversalSchedule::GeneralStrided
    }

    /// Source strides along kept and reduced axes, respectively.
    pub(crate) fn kept_reduced_strides(
        &self,
        strides: &[isize],
    ) -> (Vec<isize>, Vec<isize>) {
        let kept = self.kept_axes.iter().map(|&ax| strides[ax]).collect();
        let reduced = self.reduced_axes.iter().map(|&ax| strides[ax]).collect();
        (kept, reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_negative_axes() {
        assert_eq!(normalize_reduction_axes(&[-1, 0], 3), vec![0, 2]);
    }

    #[test]
    fn plan_shapes() {
        let p = ReducePlan::new(&[2, 3, 4], Some(&[0, 2]), false).unwrap();
        assert_eq!(p.output_shape, vec![3]);
        assert_eq!(p.kept_shape, vec![3]);
        assert_eq!(p.reduced_shape, vec![2, 4]);
        assert_eq!(p.reduction_len, 8);
        assert!(!p.reduction_is_empty());
    }

    #[test]
    fn single_axis_traversal_geometry() {
        let p = AxisTraversalPlan::new(&[2, 3, 4], 1);
        assert_eq!(p.axis, 1);
        assert_eq!(p.axis_len, 3);
        assert_eq!(p.kept_axes, vec![0, 2]);
        assert_eq!(p.kept_shape, vec![2, 4]);
        assert_eq!(p.output_len, 8);
        assert_eq!(p.kept_strides(&[12, 4, 1]), vec![12, 1]);
        assert!(!p.is_last_axis(3));
    }

    #[test]
    fn empty_outer_vs_inner() {
        let outer_empty = ReducePlan::new(&[0, 3], Some(&[1]), false).unwrap();
        assert_eq!(outer_empty.output_len, 0);
        assert_eq!(outer_empty.reduction_len, 3);
        assert!(!outer_empty.reduction_is_empty());

        let inner_empty = ReducePlan::new(&[0, 3], Some(&[0]), false).unwrap();
        assert_eq!(inner_empty.output_len, 3);
        assert_eq!(inner_empty.reduction_len, 0);
        assert!(inner_empty.reduction_is_empty());
    }

    #[test]
    fn suffix_reduction() {
        let s = ReducePlan::new(&[2, 3, 4], Some(&[1, 2]), false).unwrap();
        assert!(s.is_suffix_reduction(3));
        let all = ReducePlan::new(&[2, 3, 4], None, false).unwrap();
        assert!(all.is_suffix_reduction(3));
        let mid = ReducePlan::new(&[2, 3, 4], Some(&[1]), false).unwrap();
        assert!(!mid.is_suffix_reduction(3));
        let skip = ReducePlan::new(&[2, 3, 4], Some(&[0, 2]), false).unwrap();
        assert!(!skip.is_suffix_reduction(3));
    }

    #[test]
    fn traversal_schedule_classifies_prefix_suffix_and_general() {
        let prefix = ReducePlan::new(&[2, 3, 4], Some(&[0, 1]), false).unwrap();
        assert_eq!(
            prefix.traversal_schedule(3, true),
            TraversalSchedule::PrefixContiguous {
                reduced_len: 6,
                output_len: 4,
            }
        );

        let suffix = ReducePlan::new(&[2, 3, 4], Some(&[1, 2]), false).unwrap();
        assert_eq!(
            suffix.traversal_schedule(3, true),
            TraversalSchedule::SuffixContiguous
        );

        let middle = ReducePlan::new(&[2, 3, 4], Some(&[1]), false).unwrap();
        assert_eq!(
            middle.traversal_schedule(3, true),
            TraversalSchedule::GeneralStrided
        );
        assert_eq!(
            prefix.traversal_schedule(3, false),
            TraversalSchedule::GeneralStrided
        );
    }
}
