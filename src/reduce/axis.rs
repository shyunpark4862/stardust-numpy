//! Axis normalization and reduction geometry helpers.

use crate::error::{Error, Result};
use crate::shape::size_of_shape;

/// Normalize a single axis index into `0..ndim`.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    if ndim == 0 {
        return Err(Error::InvalidArgument(
            "axis is invalid for a 0-D array".into(),
        ));
    }
    let mut ax = axis;
    if ax < 0 {
        ax += ndim as isize;
    }
    if ax < 0 || ax as usize >= ndim {
        return Err(Error::InvalidArgument(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(ax as usize)
}

/// Normalize one or more axes; rejects duplicates. Returns sorted unique axes.
pub(crate) fn normalize_axes(
    axes: &[isize],
    ndim: usize,
) -> Result<Vec<usize>> {
    if axes.is_empty() {
        return Err(Error::InvalidArgument(
            "axes tuple must be non-empty".into(),
        ));
    }
    let mut out = Vec::with_capacity(axes.len());
    for &ax in axes {
        out.push(normalize_axis(ax, ndim)?);
    }
    out.sort_unstable();
    for w in out.windows(2) {
        if w[0] == w[1] {
            return Err(Error::InvalidArgument(
                "duplicate value in 'axis'".into(),
            ));
        }
    }
    Ok(out)
}

/// Axes to reduce: `None` → all axes; otherwise [`normalize_axes`].
pub(crate) fn resolve_reduced_axes(
    ndim: usize,
    axes: Option<&[isize]>,
) -> Result<Vec<usize>> {
    match axes {
        None => Ok((0..ndim).collect()),
        Some(axes) => normalize_axes(axes, ndim),
    }
}

/// Shape after removing `reduced` axes (ascending).
pub(crate) fn outer_shape(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
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
pub(crate) fn outer_axes(ndim: usize, reduced: &[usize]) -> Vec<usize> {
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
        outer_shape(shape, reduced)
    }
}

/// Shared geometry for operations that traverse one axis while preserving
/// independent outer slots.
#[derive(Clone, Debug)]
pub(crate) struct AxisTraversalPlan {
    pub axis: usize,
    pub axis_len: usize,
    pub outer_axes: Vec<usize>,
    pub outer_shape: Vec<usize>,
    pub outer_n: usize,
}

impl AxisTraversalPlan {
    pub fn new(shape: &[usize], axis: usize) -> Self {
        debug_assert!(axis < shape.len());
        let reduced = [axis];
        let axes = outer_axes(shape.len(), &reduced);
        let outer = outer_shape(shape, &reduced);
        Self {
            axis,
            axis_len: shape[axis],
            outer_n: size_of_shape(&outer),
            outer_axes: axes,
            outer_shape: outer,
        }
    }

    /// Strides for the non-traversed axes in outer C-order.
    pub fn outer_strides(&self, strides: &[isize]) -> Vec<isize> {
        self.outer_axes.iter().map(|&axis| strides[axis]).collect()
    }

    /// Whether the traversed axis is the last logical axis.
    #[inline]
    pub fn is_last_axis(&self, ndim: usize) -> bool {
        self.axis + 1 == ndim
    }
}

/// Shared geometry for axis reductions (computed once per call).
#[derive(Clone, Debug)]
pub(crate) struct ReducePlan {
    pub reduced: Vec<usize>,
    pub outer_axes: Vec<usize>,
    pub out_shape: Vec<usize>,
    pub outer_shape: Vec<usize>,
    pub reduced_shape: Vec<usize>,
    /// Number of independent reduction slots (product of outer dims).
    /// `0` means an empty result (no slots to fill).
    pub outer_n: usize,
    /// Elements folded per slot (product of reduced dims).
    /// `0` means the reduced block is empty.
    pub inner_n: usize,
}

impl ReducePlan {
    pub fn new(
        shape: &[usize],
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Self> {
        let reduced = resolve_reduced_axes(shape.len(), axes)?;
        let outer_ax = outer_axes(shape.len(), &reduced);
        let outer = outer_shape(shape, &reduced);
        let red_shape = reduced_shape(shape, &reduced);
        let out = output_shape(shape, &reduced, keepdims);
        Ok(Self {
            reduced,
            outer_axes: outer_ax,
            out_shape: out,
            outer_n: size_of_shape(&outer),
            inner_n: size_of_shape(&red_shape),
            outer_shape: outer,
            reduced_shape: red_shape,
        })
    }

    /// True when the reduced block has no elements (`inner_n == 0`).
    #[inline]
    pub fn inner_is_empty(&self) -> bool {
        self.inner_n == 0
    }

    /// True when reduced axes form a trailing contiguous block
    /// (`[ndim-k, …, ndim-1]`), so each outer slot is one C-order chunk.
    ///
    /// Does **not** check memory layout; callers combine this with
    /// [`Array::as_c_contiguous_slice`](crate::array::Array::as_c_contiguous_slice).
    #[inline]
    pub fn is_suffix_reduction(&self, ndim: usize) -> bool {
        let k = self.reduced.len();
        if k == 0 {
            return true;
        }
        if k > ndim {
            return false;
        }
        let start = ndim - k;
        self.reduced
            .iter()
            .enumerate()
            .all(|(i, &ax)| ax == start + i)
    }

    /// The sole reduced axis, if exactly one axis is reduced.
    #[inline]
    pub fn single_reduced_axis(&self) -> Option<usize> {
        match self.reduced.as_slice() {
            &[axis] => Some(axis),
            _ => None,
        }
    }

    /// Source strides along outer and reduced axes, respectively.
    pub fn outer_reduced_strides(
        &self,
        strides: &[isize],
    ) -> (Vec<isize>, Vec<isize>) {
        let outer = self.outer_axes.iter().map(|&ax| strides[ax]).collect();
        let reduced = self.reduced.iter().map(|&ax| strides[ax]).collect();
        (outer, reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_negative_and_dup() {
        assert_eq!(normalize_axes(&[-1, 0], 3).unwrap(), vec![0, 2]);
        assert!(normalize_axes(&[0, 0], 2).is_err());
        assert!(normalize_axes(&[3], 2).is_err());
    }

    #[test]
    fn plan_shapes() {
        let p = ReducePlan::new(&[2, 3, 4], Some(&[0, 2]), false).unwrap();
        assert_eq!(p.out_shape, vec![3]);
        assert_eq!(p.outer_shape, vec![3]);
        assert_eq!(p.reduced_shape, vec![2, 4]);
        assert_eq!(p.inner_n, 8);
        assert!(!p.inner_is_empty());
    }

    #[test]
    fn single_axis_traversal_geometry() {
        let p = AxisTraversalPlan::new(&[2, 3, 4], 1);
        assert_eq!(p.axis, 1);
        assert_eq!(p.axis_len, 3);
        assert_eq!(p.outer_axes, vec![0, 2]);
        assert_eq!(p.outer_shape, vec![2, 4]);
        assert_eq!(p.outer_n, 8);
        assert_eq!(p.outer_strides(&[12, 4, 1]), vec![12, 1]);
        assert!(!p.is_last_axis(3));
    }

    #[test]
    fn empty_outer_vs_inner() {
        let outer_empty = ReducePlan::new(&[0, 3], Some(&[1]), false).unwrap();
        assert_eq!(outer_empty.outer_n, 0);
        assert_eq!(outer_empty.inner_n, 3);
        assert!(!outer_empty.inner_is_empty());

        let inner_empty = ReducePlan::new(&[0, 3], Some(&[0]), false).unwrap();
        assert_eq!(inner_empty.outer_n, 3);
        assert_eq!(inner_empty.inner_n, 0);
        assert!(inner_empty.inner_is_empty());
    }

    #[test]
    fn suffix_reduction() {
        let s = ReducePlan::new(&[2, 3, 4], Some(&[1, 2]), false).unwrap();
        assert!(s.is_suffix_reduction(3));
        let all = ReducePlan::new(&[2, 3, 4], None, false).unwrap();
        assert!(all.is_suffix_reduction(3));
        let mid = ReducePlan::new(&[2, 3, 4], Some(&[1]), false).unwrap();
        assert!(!mid.is_suffix_reduction(3));
        assert_eq!(mid.single_reduced_axis(), Some(1));
        let skip = ReducePlan::new(&[2, 3, 4], Some(&[0, 2]), false).unwrap();
        assert!(!skip.is_suffix_reduction(3));
        assert_eq!(skip.single_reduced_axis(), None);
    }
}
