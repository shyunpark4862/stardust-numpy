//! Reduction-axis geometry and traversal scheduling.
//!
//! A [`ReducePlan`] records which axes collapse, output shape, and how many
//! elements each output slot folds. [`TraversalSchedule`] picks contiguous
//! suffix chunks, prefix rows, or a general strided walk once per call.

use crate::axis::resolve_axis_list;
use crate::error::Result;
use crate::shape::{checked_size_of_shape, size_of_shape_unchecked};

/// Resolve and sort reduction axis indices.
///
/// Negative axis indices are resolved against `ndim` before sorting so
/// kernels always see ascending, unique axis numbers.
///
/// # Arguments
///
/// * `axes` - Raw axis list (may contain negative indices).
/// * `ndim` - Rank of the array being reduced.
///
/// # Returns
///
/// Sorted, unique axis indices in `[0, ndim)`.
fn resolve_reduction_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>> {
    let mut out = resolve_axis_list(axes, ndim)?;
    out.sort_unstable();
    Ok(out)
}

/// Resolve which axes collapse for a reduction call.
///
/// `None` reduces every axis (global reduction). `Some(&[])` reduces none
/// and yields an output with the same rank as the input.
///
/// # Arguments
///
/// * `ndim` - Rank of the input array.
/// * `axes` - Explicit axis list, empty slice, or `None` for all axes.
///
/// # Returns
///
/// Ascending, deduplicated axis indices to fold.
///
/// # Errors
///
/// Returns an error when an axis index is out of range or duplicated after
/// normalization.
pub(crate) fn resolve_reduced_axes(
    ndim: usize,
    axes: Option<&[isize]>,
) -> Result<Vec<usize>> {
    Ok(match axes {
        None => (0..ndim).collect(),
        Some([]) => vec![],
        Some(axes) => resolve_reduction_axes(axes, ndim)?,
    })
}

/// Shape after dropping reduced axes (ascending axis order).
///
/// Used when `keepdims = false`. Reduced dimensions are removed rather than
/// replaced with length-one slots.
///
/// # Arguments
///
/// * `shape` - Input array shape.
/// * `reduced` - Sorted axis indices being collapsed.
///
/// # Returns
///
/// Shape containing only non-reduced dimensions, in original axis order.
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

/// Non-reduced axis indices in ascending order.
///
/// Complements [`kept_shape`]: same filtering rule, but returns axis numbers
/// instead of dimension lengths.
///
/// # Arguments
///
/// * `ndim` - Rank of the input array.
/// * `reduced` - Sorted axis indices being collapsed.
///
/// # Returns
///
/// Axis indices that survive the reduction.
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

/// Shape of the sub-array folded into each output slot.
///
/// Each reduced axis contributes its length; the product is
/// [`ReducePlan::reduction_len`].
///
/// # Arguments
///
/// * `shape` - Input array shape.
/// * `reduced` - Sorted axis indices being collapsed.
///
/// # Returns
///
/// Dimension lengths along reduced axes only, in ascending axis order.
pub(crate) fn reduced_shape(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
    reduced.iter().map(|&ax| shape[ax]).collect()
}

/// Output shape when reduced axes are kept as length-one dimensions.
///
/// Used when `keepdims = true`. Collapsed axes remain in the shape tuple
/// with size 1 so broadcasting semantics match NumPy.
///
/// # Arguments
///
/// * `shape` - Input array shape.
/// * `reduced` - Sorted axis indices being collapsed.
///
/// # Returns
///
/// A copy of `shape` with reduced axes set to 1.
pub(crate) fn keepdims_shape(shape: &[usize], reduced: &[usize]) -> Vec<usize> {
    let mut out = shape.to_vec();
    for &ax in reduced {
        out[ax] = 1;
    }
    out
}

/// Final output shape for a reduction, honoring `keepdims`.
///
/// Dispatches to [`keepdims_shape`] or [`kept_shape`] depending on the
/// caller flag.
///
/// # Arguments
///
/// * `shape` - Input array shape.
/// * `reduced` - Sorted axis indices being collapsed.
/// * `keepdims` - When true, reduced axes stay as length-1 dimensions.
///
/// # Returns
///
/// The shape every reduction kernel writes into.
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

/// Geometry for ops that scan one axis while filling independent outputs.
///
/// Used by cumulative scans and arg-extremum along a single axis. Records
/// which axis is traversed and how many independent output slots exist.
#[derive(Clone, Debug)]
pub(crate) struct AxisTraversalPlan {
    pub(crate) axis: usize,
    pub(crate) axis_len: usize,
    pub(crate) kept_axes: Vec<usize>,
    pub(crate) kept_shape: Vec<usize>,
    pub(crate) output_len: usize,
}

impl AxisTraversalPlan {
    /// Plan a single-axis traversal (cumsum, argmin along one axis, etc.).
    ///
    /// Treats `axis` as the sole reduced dimension and precomputes outer
    /// shape metadata for strided or contiguous kernel dispatch.
    ///
    /// # Arguments
    ///
    /// * `shape` - Input array shape.
    /// * `axis` - Normalized axis index to scan along.
    ///
    /// # Returns
    ///
    /// An [`AxisTraversalPlan`] with `output_len` equal to the product of
    /// non-scanned dimensions.
    pub(crate) fn new(shape: &[usize], axis: usize) -> Self {
        let axis_len = shape[axis];
        let reduced_axes = [axis];
        let axes = kept_axes(shape.len(), &reduced_axes);
        let kept_shape = kept_shape(shape, &reduced_axes);
        Self {
            axis,
            axis_len,
            output_len: size_of_shape_unchecked(&kept_shape),
            kept_axes: axes,
            kept_shape,
        }
    }

    /// Strides of non-traversed axes in outer C order.
    ///
    /// Outer run walkers use these strides to advance between independent
    /// scan lines without touching the traversed axis stride.
    ///
    /// # Arguments
    ///
    /// * `strides` - Full input array strides.
    ///
    /// # Returns
    ///
    /// Strides along [`Self::kept_axes`], in ascending axis order.
    pub(crate) fn kept_strides(&self, strides: &[isize]) -> Vec<isize> {
        self.kept_axes.iter().map(|&axis| strides[axis]).collect()
    }

    /// Whether the traversed axis is the last logical dimension.
    ///
    /// When true and the buffer is C-contiguous, each output row is one
    /// contiguous memory chunk.
    ///
    /// # Arguments
    ///
    /// * `ndim` - Rank of the input array.
    ///
    /// # Returns
    ///
    /// `true` when `axis + 1 == ndim`.
    #[inline]
    pub(crate) fn is_last_axis(&self, ndim: usize) -> bool {
        self.axis + 1 == ndim
    }
}

/// Precomputed axis layout shared by all reduction kernels on one call.
///
/// Built once per public reduction; stores output shape, slot counts, and
/// enough metadata to pick a [`TraversalSchedule`].
#[derive(Clone, Debug)]
pub(crate) struct ReducePlan {
    pub(crate) reduced_axes: Vec<usize>,
    pub(crate) kept_axes: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) kept_shape: Vec<usize>,
    pub(crate) reduced_shape: Vec<usize>,
    /// Independent output slots (product of kept dimensions).
    pub(crate) output_len: usize,
    /// Elements folded per slot (product of reduced dimensions).
    pub(crate) reduction_len: usize,
}

/// Physical memory walk chosen from plan geometry and layout.
///
/// Kernels branch once on this enum instead of re-deriving layout at every
/// inner loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraversalSchedule {
    /// Each output slot owns one trailing C-order chunk.
    SuffixContiguous,
    /// Prefix rows are scanned while all outputs accumulate in parallel.
    PrefixContiguous {
        reduced_len: usize,
        output_len: usize,
    },
    /// Arbitrary strides: outer `RunPlan` plus coalesced reduced-axis runs.
    GeneralStrided,
}

impl ReducePlan {
    /// Build a plan from input shape, axis selection, and `keepdims`.
    ///
    /// Normalizes axes, computes kept/reduced shapes, and checks that output
    /// and reduction lengths fit in `usize`.
    ///
    /// # Arguments
    ///
    /// * `shape` - Input array shape.
    /// * `axes` - Axes to reduce, or `None` for all axes.
    /// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
    ///
    /// # Returns
    ///
    /// A [`ReducePlan`] shared by every kernel on this reduction call.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid axis indices or shape products that
    /// overflow.
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
            output_len: checked_size_of_shape(&kept_shape)?,
            reduction_len: checked_size_of_shape(&reduced_shape)?,
            kept_shape,
            reduced_shape,
        })
    }

    /// True when the reduced block has zero elements.
    ///
    /// Callers use this to select an identity, sentinel, or structured error
    /// without touching input memory.
    ///
    /// # Returns
    ///
    /// `true` when [`Self::reduction_len`] is zero.
    #[inline]
    pub(crate) fn reduction_is_empty(&self) -> bool {
        self.reduction_len == 0
    }

    /// True when reduced axes form a trailing contiguous block
    /// `[ndim - k, …, ndim - 1]`.
    ///
    /// Does not inspect memory layout; pair with
    /// [`Array::as_c_contiguous_slice`].
    ///
    /// # Arguments
    ///
    /// * `ndim` - Rank of the input array.
    ///
    /// # Returns
    ///
    /// `true` when reduced axes are a suffix of `[0, ndim)`.
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

    /// True when reduced axes form a leading block `[0, …, k - 1]`.
    ///
    /// Prefix reductions scan contiguous rows and update every output slot in
    /// parallel.
    ///
    /// # Returns
    ///
    /// `true` when reduced axes are `[0, 1, …, k - 1]`.
    #[inline]
    pub(crate) fn is_prefix_reduction(&self) -> bool {
        self.reduced_axes
            .iter()
            .enumerate()
            .all(|(axis, &reduced)| axis == reduced)
    }

    /// Pick a traversal schedule from plan shape and C-contiguity.
    ///
    /// Non-contiguous buffers always use [`TraversalSchedule::GeneralStrided`].
    /// Contiguous buffers prefer suffix chunks, then prefix rows.
    ///
    /// # Arguments
    ///
    /// * `ndim` - Rank of the input array.
    /// * `is_c_contiguous` - Whether the buffer is C-contiguous with
    ///   offset zero.
    ///
    /// # Returns
    ///
    /// The schedule kernels should use for this plan and layout.
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

    /// Strides along kept and reduced axes, respectively.
    ///
    /// General-strided kernels split outer kept-axis walks from inner
    /// reduced-axis runs using these stride vectors.
    ///
    /// # Arguments
    ///
    /// * `strides` - Full input array strides.
    ///
    /// # Returns
    ///
    /// `(kept_strides, reduced_strides)` in ascending axis order within each
    /// group.
    pub(crate) fn kept_reduced_strides(
        &self,
        strides: &[isize],
    ) -> (Vec<isize>, Vec<isize>) {
        let kept = self.kept_axes.iter().map(|&ax| strides[ax]).collect();
        let reduced = self.reduced_axes.iter().map(|&ax| strides[ax]).collect();
        (kept, reduced)
    }
}
