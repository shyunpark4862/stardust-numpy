//! Coalesced inner layout for general-strided axis reductions.
//!
//! Reduced axes are flattened into a [`RunPlan`] grid once per call. Each
//! output slot resets a [`StrideCursor`] to its base and walks
//! `run_len` elements at a fixed operand stride, replacing per-element
//! N-dimensional carry with far fewer run boundaries.

use super::*;

/// Shared reduced-axis run geometry for general-strided kernels.
///
/// Built once per reduction call and reused for every outer output slot.
pub(crate) struct ReducedAxisRuns {
    plan: RunPlan<1>,
    pub(crate) run_count: usize,
    pub(crate) run_len: usize,
    pub(crate) operand_stride: isize,
}

impl ReducedAxisRuns {
    /// Build run metadata from reduced shape and per-axis strides.
    ///
    /// Coalesces the reduced sub-array into a small run grid so inner loops
    /// advance with a fixed stride instead of N-dimensional index carry.
    ///
    /// # Arguments
    ///
    /// * `reduced_shape` - Shape of axes folded into each output slot.
    /// * `reduced_strides` - Input strides along those axes.
    ///
    /// # Returns
    ///
    /// A [`ReducedAxisRuns`] describing inner walk length and stride.
    pub(crate) fn new(
        reduced_shape: &[usize],
        reduced_strides: &[isize],
    ) -> Self {
        let plan = RunPlan::new(reduced_shape, [reduced_strides]);
        let run_count = plan.run_count();
        let run_len = plan.run_len();
        let operand_stride = plan.operand_stride(0);
        Self {
            plan,
            run_count,
            run_len,
            operand_stride,
        }
    }

    /// Cursor over the run grid, reset per output slot.
    ///
    /// Each outer position calls [`StrideCursor::reset`] with its base
    /// offset before walking reduced-axis elements.
    ///
    /// # Arguments
    ///
    /// * `offset` - Base byte/element offset for the current outer slot.
    ///
    /// # Returns
    ///
    /// A [`StrideCursor`] positioned at `offset` on the run grid.
    pub(crate) fn cursor(&self, offset: isize) -> StrideCursor<'_, 1> {
        self.plan.cursor([offset])
    }
}
