use super::*;

/// Coalesced reduced-axis layout shared by every general-strided reduction
/// kernel below.
///
/// Built once per call from `plan.reduced_shape`/reduced strides; each output
/// position then resets a cursor over the run grid to its own base offset and
/// walks [`Self::run_len`] elements at a fixed [`Self::operand_stride`] —
/// replacing the old per-reduced-element N-dimensional carry with a run-grid
/// traversal over far fewer axes (often
/// exactly one) plus a linear inner run.
pub(crate) struct ReducedAxisRuns {
    plan: RunPlan<1>,
    pub(crate) run_count: usize,
    pub(crate) run_len: usize,
    pub(crate) operand_stride: isize,
}

impl ReducedAxisRuns {
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

    /// A cursor over the run grid, ready to reset to an output base offset.
    pub(crate) fn cursor(&self, offset: isize) -> StrideCursor<'_, 1> {
        self.plan.cursor([offset])
    }
}
