//! Prepared coalesced runs shared by strided kernels.

use std::array;

use crate::traversal::CoalescedLayout;
use crate::traversal::StrideCursor;

/// Address progression of one operand inside a prepared run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunKind {
    /// Consecutive elements (`stride == 1`).
    UnitStride,
    /// One element reused throughout the run (`stride == 0`).
    Repeated,
    /// Any other fixed stride.
    Strided,
}

impl RunKind {
    #[inline]
    fn from_stride(stride: isize) -> Self {
        match stride {
            1 => Self::UnitStride,
            0 => Self::Repeated,
            _ => Self::Strided,
        }
    }
}

/// One fixed-stride run for `N` jointly coalesced operands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run<const N: usize> {
    /// Buffer offsets at the start of the run.
    pub(crate) bases: [usize; N],
    /// Number of logical elements in the run.
    pub(crate) len: usize,
    /// Per-element buffer stride for each operand.
    pub(crate) strides: [isize; N],
    /// Classified stride kind for each operand.
    pub(crate) kinds: [RunKind; N],
}

/// Reusable run-grid traversal and linear-run description for `N` operands.
#[derive(Clone, Debug)]
pub(crate) struct RunPlan<const N: usize> {
    run_grid_shape: Vec<usize>,
    run_grid_strides: [Vec<isize>; N],
    run_count: usize,
    run_len: usize,
    operand_strides: [isize; N],
    operand_kinds: [RunKind; N],
}

impl<const N: usize> RunPlan<N> {
    /// Jointly coalesce operands over one logical shape.
    pub(crate) fn new(shape: &[usize], strides: [&[isize]; N]) -> Self {
        let layout = CoalescedLayout::new(shape, &strides);
        let run_grid_shape = layout.run_grid_shape().to_vec();
        let run_grid_strides =
            array::from_fn(|operand| layout.run_grid_strides(operand).to_vec());
        let operand_strides =
            array::from_fn(|operand| layout.operand_stride(operand));
        let operand_kinds = operand_strides.map(RunKind::from_stride);
        Self {
            run_count: layout.run_count(),
            run_len: layout.run_len(),
            run_grid_shape,
            run_grid_strides,
            operand_strides,
            operand_kinds,
        }
    }

    /// Number of runs.
    #[inline]
    pub(crate) fn run_count(&self) -> usize {
        self.run_count
    }

    /// Number of elements in each run.
    #[inline]
    pub(crate) fn run_len(&self) -> usize {
        self.run_len
    }

    /// Fixed per-element stride for one operand.
    #[inline]
    pub(crate) fn operand_stride(&self, operand: usize) -> isize {
        self.operand_strides[operand]
    }

    /// Create a run-grid cursor at the supplied operand offsets.
    pub(crate) fn cursor(&self, offsets: [isize; N]) -> StrideCursor<'_, N> {
        let stride_refs =
            array::from_fn(|i| self.run_grid_strides[i].as_slice());
        StrideCursor::new(&self.run_grid_shape, stride_refs, offsets)
    }

    /// Visit every prepared run from the supplied operand offsets.
    pub(crate) fn for_each(
        &self,
        offsets: [isize; N],
        mut visit: impl FnMut(Run<N>),
    ) {
        if self.run_len == 0 || self.run_count == 0 {
            return;
        }
        let mut run_grid = self.cursor(offsets);
        for run_index in 0..self.run_count {
            let bases = array::from_fn(|i| run_grid.operand_offset(i));
            visit(Run {
                bases,
                len: self.run_len,
                strides: self.operand_strides,
                kinds: self.operand_kinds,
            });
            if run_index + 1 < self.run_count {
                run_grid.advance();
            }
        }
    }

    /// Fallible counterpart of [`Self::for_each`].
    pub(crate) fn try_for_each<E>(
        &self,
        offsets: [isize; N],
        mut visit: impl FnMut(Run<N>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        if self.run_len == 0 || self.run_count == 0 {
            return Ok(());
        }
        let mut run_grid = self.cursor(offsets);
        for run_index in 0..self.run_count {
            let bases = array::from_fn(|i| run_grid.operand_offset(i));
            visit(Run {
                bases,
                len: self.run_len,
                strides: self.operand_strides,
                kinds: self.operand_kinds,
            })?;
            if run_index + 1 < self.run_count {
                run_grid.advance();
            }
        }
        Ok(())
    }

    /// Visit every logical element as jointly advanced operand offsets.
    pub(crate) fn for_each_element(
        &self,
        offsets: [isize; N],
        mut visit: impl FnMut([usize; N]),
    ) {
        self.for_each(offsets, |run| {
            let mut positions = run.bases.map(|base| base as isize);
            for _ in 0..run.len {
                visit(positions.map(|position| position as usize));
                for (position, stride) in positions.iter_mut().zip(run.strides)
                {
                    *position += stride;
                }
            }
        });
    }
}

/// Collect one operand in logical run order.
pub(crate) fn collect_unary<A: Copy, Out>(
    plan: &RunPlan<1>,
    data: &[A],
    offset: usize,
    map: impl FnMut(A) -> Out,
) -> Vec<Out> {
    let mut out = Vec::with_capacity(plan.run_count * plan.run_len);
    extend_unary(plan, data, offset, &mut out, map);
    out
}

/// Append one operand in logical run order to an existing output.
pub(crate) fn extend_unary<A: Copy, Out>(
    plan: &RunPlan<1>,
    data: &[A],
    offset: usize,
    out: &mut Vec<Out>,
    mut map: impl FnMut(A) -> Out,
) {
    plan.for_each([offset as isize], |run| match run.kinds[0] {
        RunKind::UnitStride => {
            out.extend(
                data[run.bases[0]..run.bases[0] + run.len]
                    .iter()
                    .copied()
                    .map(&mut map),
            );
        }
        RunKind::Repeated => {
            let value = data[run.bases[0]];
            out.extend((0..run.len).map(|_| map(value)));
        }
        RunKind::Strided => {
            let mut position = run.bases[0] as isize;
            for _ in 0..run.len {
                out.push(map(data[position as usize]));
                position += run.strides[0];
            }
        }
    });
}

/// Collect two operands in logical run order.
pub(crate) fn collect_binary<A: Copy, B: Copy, Out>(
    plan: &RunPlan<2>,
    left: &[A],
    right: &[B],
    offsets: [usize; 2],
    mut map: impl FnMut(A, B) -> Out,
) -> Vec<Out> {
    let mut out = Vec::with_capacity(plan.run_count * plan.run_len);
    plan.for_each(offsets.map(|offset| offset as isize), |run| {
        match (run.kinds[0], run.kinds[1]) {
            (RunKind::UnitStride, RunKind::UnitStride) => {
                let xs = &left[run.bases[0]..run.bases[0] + run.len];
                let ys = &right[run.bases[1]..run.bases[1] + run.len];
                out.extend(
                    xs.iter()
                        .copied()
                        .zip(ys.iter().copied())
                        .map(|(x, y)| map(x, y)),
                );
            }
            (RunKind::UnitStride, RunKind::Repeated) => {
                let y = right[run.bases[1]];
                out.extend(
                    left[run.bases[0]..run.bases[0] + run.len]
                        .iter()
                        .copied()
                        .map(|x| map(x, y)),
                );
            }
            (RunKind::Repeated, RunKind::UnitStride) => {
                let x = left[run.bases[0]];
                out.extend(
                    right[run.bases[1]..run.bases[1] + run.len]
                        .iter()
                        .copied()
                        .map(|y| map(x, y)),
                );
            }
            (RunKind::Repeated, RunKind::Repeated) => {
                let x = left[run.bases[0]];
                let y = right[run.bases[1]];
                out.extend((0..run.len).map(|_| map(x, y)));
            }
            _ => {
                let mut lhs = run.bases[0] as isize;
                let mut rhs = run.bases[1] as isize;
                for _ in 0..run.len {
                    out.push(map(left[lhs as usize], right[rhs as usize]));
                    lhs += run.strides[0];
                    rhs += run.strides[1];
                }
            }
        }
    });
    out
}

/// Fallible binary collection, kept separate from the infallible hot loop.
pub(crate) fn try_collect_binary<A: Copy, B: Copy, Out, E>(
    plan: &RunPlan<2>,
    left: &[A],
    right: &[B],
    offsets: [usize; 2],
    mut map: impl FnMut(A, B) -> std::result::Result<Out, E>,
) -> std::result::Result<Vec<Out>, E> {
    let mut out = Vec::with_capacity(plan.run_count * plan.run_len);
    plan.try_for_each(offsets.map(|offset| offset as isize), |run| {
        match (run.kinds[0], run.kinds[1]) {
            (RunKind::UnitStride, RunKind::UnitStride) => {
                let left = &left[run.bases[0]..run.bases[0] + run.len];
                let right = &right[run.bases[1]..run.bases[1] + run.len];
                for (&left, &right) in left.iter().zip(right) {
                    out.push(map(left, right)?);
                }
            }
            (RunKind::UnitStride, RunKind::Repeated) => {
                let right = right[run.bases[1]];
                for &left in &left[run.bases[0]..run.bases[0] + run.len] {
                    out.push(map(left, right)?);
                }
            }
            (RunKind::Repeated, RunKind::UnitStride) => {
                let left = left[run.bases[0]];
                for &right in &right[run.bases[1]..run.bases[1] + run.len] {
                    out.push(map(left, right)?);
                }
            }
            (RunKind::Repeated, RunKind::Repeated) => {
                let left = left[run.bases[0]];
                let right = right[run.bases[1]];
                for _ in 0..run.len {
                    out.push(map(left, right)?);
                }
            }
            _ => {
                let mut lhs = run.bases[0] as isize;
                let mut rhs = run.bases[1] as isize;
                for _ in 0..run.len {
                    out.push(map(left[lhs as usize], right[rhs as usize])?);
                    lhs += run.strides[0];
                    rhs += run.strides[1];
                }
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// Collect three operands, specializing common contiguous/broadcast runs.
pub(crate) fn collect_ternary<A: Copy, B: Copy, C: Copy, Out>(
    plan: &RunPlan<3>,
    first: &[A],
    second: &[B],
    third: &[C],
    offsets: [usize; 3],
    mut map: impl FnMut(A, B, C) -> Out,
) -> Vec<Out> {
    let mut out = Vec::with_capacity(plan.run_count * plan.run_len);
    plan.for_each(offsets.map(|offset| offset as isize), |run| {
        match (run.kinds[0], run.kinds[1], run.kinds[2]) {
            (RunKind::UnitStride, RunKind::UnitStride, RunKind::UnitStride) => {
                let first = &first[run.bases[0]..run.bases[0] + run.len];
                let second = &second[run.bases[1]..run.bases[1] + run.len];
                let third = &third[run.bases[2]..run.bases[2] + run.len];
                out.extend(
                    first
                        .iter()
                        .copied()
                        .zip(second.iter().copied())
                        .zip(third.iter().copied())
                        .map(|((a, b), c)| map(a, b, c)),
                );
            }
            (RunKind::UnitStride, RunKind::Repeated, RunKind::UnitStride) => {
                let repeated = second[run.bases[1]];
                let first = &first[run.bases[0]..run.bases[0] + run.len];
                let third = &third[run.bases[2]..run.bases[2] + run.len];
                out.extend(
                    first
                        .iter()
                        .copied()
                        .zip(third.iter().copied())
                        .map(|(a, c)| map(a, repeated, c)),
                );
            }
            (RunKind::UnitStride, RunKind::UnitStride, RunKind::Repeated) => {
                let repeated = third[run.bases[2]];
                let first = &first[run.bases[0]..run.bases[0] + run.len];
                let second = &second[run.bases[1]..run.bases[1] + run.len];
                out.extend(
                    first
                        .iter()
                        .copied()
                        .zip(second.iter().copied())
                        .map(|(a, b)| map(a, b, repeated)),
                );
            }
            (RunKind::UnitStride, RunKind::Repeated, RunKind::Repeated) => {
                let second = second[run.bases[1]];
                let third = third[run.bases[2]];
                out.extend(
                    first[run.bases[0]..run.bases[0] + run.len]
                        .iter()
                        .copied()
                        .map(|a| map(a, second, third)),
                );
            }
            _ => {
                let mut positions = run.bases.map(|base| base as isize);
                for _ in 0..run.len {
                    out.push(map(
                        first[positions[0] as usize],
                        second[positions[1] as usize],
                        third[positions[2] as usize],
                    ));
                    for (position, stride) in
                        positions.iter_mut().zip(run.strides)
                    {
                        *position += stride;
                    }
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::c_order_strides;
    use crate::traversal::StrideIter;

    fn run_offsets(
        shape: &[usize],
        strides: &[isize],
        offset: usize,
    ) -> Vec<usize> {
        let plan = RunPlan::new(shape, [strides]);
        let mut offsets = Vec::new();
        plan.for_each([offset as isize], |run| {
            let mut position = run.bases[0] as isize;
            for _ in 0..run.len {
                offsets.push(position as usize);
                position += run.strides[0];
            }
        });
        offsets
    }

    fn assert_matches(shape: &[usize], strides: &[isize], offset: usize) {
        assert_eq!(
            run_offsets(shape, strides, offset),
            StrideIter::new(shape, strides, offset).collect::<Vec<_>>()
        );
    }

    #[test]
    fn classifies_inner_runs() {
        let contiguous = RunPlan::new(&[2, 3], [&[3, 1]]);
        assert_eq!(contiguous.operand_kinds[0], RunKind::UnitStride);

        let repeated = RunPlan::new(&[2, 3], [&[0, 0]]);
        assert_eq!(repeated.operand_kinds[0], RunKind::Repeated);

        let strided = RunPlan::new(&[2, 3], [&[1, 2]]);
        assert_eq!(strided.operand_kinds[0], RunKind::Strided);
    }

    #[test]
    fn matches_stride_iter_layouts() {
        assert_matches(&[], &[], 0);
        assert_matches(&[2, 3], &c_order_strides(&[2, 3]), 0);
        assert_matches(&[3, 2], &[1, 3], 0);
        assert_matches(&[2, 1, 3], &[3, 99, 1], 0);
        assert_matches(&[2, 3], &[4, 1], 2);
        assert_matches(&[2, 3], &[3, -1], 2);
        assert_matches(&[2, 3], &[0, 1], 0);
        assert_matches(&[2, 0, 3], &[0, 3, 1], 0);
    }

    #[test]
    fn binary_collection_hoists_repeated_operand() {
        let plan = RunPlan::new(&[2, 3], [&[3, 1], &[0, 0]]);
        let left = [1, 2, 3, 4, 5, 6];
        let scalar = [10];
        assert_eq!(
            collect_binary(&plan, &left, &scalar, [0, 0], |x, y| x + y),
            vec![11, 12, 13, 14, 15, 16]
        );
    }
}
