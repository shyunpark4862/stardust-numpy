//! Prepared coalesced runs shared by strided kernels.

use std::array;

use crate::layout::CoalescedLayout;
use crate::stride_iter::StrideCursor;

/// Address progression of one operand inside a prepared run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunKind {
    /// Consecutive elements (`stride == 1`).
    Contiguous,
    /// One element reused throughout the run (`stride == 0`).
    Repeated,
    /// Any other fixed stride.
    Strided,
}

impl RunKind {
    #[inline]
    fn from_stride(stride: isize) -> Self {
        match stride {
            1 => Self::Contiguous,
            0 => Self::Repeated,
            _ => Self::Strided,
        }
    }
}

/// One fixed-stride inner run for `N` jointly coalesced operands.
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

/// Reusable outer traversal and inner-run description for `N` operands.
#[derive(Clone, Debug)]
pub(crate) struct RunPlan<const N: usize> {
    outer_shape: Vec<usize>,
    outer_strides: [Vec<isize>; N],
    outer_len: usize,
    inner_len: usize,
    inner_strides: [isize; N],
    inner_kinds: [RunKind; N],
}

impl<const N: usize> RunPlan<N> {
    /// Jointly coalesce operands over one logical shape.
    pub(crate) fn new(shape: &[usize], strides: [&[isize]; N]) -> Self {
        let layout = CoalescedLayout::new(shape, &strides);
        let outer_shape = layout.outer_shape().to_vec();
        let outer_strides =
            array::from_fn(|operand| layout.outer_strides(operand).to_vec());
        let inner_strides =
            array::from_fn(|operand| layout.inner_stride(operand));
        let inner_kinds = inner_strides.map(RunKind::from_stride);
        Self {
            outer_len: layout.outer_len(),
            inner_len: layout.inner_len(),
            outer_shape,
            outer_strides,
            inner_strides,
            inner_kinds,
        }
    }

    /// Number of outer runs.
    #[inline]
    pub(crate) fn outer_len(&self) -> usize {
        self.outer_len
    }

    /// Number of elements in each inner run.
    #[inline]
    pub(crate) fn inner_len(&self) -> usize {
        self.inner_len
    }

    /// Fixed inner stride for one operand.
    #[inline]
    pub(crate) fn inner_stride(&self, operand: usize) -> isize {
        self.inner_strides[operand]
    }

    /// Create an outer cursor at the supplied operand offsets.
    pub(crate) fn cursor(&self, offsets: [isize; N]) -> StrideCursor<'_, N> {
        let stride_refs = array::from_fn(|i| self.outer_strides[i].as_slice());
        StrideCursor::new(&self.outer_shape, stride_refs, offsets)
    }

    /// Visit every prepared run from the supplied operand offsets.
    pub(crate) fn for_each(
        &self,
        offsets: [isize; N],
        mut visit: impl FnMut(Run<N>),
    ) {
        if self.inner_len == 0 || self.outer_len == 0 {
            return;
        }
        let mut outer = self.cursor(offsets);
        for outer_index in 0..self.outer_len {
            let bases = array::from_fn(|i| outer.buffer_index(i));
            visit(Run {
                bases,
                len: self.inner_len,
                strides: self.inner_strides,
                kinds: self.inner_kinds,
            });
            if outer_index + 1 < self.outer_len {
                outer.advance();
            }
        }
    }

    /// Fallible counterpart of [`Self::for_each`].
    pub(crate) fn try_for_each<E>(
        &self,
        offsets: [isize; N],
        mut visit: impl FnMut(Run<N>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        if self.inner_len == 0 || self.outer_len == 0 {
            return Ok(());
        }
        let mut outer = self.cursor(offsets);
        for outer_index in 0..self.outer_len {
            let bases = array::from_fn(|i| outer.buffer_index(i));
            visit(Run {
                bases,
                len: self.inner_len,
                strides: self.inner_strides,
                kinds: self.inner_kinds,
            })?;
            if outer_index + 1 < self.outer_len {
                outer.advance();
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
    let mut out = Vec::with_capacity(plan.outer_len * plan.inner_len);
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
        RunKind::Contiguous => {
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
    let mut out = Vec::with_capacity(plan.outer_len * plan.inner_len);
    plan.for_each(offsets.map(|offset| offset as isize), |run| {
        match (run.kinds[0], run.kinds[1]) {
            (RunKind::Contiguous, RunKind::Contiguous) => {
                let xs = &left[run.bases[0]..run.bases[0] + run.len];
                let ys = &right[run.bases[1]..run.bases[1] + run.len];
                out.extend(
                    xs.iter()
                        .copied()
                        .zip(ys.iter().copied())
                        .map(|(x, y)| map(x, y)),
                );
            }
            (RunKind::Contiguous, RunKind::Repeated) => {
                let y = right[run.bases[1]];
                out.extend(
                    left[run.bases[0]..run.bases[0] + run.len]
                        .iter()
                        .copied()
                        .map(|x| map(x, y)),
                );
            }
            (RunKind::Repeated, RunKind::Contiguous) => {
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
    let mut out = Vec::with_capacity(plan.outer_len * plan.inner_len);
    plan.try_for_each(offsets.map(|offset| offset as isize), |run| {
        let mut lhs = run.bases[0] as isize;
        let mut rhs = run.bases[1] as isize;
        for _ in 0..run.len {
            out.push(map(left[lhs as usize], right[rhs as usize])?);
            lhs += run.strides[0];
            rhs += run.strides[1];
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
    let mut out = Vec::with_capacity(plan.outer_len * plan.inner_len);
    plan.for_each(offsets.map(|offset| offset as isize), |run| {
        match (run.kinds[0], run.kinds[1], run.kinds[2]) {
            (RunKind::Contiguous, RunKind::Contiguous, RunKind::Contiguous) => {
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
            (RunKind::Contiguous, RunKind::Repeated, RunKind::Contiguous) => {
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
            (RunKind::Contiguous, RunKind::Contiguous, RunKind::Repeated) => {
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
            (RunKind::Contiguous, RunKind::Repeated, RunKind::Repeated) => {
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
    use crate::stride_iter::StrideIter;

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
        assert_eq!(contiguous.inner_kinds[0], RunKind::Contiguous);

        let repeated = RunPlan::new(&[2, 3], [&[0, 0]]);
        assert_eq!(repeated.inner_kinds[0], RunKind::Repeated);

        let strided = RunPlan::new(&[2, 3], [&[1, 2]]);
        assert_eq!(strided.inner_kinds[0], RunKind::Strided);
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
