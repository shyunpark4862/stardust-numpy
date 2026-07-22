//! Shared layout normalization for generic strided traversal.
//!
//! [`CoalescedLayout`] takes a logical shape together with one or more
//! operand stride arrays (all already aligned to that shape) and reduces
//! them to the smallest equivalent axis list: singleton axes are dropped,
//! and adjacent axes are merged into one whenever *every* operand can
//! express them as a single linear run (`outer_stride == inner_stride *
//! inner_len`). The last remaining axis is then treated as a fixed-stride
//! "inner run"; everything above it is the "outer" traversal.
//!
//! This does not know about scatter, ufunc, or reduction semantics — it only
//! answers "how few axes does this layout need, and what is the innermost
//! linear run". Callers drive the actual outer traversal (typically with
//! [`crate::stride_iter::StrideCursor`]) and inner fixed-stride loop
//! themselves, since operand count and per-element behavior are kernel
//! specific.

use crate::shape::size_of_shape;

/// A shape plus per-operand strides, coalesced to the fewest axes that still
/// describe the same traversal for every operand.
///
/// The final axis (if any) is the innermost linear run: stepping through it
/// by [`Self::inner_stride`] for [`Self::inner_len`] steps reaches every
/// element that axis represents, for every operand, without revisiting or
/// skipping any. A fully mergeable layout coalesces down to exactly one
/// axis, so [`Self::outer_len`] is `1` and the whole traversal is one run.
#[derive(Debug, Clone)]
pub(crate) struct CoalescedLayout {
    shape: Vec<usize>,
    strides: Vec<Vec<isize>>,
}

impl CoalescedLayout {
    /// Coalesce `shape` against every entry in `operand_strides`.
    ///
    /// All stride arrays must have `shape.len()` entries (they describe the
    /// same logical shape from different operands' point of view, e.g. after
    /// broadcasting).
    pub(crate) fn new(shape: &[usize], operand_strides: &[&[isize]]) -> Self {
        debug_assert!(operand_strides.iter().all(|s| s.len() == shape.len()));
        let n_operands = operand_strides.len();

        // Drop singleton axes: they never affect address progression, and
        // keeping them around would block otherwise-valid merges.
        let mut kept_shape = Vec::with_capacity(shape.len());
        let mut kept_strides: Vec<Vec<isize>> =
            vec![Vec::with_capacity(shape.len()); n_operands];
        for (axis, &len) in shape.iter().enumerate() {
            if len == 1 {
                continue;
            }
            kept_shape.push(len);
            for (op, strides) in operand_strides.iter().enumerate() {
                kept_strides[op].push(strides[axis]);
            }
        }

        // Merge adjacent axes back-to-front. `shape_stack`/`stride_stack`
        // hold already-coalesced frames, innermost last; each candidate axis
        // either merges into the current innermost frame or starts a new one.
        let mut shape_stack: Vec<usize> = Vec::with_capacity(kept_shape.len());
        let mut stride_stack: Vec<Vec<isize>> =
            Vec::with_capacity(kept_shape.len());

        for axis in (0..kept_shape.len()).rev() {
            let axis_len = kept_shape[axis];
            let axis_strides: Vec<isize> =
                (0..n_operands).map(|op| kept_strides[op][axis]).collect();

            let merges = match (shape_stack.last(), stride_stack.last()) {
                (Some(&inner_len), Some(inner_strides)) => {
                    let inner_len = inner_len as isize;
                    axis_strides.iter().zip(inner_strides.iter()).all(
                        |(&outer_s, &inner_s)| {
                            inner_s.checked_mul(inner_len) == Some(outer_s)
                        },
                    )
                }
                _ => false,
            };

            if merges {
                let top = shape_stack.last_mut().unwrap();
                *top = match top.checked_mul(axis_len) {
                    Some(merged) => merged,
                    None => {
                        // Overflow: bail out of the merge, keep axes split.
                        shape_stack.push(axis_len);
                        stride_stack.push(axis_strides);
                        continue;
                    }
                };
            } else {
                shape_stack.push(axis_len);
                stride_stack.push(axis_strides);
            }
        }

        shape_stack.reverse();
        stride_stack.reverse();

        let mut strides: Vec<Vec<isize>> =
            vec![Vec::with_capacity(shape_stack.len()); n_operands];
        for axis_strides in &stride_stack {
            for (op, &s) in axis_strides.iter().enumerate() {
                strides[op].push(s);
            }
        }

        Self {
            shape: shape_stack,
            strides,
        }
    }

    /// Length of the innermost linear run (`1` for a fully-merged / 0-D /
    /// all-singleton layout — a single logical element).
    #[inline]
    pub(crate) fn inner_len(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    /// Fixed per-step stride of the innermost run, for one operand.
    #[inline]
    pub(crate) fn inner_stride(&self, operand: usize) -> isize {
        self.strides[operand].last().copied().unwrap_or(0)
    }

    /// Shape of the axes outside the innermost run (possibly empty).
    #[inline]
    pub(crate) fn outer_shape(&self) -> &[usize] {
        let n = self.shape.len();
        if n == 0 {
            &[]
        } else {
            &self.shape[..n - 1]
        }
    }

    /// Strides of the axes outside the innermost run, for one operand.
    #[inline]
    pub(crate) fn outer_strides(&self, operand: usize) -> &[isize] {
        let s = &self.strides[operand];
        let n = s.len();
        if n == 0 {
            &[]
        } else {
            &s[..n - 1]
        }
    }

    /// Number of inner runs to visit (product of [`Self::outer_shape`]).
    #[inline]
    pub(crate) fn outer_len(&self) -> usize {
        size_of_shape(self.outer_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::c_order_strides;
    use crate::stride_iter::{StrideCursor, StrideIter};

    /// Offsets produced by walking `layout` outer-then-inner, for one operand.
    fn coalesced_offsets(
        shape: &[usize],
        strides: &[isize],
        offset: usize,
    ) -> Vec<usize> {
        let layout = CoalescedLayout::new(shape, &[strides]);
        let inner_len = layout.inner_len();
        let outer_shape = layout.outer_shape();
        let outer_n = layout.outer_len();
        let inner_stride = layout.inner_stride(0);

        let mut out = Vec::new();
        if inner_len == 0 || outer_n == 0 {
            return out;
        }
        let mut outer = StrideCursor::new(
            outer_shape,
            [layout.outer_strides(0)],
            [offset as isize],
        );
        for outer_i in 0..outer_n {
            let mut pos = outer.buffer_index(0) as isize;
            for _ in 0..inner_len {
                out.push(pos as usize);
                pos += inner_stride;
            }
            if outer_i + 1 < outer_n {
                outer.advance();
            }
        }
        out
    }

    fn assert_matches_stride_iter(
        shape: &[usize],
        strides: &[isize],
        offset: usize,
    ) {
        let expected: Vec<usize> =
            StrideIter::new(shape, strides, offset).collect();
        let actual = coalesced_offsets(shape, strides, offset);
        assert_eq!(
            actual, expected,
            "mismatch for shape={shape:?} strides={strides:?} offset={offset}"
        );
    }

    #[test]
    fn matches_stride_iter_c_contiguous() {
        let shape = [4usize, 5];
        let strides = c_order_strides(&shape);
        assert_matches_stride_iter(&shape, &strides, 0);
        // Fully mergeable: one axis, one run.
        let layout = CoalescedLayout::new(&shape, &[&strides]);
        assert_eq!(layout.outer_len(), 1);
        assert_eq!(layout.inner_len(), 20);
        assert_eq!(layout.inner_stride(0), 1);
    }

    #[test]
    fn matches_stride_iter_transposed() {
        assert_matches_stride_iter(&[3, 2], &[1, 3], 0);
    }

    #[test]
    fn matches_stride_iter_single_strided_axis() {
        assert_matches_stride_iter(&[5], &[2], 0);
    }

    #[test]
    fn matches_stride_iter_multiple_strided_axes() {
        assert_matches_stride_iter(&[3, 4], &[10, 2], 0);
    }

    #[test]
    fn matches_stride_iter_negative_stride() {
        assert_matches_stride_iter(&[4], &[-1], 3);
        assert_matches_stride_iter(&[2, 3], &[-3, -1], 5);
    }

    #[test]
    fn matches_stride_iter_zero_stride_broadcast() {
        assert_matches_stride_iter(&[2, 3], &[0, 1], 0);
    }

    #[test]
    fn matches_stride_iter_nonzero_offset() {
        assert_matches_stride_iter(&[2, 3], &[3, 1], 7);
    }

    #[test]
    fn matches_stride_iter_singleton_axes() {
        assert_matches_stride_iter(&[1, 5, 1], &[100, 1, 7], 0);
        let layout = CoalescedLayout::new(&[1, 5, 1], &[&[100, 1, 7]]);
        assert_eq!(layout.outer_len(), 1);
        assert_eq!(layout.inner_len(), 5);
    }

    #[test]
    fn zero_d_is_one_element() {
        let layout = CoalescedLayout::new(&[], &[&[]]);
        assert_eq!(layout.outer_len(), 1);
        assert_eq!(layout.inner_len(), 1);
        assert_eq!(layout.inner_stride(0), 0);
        assert_matches_stride_iter(&[], &[], 0);
    }

    #[test]
    fn empty_dimension() {
        assert_matches_stride_iter(&[0, 3], &[3, 1], 0);
        assert_matches_stride_iter(&[3, 0], &[3, 1], 0);
        let layout = CoalescedLayout::new(&[3, 0], &[&[3, 1]]);
        assert_eq!(layout.outer_len() * layout.inner_len(), 0);
    }

    #[test]
    fn partially_coalescible_row_gap() {
        // shape=[1024,512] strides=[2048,2]: rows have a gap, so the outer
        // axis cannot merge into the inner one.
        let layout = CoalescedLayout::new(&[1024, 512], &[&[2048isize, 2]]);
        assert_eq!(layout.outer_shape(), &[1024]);
        assert_eq!(layout.outer_strides(0), &[2048]);
        assert_eq!(layout.inner_len(), 512);
        assert_eq!(layout.inner_stride(0), 2);
        assert_matches_stride_iter(&[1024, 512], &[2048, 2], 0);
    }

    #[test]
    fn fully_coalescible_layout() {
        // shape=[1024,512] strides=[1024,2]: rows are back-to-back, so this
        // merges into one run of len 524288, stride 2.
        let layout = CoalescedLayout::new(&[1024, 512], &[&[1024isize, 2]]);
        assert_eq!(layout.outer_len(), 1);
        assert_eq!(layout.inner_len(), 524288);
        assert_eq!(layout.inner_stride(0), 2);
        assert_matches_stride_iter(&[1024, 512], &[1024, 2], 0);
    }

    #[test]
    fn multiple_operands_differing_strides() {
        let shape = [4usize, 5];
        let a = [10isize, 2];
        let b = [5isize, 1];
        let layout = CoalescedLayout::new(&shape, &[&a, &b]);
        // Both operands individually merge (5*2=10, 1*5=5), and the merge
        // decision is taken jointly, so both collapse to one run.
        assert_eq!(layout.outer_len(), 1);
        assert_eq!(layout.inner_len(), 20);
        assert_eq!(layout.inner_stride(0), 2);
        assert_eq!(layout.inner_stride(1), 1);
    }

    #[test]
    fn broadcast_operand_joint_merge() {
        // shape=[4,5], lhs=[10,2] (real), rhs=[0,0] (broadcast read).
        // Both merge jointly into one run: lhs stride 2, rhs stride 0, len 20.
        let shape = [4usize, 5];
        let lhs = [10isize, 2];
        let rhs = [0isize, 0];
        let layout = CoalescedLayout::new(&shape, &[&lhs, &rhs]);
        assert_eq!(layout.outer_len(), 1);
        assert_eq!(layout.inner_len(), 20);
        assert_eq!(layout.inner_stride(0), 2);
        assert_eq!(layout.inner_stride(1), 0);
    }

    #[test]
    fn non_coalescible_when_one_operand_breaks_linearity() {
        // rhs breaks linear progression across the axis boundary (broadcast
        // on the outer axis only): must not jointly merge.
        let shape = [4usize, 5];
        let lhs = [10isize, 2];
        let rhs = [0isize, 1];
        let layout = CoalescedLayout::new(&shape, &[&lhs, &rhs]);
        assert_eq!(layout.outer_shape(), &[4]);
        assert_eq!(layout.inner_len(), 5);
        assert_eq!(layout.outer_strides(0), &[10]);
        assert_eq!(layout.outer_strides(1), &[0]);
        assert_eq!(layout.inner_stride(0), 2);
        assert_eq!(layout.inner_stride(1), 1);
    }
}
