//! Contiguous-or-strided buffer index iteration ([`StrideIter`]).
//!
//! Phase 7 may add a public `nditer`-style API on top of this primitive.

use crate::shape::size_of_shape;

/// Reusable C-order cursor over one shape and one or more strided layouts.
///
/// All lanes share the same logical multi-index while maintaining independent
/// buffer offsets. `N = 1` is used by normal strided traversal; `N = 2` lets
/// cumulative kernels advance input and output offsets in lockstep.
#[derive(Debug, Clone)]
pub(crate) struct StrideCursor<'a, const N: usize> {
    shape: &'a [usize],
    strides: [&'a [isize]; N],
    indices: Vec<usize>,
    offsets: [isize; N],
}

impl<'a, const N: usize> StrideCursor<'a, N> {
    /// Create a cursor at the supplied lane offsets.
    pub(crate) fn new(
        shape: &'a [usize],
        strides: [&'a [isize]; N],
        offsets: [isize; N],
    ) -> Self {
        debug_assert!(strides.iter().all(|s| s.len() == shape.len()));
        Self {
            shape,
            strides,
            indices: vec![0; shape.len()],
            offsets,
        }
    }

    /// Current buffer index for one lane.
    #[inline]
    pub(crate) fn buffer_index(&self, lane: usize) -> usize {
        debug_assert!(lane < N);
        debug_assert!(self.offsets[lane] >= 0);
        self.offsets[lane] as usize
    }

    /// Current logical multi-index.
    #[inline]
    pub(crate) fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Reset logical coordinates and replace all lane offsets.
    pub(crate) fn reset(&mut self, offsets: [isize; N]) {
        self.indices.fill(0);
        self.offsets = offsets;
    }

    /// Advance all lanes to the next logical C-order coordinate.
    pub(crate) fn advance(&mut self) {
        if self.shape.is_empty() {
            return;
        }
        for axis in (0..self.shape.len()).rev() {
            if self.indices[axis] + 1 < self.shape[axis] {
                self.indices[axis] += 1;
                for lane in 0..N {
                    self.offsets[lane] += self.strides[lane][axis];
                }
                return;
            }

            let undo = self.shape[axis].saturating_sub(1) as isize;
            for lane in 0..N {
                self.offsets[lane] -= self.strides[lane][axis] * undo;
            }
            self.indices[axis] = 0;
        }
    }
}

/// C-order walker over an arbitrary strided layout.
///
/// Yields **buffer indices** (`offset + Σ i_k * stride_k`) without
/// recomputing `unravel_index` each step. Used by ufunc non-contiguous
/// paths; Phase 7 can wrap this for a public iterator API.
///
/// Handles `stride == 0` (broadcast) and empty/`0`-size shapes.
#[derive(Debug, Clone)]
pub(crate) struct StrideIter<'a> {
    cursor: StrideCursor<'a, 1>,
    remaining: usize,
}

impl<'a> StrideIter<'a> {
    /// Create an iterator for `shape` / `strides` / `offset`.
    ///
    /// `shape.len()` must equal `strides.len()`.
    pub(crate) fn new(
        shape: &'a [usize],
        strides: &'a [isize],
        offset: usize,
    ) -> Self {
        debug_assert_eq!(shape.len(), strides.len());
        Self {
            cursor: StrideCursor::new(shape, [strides], [offset as isize]),
            remaining: size_of_shape(shape),
        }
    }

    /// Visit each logical element as `(buffer_index, multi_index)`.
    pub(crate) fn for_each(mut self, mut f: impl FnMut(usize, &[usize])) {
        while self.remaining > 0 {
            f(self.cursor.buffer_index(0), self.cursor.indices());
            self.remaining -= 1;
            if self.remaining > 0 {
                self.cursor.advance();
            }
        }
    }
}

impl Iterator for StrideIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.cursor.buffer_index(0);
        self.remaining -= 1;
        if self.remaining > 0 {
            self.cursor.advance();
        }
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StrideIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::c_order_strides;

    #[test]
    fn stride_iter_c_order_2d() {
        let shape = [2, 3];
        let strides = c_order_strides(&shape);
        let idxs: Vec<_> = StrideIter::new(&shape, &strides, 0).collect();
        assert_eq!(idxs, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn stride_iter_transpose_layout() {
        // Logical (3,2) view of C-order (2,3) buffer: strides (1, 3)
        let shape = [3, 2];
        let strides = [1_isize, 3];
        let idxs: Vec<_> = StrideIter::new(&shape, &strides, 0).collect();
        // [0,0]=0, [0,1]=3, [1,0]=1, [1,1]=4, [2,0]=2, [2,1]=5
        assert_eq!(idxs, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn stride_iter_broadcast_stride0() {
        let shape = [2, 3];
        let strides = [0_isize, 1];
        let idxs: Vec<_> = StrideIter::new(&shape, &strides, 0).collect();
        assert_eq!(idxs, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn stride_iter_0d() {
        let idxs: Vec<_> = StrideIter::new(&[], &[], 0).collect();
        assert_eq!(idxs, vec![0]);
    }
}
