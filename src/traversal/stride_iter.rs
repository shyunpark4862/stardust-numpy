//! C-order buffer-index iteration over arbitrary strided layouts.
//!
//! [`StrideIter`] and [`StrideCursor`] walk memory without recomputing
//! `unravel_index` each step. They underpin non-contiguous ufunc paths,
//! fancy-index normalization, and higher-level array iterators.

use crate::shape::size_of_shape_unchecked;

/// Reusable C-order cursor over one shape and one or more stride layouts.
///
/// Maintains a shared logical multi-index while tracking independent buffer
/// offsets per operand. Used as the run-grid walker inside [`super::RunPlan`]:
/// advancing the cursor jumps to the next coalesced inner run without nested
/// loops over the full logical shape. `N = 1` covers ordinary strided walks;
/// `N = 2` lets cumulative kernels advance input and output together.
#[derive(Debug, Clone)]
pub(crate) struct StrideCursor<'a, const N: usize> {
    shape: &'a [usize],
    strides: [&'a [isize]; N],
    indices: Vec<usize>,
    offsets: [isize; N],
}

impl<'a, const N: usize> StrideCursor<'a, N> {
    /// Create a cursor at the supplied operand buffer offsets.
    ///
    /// Initializes logical indices to zero and records starting offsets. The
    /// cursor is positioned at the first C-order coordinate of `shape`.
    ///
    /// # Arguments
    ///
    /// * `shape` - Logical axis lengths (run-grid shape or full array shape).
    /// * `strides` - Per-operand stride slice, one entry per axis in `shape`.
    /// * `offsets` - Starting buffer offset in elements for each operand.
    ///
    /// # Returns
    ///
    /// A cursor ready for [`Self::advance`] or [`Self::operand_offset`].
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub(crate) fn new(
        shape: &'a [usize],
        strides: [&'a [isize]; N],
        offsets: [isize; N],
    ) -> Self {
        Self {
            shape,
            strides,
            indices: vec![0; shape.len()],
            offsets,
        }
    }

    /// Current backing-buffer offset for one operand.
    ///
    /// Equivalent to `base_offset + Σ indices[k] * strides[operand][k]` but
    /// maintained incrementally by [`Self::advance`].
    ///
    /// # Arguments
    ///
    /// * `operand` - Operand index in `0..N`.
    ///
    /// # Returns
    ///
    /// Buffer index in elements for that operand at the current coordinate.
    ///
    /// # Errors
    ///
    /// This function does not fail. Callers must keep `operand` in range.
    #[inline]
    pub(crate) fn operand_offset(&self, operand: usize) -> usize {
        self.offsets[operand] as usize
    }

    /// Current logical multi-index in C-order coordinates.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A slice parallel to `shape` giving the current index along each axis.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    #[inline]
    pub(crate) fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Reset coordinates and replace all operand offsets.
    ///
    /// Rewinds the multi-index to all zeros and sets new base offsets. Useful
    /// when reusing a cursor across multiple passes over the same layout.
    ///
    /// # Arguments
    ///
    /// * `offsets` - New starting buffer offset per operand.
    ///
    /// # Returns
    ///
    /// Nothing; mutates the cursor in place.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub(crate) fn reset(&mut self, offsets: [isize; N]) {
        self.indices.fill(0);
        self.offsets = offsets;
    }

    /// Advance all operands to the next C-order logical coordinate.
    ///
    /// Implements odometer-style carry from the innermost axis outward. On
    /// carry, offsets are adjusted by subtracting `stride * (len - 1)` rather
    /// than recomputing from scratch—critical when this drives run-grid steps
    /// in coalesced traversal. No-op when `shape` is empty.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// Nothing; mutates indices and offsets in place.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub(crate) fn advance(&mut self) {
        if self.shape.is_empty() {
            return;
        }
        for axis in (0..self.shape.len()).rev() {
            if self.indices[axis] + 1 < self.shape[axis] {
                self.indices[axis] += 1;
                for operand in 0..N {
                    self.offsets[operand] += self.strides[operand][axis];
                }
                return;
            }

            // Carry: rewind this axis and continue leftward.
            let undo = self.shape[axis].saturating_sub(1) as isize;
            for operand in 0..N {
                self.offsets[operand] -= self.strides[operand][axis] * undo;
            }
            self.indices[axis] = 0;
        }
    }
}

/// C-order iterator yielding buffer indices for one strided layout.
///
/// Each step returns the current buffer offset without full unravel math.
/// Handles `stride == 0` (broadcast repetition) and empty shapes. Often used
/// when coalescing cannot merge to a single run and element-wise offsets are
/// needed directly.
#[derive(Debug, Clone)]
pub(crate) struct StrideIter<'a> {
    cursor: StrideCursor<'a, 1>,
    remaining: usize,
}

impl<'a> StrideIter<'a> {
    /// Create an iterator for a strided layout.
    ///
    /// # Arguments
    ///
    /// * `shape` - Logical axis lengths.
    /// * `strides` - Per-axis strides for the single operand.
    /// * `offset` - Base buffer index at logical origin `(0, …, 0)`.
    ///
    /// # Returns
    ///
    /// A [`StrideIter`] yielding `remaining == product(shape)` buffer indices.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub(crate) fn new(
        shape: &'a [usize],
        strides: &'a [isize],
        offset: usize,
    ) -> Self {
        Self {
            cursor: StrideCursor::new(shape, [strides], [offset as isize]),
            remaining: size_of_shape_unchecked(shape),
        }
    }

    /// Visit each element as `(buffer_offset, multi_index)`.
    ///
    /// Consumes the iterator and calls `f` once per logical element in C-order.
    /// Prefer [`super::RunPlan`] when operands share coalesced runs.
    ///
    /// # Arguments
    ///
    /// * `f` - Called with buffer offset and current multi-index slice.
    ///
    /// # Returns
    ///
    /// Nothing; effects are entirely through `f`.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub(crate) fn for_each(mut self, mut f: impl FnMut(usize, &[usize])) {
        while self.remaining > 0 {
            f(self.cursor.operand_offset(0), self.cursor.indices());
            self.remaining -= 1;
            if self.remaining > 0 {
                self.cursor.advance();
            }
        }
    }
}

impl Iterator for StrideIter<'_> {
    type Item = usize;

    /// Yield the next buffer offset in C-order.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// `Some(offset)` while elements remain, then `None`.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.cursor.operand_offset(0);
        self.remaining -= 1;
        if self.remaining > 0 {
            self.cursor.advance();
        }
        Some(item)
    }

    /// Exact remaining element count.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// `(remaining, Some(remaining))` because length is known exactly.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StrideIter<'_> {}
