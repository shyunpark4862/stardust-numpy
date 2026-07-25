//! Axis coalescing: shrink a logical shape to the fewest equivalent axes.
//!
//! Given one shape and several per-operand stride arrays (already broadcast-
//! aligned), [`CoalescedLayout`] drops length-1 axes and merges neighbors when
//! every operand sees a single linear run. The innermost remaining axis is
//! the fixed-stride run; axes above it form the run grid.

use crate::shape::size_of_shape_unchecked;

/// Broadcast-aligned shape and per-operand strides after coalescing.
///
/// Coalescing is the first step before building a [`super::RunPlan`]. Length-1
/// axes are removed because they never advance a buffer pointer. Adjacent axes
/// merge when every operand's outer stride equals `inner_stride * inner_len`,
/// meaning a nested loop can become one linear scan. The innermost surviving
/// axis is the **run**: stepping through it by [`Self::operand_stride`] for
/// [`Self::run_len`] steps visits every element without skips or repeats.
/// Axes above the run form the **run grid**, indexed by [`StrideCursor`](
/// super::StrideCursor) to jump between runs.
///
/// A fully merged layout has `run_count == 1` and one contiguous inner run.
#[derive(Debug, Clone)]
pub(crate) struct CoalescedLayout {
    shape: Vec<usize>,
    strides: Vec<Vec<isize>>,
}

impl CoalescedLayout {
    /// Build a coalesced layout from a logical shape and operand strides.
    ///
    /// This is the core stride-coalescing pass shared by ufuncs, reductions,
    /// and iterators. Operands must already share the same logical `shape`
    /// (after broadcasting); each row of `operand_strides` is that operand's
    /// stride vector in C-order axis order.
    ///
    /// The algorithm drops length-1 axes, then merges from innermost outward
    /// when `outer_stride == inner_stride * inner_len` for every operand.
    /// Overflow during merge keeps axes separate rather than producing wrong
    /// strides.
    ///
    /// # Arguments
    ///
    /// * `shape` - Logical broadcast-aligned shape (one entry per axis).
    /// * `operand_strides` - One stride slice per operand, same axis count as
    ///   `shape`.
    ///
    /// # Returns
    ///
    /// A [`CoalescedLayout`] whose [`Self::run_len`] and [`Self::run_count`]
    /// describe how kernels should split traversal into inner runs and an
    /// outer run grid.
    ///
    /// # Errors
    ///
    /// This function does not fail; stride overflow during merge prevents
    /// incorrect coalescing by leaving axes unmerged.
    pub(crate) fn new(shape: &[usize], operand_strides: &[&[isize]]) -> Self {
        let n_operands = operand_strides.len();

        // Length-1 axes never move the buffer pointer; drop them early.
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

        // Merge from innermost axis outward; stacks hold coalesced frames.
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
                    // Merge when outer_stride == inner_stride * inner_len.
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
                        // Overflow: keep axes separate rather than merge.
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

    /// Length of each inner linear run.
    ///
    /// After coalescing, the innermost axis (if any) is visited as a fixed-
    /// stride segment of this many elements. When the layout is fully merged
    /// or empty, this returns `1`.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// The product length of the innermost coalesced axis, or `1` when there
    /// are no non-trivial axes.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    #[inline]
    pub(crate) fn run_len(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    /// Fixed per-step stride inside the innermost run for one operand.
    ///
    /// Inside each run, advancing one logical element adds this value to the
    /// operand's buffer offset. `1` means contiguous memory; `0` means
    /// broadcast (the same element is reused for the whole run).
    ///
    /// # Arguments
    ///
    /// * `operand` - Index into the operand list passed to [`Self::new`].
    ///
    /// # Returns
    ///
    /// The coalesced inner stride for that operand, or `0` when the layout
    /// has no axes.
    ///
    /// # Errors
    ///
    /// This function does not fail. Callers must keep `operand` in range.
    #[inline]
    pub(crate) fn operand_stride(&self, operand: usize) -> isize {
        self.strides[operand].last().copied().unwrap_or(0)
    }

    /// Shape of the outer run grid.
    ///
    /// All axes except the innermost coalesced axis form a C-order grid. Each
    /// grid cell selects one inner run of length [`Self::run_len`]. An empty
    /// slice means a single run covers the whole array (`run_count == 1`).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A slice of axis lengths for the run grid (may be empty).
    ///
    /// # Errors
    ///
    /// This function does not fail.
    #[inline]
    pub(crate) fn run_grid_shape(&self) -> &[usize] {
        let n = self.shape.len();
        if n == 0 {
            &[]
        } else {
            &self.shape[..n - 1]
        }
    }

    /// Strides across the run grid for one operand.
    ///
    /// When the run-grid cursor advances along axis `k`, this operand's base
    /// offset changes by `run_grid_strides[k]`. These are the coalesced
    /// strides for all axes above the inner run.
    ///
    /// # Arguments
    ///
    /// * `operand` - Index into the operand list passed to [`Self::new`].
    ///
    /// # Returns
    ///
    /// A slice of per-axis strides for the run grid (may be empty).
    ///
    /// # Errors
    ///
    /// This function does not fail. Callers must keep `operand` in range.
    #[inline]
    pub(crate) fn run_grid_strides(&self, operand: usize) -> &[isize] {
        let s = &self.strides[operand];
        let n = s.len();
        if n == 0 {
            &[]
        } else {
            &s[..n - 1]
        }
    }

    /// Total number of inner runs to visit.
    ///
    /// This is the product of [`Self::run_grid_shape`]. Kernels iterate this
    /// many runs, each of length [`Self::run_len`], to cover every logical
    /// element without materializing a broadcast shape.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// The run-grid volume; `1` when there is no outer grid.
    ///
    /// # Errors
    ///
    /// This function does not fail. Empty grid shape yields `1`.
    #[inline]
    pub(crate) fn run_count(&self) -> usize {
        size_of_shape_unchecked(self.run_grid_shape())
    }
}
