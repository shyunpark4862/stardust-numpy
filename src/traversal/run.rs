//! Prepared coalesced runs shared by strided element-wise kernels.
//!
//! A [`RunPlan`] splits traversal into an outer run grid and an inner fixed-
//! stride segment per operand. Kernels specialize on [`RunKind`] (unit stride,
//! broadcast/repeated, or general strided) for contiguous fast paths.

use std::array;

use crate::traversal::CoalescedLayout;
use crate::traversal::StrideCursor;

/// How one operand's address advances inside a prepared run.
///
/// Classification lets hot loops dispatch without re-checking raw strides.
/// [`RunKind::Repeated`] corresponds to broadcast axes (`stride == 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunKind {
    /// Consecutive elements (`stride == 1`); enables slice `extend` fast paths.
    UnitStride,
    /// One element reused for the whole run (`stride == 0`, broadcast).
    Repeated,
    /// Any other fixed per-element stride inside the coalesced inner run.
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
///
/// Produced by [`RunPlan::for_each`] for each cell of the run grid. All
/// operands share the same logical run length; per-operand [`Self::strides`]
/// and [`Self::kinds`] describe how to walk memory inside the run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run<const N: usize> {
    /// Buffer offsets at the start of the run for each operand.
    pub(crate) bases: [usize; N],
    /// Number of logical elements in the run.
    pub(crate) len: usize,
    /// Per-element buffer stride for each operand.
    pub(crate) strides: [isize; N],
    /// Classified stride kind for each operand.
    pub(crate) kinds: [RunKind; N],
}

/// Reusable run-grid traversal plan for `N` broadcast-aligned operands.
///
/// Built once from a logical shape and per-operand strides via stride
/// coalescing ([`CoalescedLayout`]). Traversal is two-level:
///
/// 1. **Run grid** — outer C-order indices; [`StrideCursor`] advances operand
///    base offsets between runs.
/// 2. **Inner run** — a fixed-stride segment of length [`Self::run_len`];
///    each operand uses [`Self::operand_stride`] and a [`RunKind`] fast path.
///
/// This avoids materializing broadcast shapes while still exposing contiguous
/// slice loops when coalescing merges axes into unit-stride runs.
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
    /// Build a run plan by jointly coalescing `N` operands over one shape.
    ///
    /// Delegates to [`CoalescedLayout::new`], then caches run-grid geometry
    /// and per-operand inner-run stride kinds for repeated kernel dispatch.
    ///
    /// # Arguments
    ///
    /// * `shape` - Broadcast-aligned logical shape shared by all operands.
    /// * `strides` - One stride slice per operand, aligned with `shape`.
    ///
    /// # Returns
    ///
    /// A [`RunPlan`] ready for [`Self::for_each`], [`Self::cursor`], or the
    /// `collect_*` helpers.
    ///
    /// # Errors
    ///
    /// This function does not fail.
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

    /// Number of outer runs in the coalesced run grid.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// How many [`Run`] values [`Self::for_each`] will emit (may be `0` for
    /// empty arrays).
    ///
    /// # Errors
    ///
    /// This function does not fail.
    #[inline]
    pub(crate) fn run_count(&self) -> usize {
        self.run_count
    }

    /// Number of logical elements inside each inner run.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// Length of the coalesced innermost axis; `0` skips all traversal.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    #[inline]
    pub(crate) fn run_len(&self) -> usize {
        self.run_len
    }

    /// Fixed per-element stride for one operand inside each inner run.
    ///
    /// # Arguments
    ///
    /// * `operand` - Operand index in `0..N`.
    ///
    /// # Returns
    ///
    /// Coalesced inner stride (`1`, `0`, or general strided).
    ///
    /// # Errors
    ///
    /// This function does not fail. Callers must keep `operand` in range.
    #[inline]
    pub(crate) fn operand_stride(&self, operand: usize) -> isize {
        self.operand_strides[operand]
    }

    /// Create a run-grid cursor at the given operand buffer offsets.
    ///
    /// The cursor walks [`Self::run_count`] cells of the outer grid. Each
    /// cell yields base offsets for a [`Run`] via [`StrideCursor::operand_offset`].
    ///
    /// # Arguments
    ///
    /// * `offsets` - Starting buffer offset (in elements) per operand.
    ///
    /// # Returns
    ///
    /// A [`StrideCursor`] bound to this plan's run-grid shape and strides.
    ///
    /// # Errors
    ///
    /// This function does not fail.
    pub(crate) fn cursor(&self, offsets: [isize; N]) -> StrideCursor<'_, N> {
        let stride_refs =
            array::from_fn(|i| self.run_grid_strides[i].as_slice());
        StrideCursor::new(&self.run_grid_shape, stride_refs, offsets)
    }

    /// Visit every prepared run in C-order over the run grid.
    ///
    /// Coalesced stride iteration in action: instead of nested loops over the
    /// full logical shape, the callback receives compact [`Run`] records.
    /// Each run bundles base offsets, length, strides, and [`RunKind`] tags so
    /// kernels can use contiguous slices, broadcast loops, or general strided
    /// walks. Empty `run_len` or `run_count` is a no-op.
    ///
    /// # Arguments
    ///
    /// * `offsets` - Starting buffer offset per operand before the first run.
    /// * `visit` - Called once per run with bases and inner-run metadata.
    ///
    /// # Returns
    ///
    /// Nothing; effects are entirely through `visit`.
    ///
    /// # Errors
    ///
    /// This function does not fail. Use [`Self::try_for_each`] for fallible
    /// callbacks.
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

    /// Fallible version of [`Self::for_each`].
    ///
    /// Same coalesced run-grid walk; stops and propagates the first error from
    /// `visit`.
    ///
    /// # Arguments
    ///
    /// * `offsets` - Starting buffer offset per operand.
    /// * `visit` - Called once per run; may return an error to abort.
    ///
    /// # Returns
    ///
    /// `Ok(())` after all runs, or the first error from `visit`.
    ///
    /// # Errors
    ///
    /// Returns whatever error type `visit` returns on the first failing run.
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
    ///
    /// Expands each coalesced run into per-element coordinates. Useful when a
    /// kernel cannot specialize on [`RunKind`] but still benefits from run-grid
    /// coalescing for the outer loop. Inner steps add each run's per-operand
    /// strides, including broadcast (`0`) and general strided cases.
    ///
    /// # Arguments
    ///
    /// * `offsets` - Starting buffer offset per operand.
    /// * `visit` - Called once per logical element with buffer indices.
    ///
    /// # Returns
    ///
    /// Nothing; effects are entirely through `visit`.
    ///
    /// # Errors
    ///
    /// This function does not fail.
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

/// Collect one operand in logical C-order into a new vector.
///
/// Uses [`RunPlan::for_each`] with [`RunKind`] dispatch: unit-stride runs
/// copy via slice `extend`, repeated runs fill from one value, and strided
/// runs walk with a running offset.
///
/// # Arguments
///
/// * `plan` - Coalesced unary run plan for the source array's shape/strides.
/// * `data` - Backing storage of the source array.
/// * `offset` - Base buffer index where logical index `(0, …, 0)` lives.
/// * `map` - Per-element transform applied while collecting.
///
/// # Returns
///
/// A `Vec<Out>` in C-order with length `plan.run_count * plan.run_len`.
///
/// # Errors
///
/// This function does not fail.
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

/// Append one operand in logical C-order to an existing output vector.
///
/// Same coalesced run iteration as [`collect_unary`], but appends to `out`.
/// Handy for chained ufunc pipelines that reuse one allocation.
///
/// # Arguments
///
/// * `plan` - Coalesced unary run plan.
/// * `data` - Source backing storage.
/// * `offset` - Base buffer index for logical origin.
/// * `out` - Destination vector extended in C-order.
/// * `map` - Per-element transform.
///
/// # Returns
///
/// Nothing; appended elements are written to `out`.
///
/// # Errors
///
/// This function does not fail.
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

/// Collect two operands in logical C-order into a new vector.
///
/// Binary coalesced iteration: [`RunPlan`] merges both operands' strides, then
/// each run's [`RunKind`] pair selects among contiguous zip, broadcast-with-
/// contiguous, dual-broadcast, or general dual-strided inner loops.
///
/// # Arguments
///
/// * `plan` - Joint run plan for both operands over one broadcast shape.
/// * `left` - Left operand backing storage.
/// * `right` - Right operand backing storage.
/// * `offsets` - Base buffer index per operand at logical origin.
/// * `map` - Binary transform applied element-wise in C-order.
///
/// # Returns
///
/// A `Vec<Out>` with one entry per logical element.
///
/// # Errors
///
/// This function does not fail. See [`try_collect_binary`] for fallible maps.
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

/// Fallible binary collection over a coalesced [`RunPlan`].
///
/// Mirrors [`collect_binary`] but propagates errors from `map` (e.g. integer
/// division). Kept separate so the infallible hot loop stays branch-light.
///
/// # Arguments
///
/// * `plan` - Joint binary run plan.
/// * `left` - Left operand storage.
/// * `right` - Right operand storage.
/// * `offsets` - Base buffer indices per operand.
/// * `map` - Fallible binary transform.
///
/// # Returns
///
/// `Ok(Vec<Out>)` in C-order on success.
///
/// # Errors
///
/// Returns the first error produced by `map`, or from [`RunPlan::try_for_each`].
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
///
/// Ternary coalesced iteration with hand-tuned inner loops for frequent
/// `(UnitStride, Repeated, UnitStride)` and all-contiguous patterns; falls
/// back to per-element strided advancement for uncommon combinations.
///
/// # Arguments
///
/// * `plan` - Joint run plan for three broadcast-aligned operands.
/// * `first` - First operand storage.
/// * `second` - Second operand storage.
/// * `third` - Third operand storage.
/// * `offsets` - Base buffer index per operand.
/// * `map` - Ternary transform in C-order.
///
/// # Returns
///
/// A `Vec<Out>` with one entry per logical element.
///
/// # Errors
///
/// This function does not fail.
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
