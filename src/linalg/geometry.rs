//! Contraction geometry: batch broadcasting, ranks, and diagonal layout.
//!
//! `MatmulPlan` and `DiagonalPlan` capture every stride an execution kernel
//! needs. Planning is separate from numeric work so kernels can focus on
//! memory access patterns (IKJ tiles, diagonal walks, batch iteration).

use crate::array::Array;
use crate::axis::normalize_axis;
use crate::broadcast::broadcast_shape;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::linalg::diagonal_geometry::{diagonal_geometry, DiagonalGeometry};
use crate::shape::checked_size_of_shape;

/// Rank pattern recognized by [`plan_dot`].
///
/// Each variant names the effective matrix/vector ranks after treating 1-D
/// operands as row or column vectors. The shared contraction axis is always
/// the **inner** (last) axis of the left operand and the **leading** axis of
/// the right operand (length `K` in `(M, K) · (K, N)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DotKind {
    /// `(K,) · (K,) → scalar`; one contraction axis, no batch.
    VectorVector,
    /// `(M, K) · (K,) → (M,)`; matrix times column vector.
    MatrixVector,
    /// `(K,) · (K, N) → (N,)`; row vector times matrix.
    VectorMatrix,
    /// `(M, K) · (K, N) → (M, N)`; full matrix multiply.
    MatrixMatrix,
}

/// Prepared batched matrix multiply, including virtual vector axes.
///
/// Describes how to walk batch dimensions and contract the inner `K` axis
/// between left rows `(M, K)` and right columns `(K, N)`. One-dimensional
/// operands are promoted to `(1, K)` or `(K, 1)` without copying memory:
/// `left_was_vector` / `right_was_vector` record that promotion so
/// [`Self::output_shape`] can strip length-1 matrix axes from the result.
///
/// # Contraction geometry
///
/// ```text
///   batch… × M × K  @  batch… × K × N  →  batch… × M × N
///              └──── contraction_len ────┘
/// ```
///
/// Batch leading dimensions are independently broadcast; size-1 batch axes
/// contribute zero stride in the aligned batch stride vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatmulPlan {
    /// Broadcast shape of all leading batch axes.
    pub(crate) batch_shape: Vec<usize>,
    /// Batch-axis strides for the left operand after broadcasting.
    pub(crate) left_batch_strides: Vec<isize>,
    /// Batch-axis strides for the right operand after broadcasting.
    pub(crate) right_batch_strides: Vec<isize>,
    /// Row count `M` of the left matrix face (1 when left was a vector).
    pub(crate) rows: usize,
    /// `rows * columns`; size of one `(M, N)` output tile per batch element.
    pub(crate) matrix_len: usize,
    /// Inner dimension `K` summed during contraction.
    pub(crate) contraction_len: usize,
    /// Column count `N` of the right matrix face (1 when right was a vector).
    pub(crate) columns: usize,
    /// Stride between consecutive rows on the left matrix face.
    pub(crate) left_row_stride: isize,
    /// Stride along the contraction axis on the left matrix face.
    pub(crate) left_contraction_stride: isize,
    /// Stride along the contraction axis on the right matrix face.
    pub(crate) right_contraction_stride: isize,
    /// Stride between consecutive columns on the right matrix face.
    pub(crate) right_column_stride: isize,
    /// Left operand was 1-D and is treated as `(1, K)`.
    pub(crate) left_was_vector: bool,
    /// Right operand was 1-D and is treated as `(K, 1)`.
    pub(crate) right_was_vector: bool,
}

impl MatmulPlan {
    /// Build a plan from operand shapes, strides, and batch broadcasting.
    ///
    /// Reads the trailing `(M, K)` / `(K, N)` faces (or vector promotions),
    /// broadcasts leading batch shapes, and precomputes all strides needed
    /// by [`crate::linalg::kernels::contract`].
    ///
    /// # Arguments
    ///
    /// * `left` — left contraction operand
    /// * `right` — right contraction operand (inner axis must match)
    ///
    /// # Returns
    ///
    /// A fully populated [`MatmulPlan`].
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — incompatible batch broadcast or shape
    ///   product overflow
    pub(crate) fn new<L: Scalar, R: Scalar>(
        left: &Array<L>,
        right: &Array<R>,
    ) -> Result<Self> {
        let left_was_vector = left.ndim() == 1;
        let right_was_vector = right.ndim() == 1;
        let (rows, contraction_len, left_row_stride, left_contraction_stride) =
            if left_was_vector {
                (1, left.shape()[0], 0, left.strides()[0])
            } else {
                let rank = left.ndim();
                (
                    left.shape()[rank - 2],
                    left.shape()[rank - 1],
                    left.strides()[rank - 2],
                    left.strides()[rank - 1],
                )
            };
        let (
            _right_contraction_len,
            columns,
            right_contraction_stride,
            right_column_stride,
        ) = if right_was_vector {
            (right.shape()[0], 1, right.strides()[0], 0)
        } else {
            let rank = right.ndim();
            (
                right.shape()[rank - 2],
                right.shape()[rank - 1],
                right.strides()[rank - 2],
                right.strides()[rank - 1],
            )
        };

        let left_batch_rank = left.ndim().saturating_sub(2);
        let right_batch_rank = right.ndim().saturating_sub(2);
        let left_batch_shape = &left.shape()[..left_batch_rank];
        let right_batch_shape = &right.shape()[..right_batch_rank];
        let batch_shape = broadcast_shape(left_batch_shape, right_batch_shape)?;
        let left_batch_strides = align_batch_strides(
            left_batch_shape,
            &left.strides()[..left_batch_rank],
            &batch_shape,
        );
        let right_batch_strides = align_batch_strides(
            right_batch_shape,
            &right.strides()[..right_batch_rank],
            &batch_shape,
        );
        let matrix_len = checked_size_of_shape(&[rows, columns])?;

        Ok(Self {
            batch_shape,
            left_batch_strides,
            right_batch_strides,
            rows,
            matrix_len,
            contraction_len,
            columns,
            left_row_stride,
            left_contraction_stride,
            right_contraction_stride,
            right_column_stride,
            left_was_vector,
            right_was_vector,
        })
    }

    /// Output shape after stripping virtual vector dimensions.
    ///
    /// Batch axes are preserved. Matrix rows appear when the left operand was
    /// not a vector; columns appear when the right operand was not a vector.
    ///
    /// # Arguments
    ///
    /// * `self` - A validated [`MatmulPlan`] built from operand shapes.
    ///
    /// # Returns
    ///
    /// Shape of the contracted result array.
    pub(crate) fn output_shape(&self) -> Vec<usize> {
        let mut shape = self.batch_shape.clone();
        if !self.left_was_vector {
            shape.push(self.rows);
        }
        if !self.right_was_vector {
            shape.push(self.columns);
        }
        shape
    }
}

/// Classify `dot` operand ranks and build the shared contraction plan.
///
/// Maps `(left.ndim(), right.ndim())` to a [`DotKind`] for kernel dispatch,
/// then delegates geometry to [`MatmulPlan::new`]. Higher-rank operands are
/// treated as [`DotKind::MatrixMatrix`] with batched leading axes.
///
/// # Arguments
///
/// * `left` — left operand
/// * `right` — right operand
///
/// # Returns
///
/// The detected [`DotKind`] and a shared [`MatmulPlan`].
///
/// # Errors
///
/// Propagates errors from [`MatmulPlan::new`].
pub(crate) fn plan_dot<L: Scalar, R: Scalar>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<(DotKind, MatmulPlan)> {
    let kind = match (left.ndim(), right.ndim()) {
        (1, 1) => DotKind::VectorVector,
        (2, 1) => DotKind::MatrixVector,
        (1, 2) => DotKind::VectorMatrix,
        (2, 2) => DotKind::MatrixMatrix,
        _ => DotKind::MatrixMatrix,
    };
    Ok((kind, MatmulPlan::new(left, right)?))
}

/// Broadcast batch axes: size-1 source axes contribute zero stride.
///
/// Pads leading target axes with zero stride, then maps each trailing axis.
/// When the source length is 1 and the broadcast target length is not 1, the
/// stride is zero so the same element is reused.
///
/// # Arguments
///
/// * `source_shape` — batch shape of one operand before broadcast
/// * `source_strides` — batch strides aligned with `source_shape`
/// * `target_shape` — broadcast batch shape (same length or longer)
///
/// # Returns
///
/// Batch strides aligned with `target_shape`.
fn align_batch_strides(
    source_shape: &[usize],
    source_strides: &[isize],
    target_shape: &[usize],
) -> Vec<isize> {
    let leading = target_shape.len() - source_shape.len();
    target_shape
        .iter()
        .enumerate()
        .map(|(axis, &target_dim)| {
            if axis < leading {
                return 0;
            }
            let source_axis = axis - leading;
            if source_shape[source_axis] == 1 && target_dim != 1 {
                0
            } else {
                source_strides[source_axis]
            }
        })
        .collect()
}

/// Geometry for N-dimensional diagonal extraction and trace.
///
/// Holds the shape/strides of all axes **outside** the two diagonal axes,
/// plus byte offsets for stepping along one diagonal element at a time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiagonalPlan {
    /// Shape of axes other than the two diagonal axes.
    pub(crate) outer_shape: Vec<usize>,
    /// Strides for [`Self::outer_shape`] axes.
    pub(crate) outer_strides: Vec<isize>,
    /// Row/column start and length on the 2-D diagonal face.
    pub(crate) geometry: DiagonalGeometry,
    /// Buffer offset of the first diagonal element for one outer tile.
    pub(crate) diagonal_start_offset: isize,
    /// Added to the buffer index for each successive diagonal element.
    pub(crate) diagonal_stride: isize,
}

impl DiagonalPlan {
    /// Resolve axes, diagonal length, and all offsets for the kernels.
    ///
    /// Normalizes `axis1` / `axis2`, computes [`DiagonalGeometry`] for their
    /// face dimensions and `offset`, and derives linear memory offsets from
    /// the array strides.
    ///
    /// # Arguments
    ///
    /// * `array` — source array (at least 2-D when both axes differ)
    /// * `offset` — NumPy-style diagonal offset (0 = main diagonal)
    /// * `axis1` — first diagonal axis (may be negative)
    /// * `axis2` — second diagonal axis (may be negative)
    ///
    /// # Returns
    ///
    /// A [`DiagonalPlan`] ready for diagonal gather or trace kernels.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — diagonal offset arithmetic overflows
    ///   `isize`
    pub(crate) fn new<T: Scalar>(
        array: &Array<T>,
        offset: isize,
        axis1: isize,
        axis2: isize,
    ) -> Result<Self> {
        let axis1 = normalize_axis(axis1, array.ndim());
        let axis2 = normalize_axis(axis2, array.ndim());

        let geometry = diagonal_geometry(
            array.shape()[axis1],
            array.shape()[axis2],
            offset,
        );
        let mut outer_shape = Vec::with_capacity(array.ndim() - 2);
        let mut outer_strides = Vec::with_capacity(array.ndim() - 2);
        for axis in 0..array.ndim() {
            if axis != axis1 && axis != axis2 {
                outer_shape.push(array.shape()[axis]);
                outer_strides.push(array.strides()[axis]);
            }
        }
        let row_offset = isize::try_from(geometry.row_start)
            .ok()
            .and_then(|row| row.checked_mul(array.strides()[axis1]))
            .ok_or_else(|| {
                Error::InvalidArgument(
                    "diagonal row offset overflows isize".into(),
                )
            })?;
        let column_offset = isize::try_from(geometry.column_start)
            .ok()
            .and_then(|column| column.checked_mul(array.strides()[axis2]))
            .ok_or_else(|| {
                Error::InvalidArgument(
                    "diagonal column offset overflows isize".into(),
                )
            })?;
        let diagonal_start_offset =
            row_offset.checked_add(column_offset).ok_or_else(|| {
                Error::InvalidArgument(
                    "diagonal start offset overflows isize".into(),
                )
            })?;
        // Moving one step along the diagonal advances both axes.
        let diagonal_stride = array.strides()[axis1]
            .checked_add(array.strides()[axis2])
            .ok_or_else(|| {
                Error::InvalidArgument("diagonal stride overflows isize".into())
            })?;

        Ok(Self {
            outer_shape,
            outer_strides,
            geometry,
            diagonal_start_offset,
            diagonal_stride,
        })
    }

    /// Shape produced by [`crate::linalg::diagonal`].
    ///
    /// Appends the diagonal length as the last axis after all outer axes.
    ///
    /// # Arguments
    ///
    /// * `self` - A validated [`DiagonalPlan`] for the source array layout.
    ///
    /// # Returns
    ///
    /// Output shape `[outer…, diagonal_len]`.
    pub(crate) fn diagonal_output_shape(&self) -> Vec<usize> {
        let mut shape = self.outer_shape.clone();
        shape.push(self.geometry.len);
        shape
    }
}
