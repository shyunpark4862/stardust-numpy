//! Shape, rank-promotion, and contraction geometry for linear algebra.

use crate::array::Array;
use crate::axis::normalize_axis;
use crate::broadcast::broadcast_shape;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::linalg::diagonal_geometry::{diagonal_geometry, DiagonalGeometry};

/// Supported rank combinations for `dot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DotKind {
    /// `(K) dot (K) -> ()`.
    VectorVector,
    /// `(M, K) dot (K) -> (M)`.
    MatrixVector,
    /// `(K) dot (K, N) -> (N)`.
    VectorMatrix,
    /// `(M, K) dot (K, N) -> (M, N)`.
    MatrixMatrix,
}

/// Prepared matrix contraction, including virtual vector axes and batch strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatmulPlan {
    pub(crate) batch_shape: Vec<usize>,
    pub(crate) left_batch_strides: Vec<isize>,
    pub(crate) right_batch_strides: Vec<isize>,
    pub(crate) rows: usize,
    pub(crate) contraction_len: usize,
    pub(crate) columns: usize,
    pub(crate) left_row_stride: isize,
    pub(crate) left_contraction_stride: isize,
    pub(crate) right_contraction_stride: isize,
    pub(crate) right_column_stride: isize,
    pub(crate) left_was_vector: bool,
    pub(crate) right_was_vector: bool,
}

impl MatmulPlan {
    /// Build a batched matrix multiplication plan.
    pub(crate) fn new<L: Scalar, R: Scalar>(
        left: &Array<L>,
        right: &Array<R>,
    ) -> Result<Self> {
        if left.ndim() == 0 || right.ndim() == 0 {
            debug_assert!(false, "matmul does not support 0-D operands");
        }

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
            right_contraction_len,
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

        debug_assert_eq!(
            contraction_len, right_contraction_len,
            "matmul inner dimensions differ"
        );

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

        Ok(Self {
            batch_shape,
            left_batch_strides,
            right_batch_strides,
            rows,
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

    /// Result shape after removing virtual vector axes.
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

/// Validate `dot` ranks and construct its contraction plan.
pub(crate) fn plan_dot<L: Scalar, R: Scalar>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<(DotKind, MatmulPlan)> {
    let kind = match (left.ndim(), right.ndim()) {
        (1, 1) => DotKind::VectorVector,
        (2, 1) => DotKind::MatrixVector,
        (1, 2) => DotKind::VectorMatrix,
        (2, 2) => DotKind::MatrixMatrix,
        _ => {
            debug_assert!(false, "dot supports only 1-D or 2-D operands");
            DotKind::MatrixMatrix
        }
    };
    Ok((kind, MatmulPlan::new(left, right)?))
}

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

/// Prepared geometry for N-dimensional diagonal extraction and trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiagonalPlan {
    pub(crate) outer_shape: Vec<usize>,
    pub(crate) outer_strides: Vec<isize>,
    pub(crate) geometry: DiagonalGeometry,
    pub(crate) diagonal_start_offset: isize,
    pub(crate) diagonal_stride: isize,
}

impl DiagonalPlan {
    /// Validate axes and resolve all strides needed by diagonal kernels.
    pub(crate) fn new<T: Scalar>(
        array: &Array<T>,
        offset: isize,
        axis1: isize,
        axis2: isize,
    ) -> Result<Self> {
        let axis1 = normalize_axis(axis1, array.ndim());
        let axis2 = normalize_axis(axis2, array.ndim());
        debug_assert_ne!(axis1, axis2, "axis1 and axis2 must differ");

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
        let diagonal_start_offset = geometry.row_start as isize
            * array.strides()[axis1]
            + geometry.column_start as isize * array.strides()[axis2];

        Ok(Self {
            outer_shape,
            outer_strides,
            geometry,
            diagonal_start_offset,
            diagonal_stride: array.strides()[axis1] + array.strides()[axis2],
        })
    }

    /// Shape returned by `diagonal`.
    pub(crate) fn diagonal_output_shape(&self) -> Vec<usize> {
        let mut shape = self.outer_shape.clone();
        shape.push(self.geometry.len);
        shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_plan_broadcasts_and_removes_vector_axes() {
        let left = Array::from_vec(vec![0_i64; 12], &[2, 2, 3]).unwrap();
        let right = Array::from_vec(vec![0_i64; 3], &[3]).unwrap();
        let plan = MatmulPlan::new(&left, &right).unwrap();
        assert_eq!(plan.batch_shape, vec![2]);
        assert_eq!(plan.output_shape(), vec![2, 2]);
        assert!(plan.right_was_vector);
    }

    #[test]
    fn diagonal_plan_preserves_remaining_axis_order() {
        let array = Array::from_vec((0_i64..24).collect(), &[2, 3, 4]).unwrap();
        let plan = DiagonalPlan::new(&array, 1, 0, 2).unwrap();
        assert_eq!(plan.outer_shape, vec![3]);
        assert_eq!(plan.diagonal_output_shape(), vec![3, 2]);
    }
}
