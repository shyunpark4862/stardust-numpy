//! Public linear algebra operation entry points.

use crate::array::Array;
use crate::dtype::{CastTo, Promote, Scalar};
use crate::error::{Error, Result};
use crate::linalg::geometry::{plan_dot, DiagonalPlan, MatmulPlan};
use crate::linalg::kernels::{
    contract, gather_diagonal, trace_diagonal, vector_dot,
};
use crate::linalg::traits::ContractElement;
use crate::reduction::SumReduce;
use crate::shape::checked_size_of_shape;

/// Compute a dot product for one- and two-dimensional operands.
///
/// Supported combinations are vector-vector, matrix-vector, vector-matrix,
/// and matrix-matrix. Unlike [`vdot`], complex values are not conjugated.
pub fn dot<L, R>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<Array<<L as Promote<R>>::Output>>
where
    L: Promote<R> + CastTo<<L as Promote<R>>::Output>,
    R: Scalar + CastTo<<L as Promote<R>>::Output>,
    <L as Promote<R>>::Output: ContractElement,
{
    let right = prepare_right_for_contraction(right);
    let (_, plan) = plan_dot(left, &right)?;
    contract(left, &right, &plan)
}

/// Compute a NumPy-style batched matrix product.
///
/// One-dimensional operands are promoted to virtual row/column matrices and
/// the corresponding size-one output axis is removed. Leading batch axes are
/// broadcast; scalar operands are rejected.
pub fn matmul<L, R>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<Array<<L as Promote<R>>::Output>>
where
    L: Promote<R> + CastTo<<L as Promote<R>>::Output>,
    R: Scalar + CastTo<<L as Promote<R>>::Output>,
    <L as Promote<R>>::Output: ContractElement,
{
    let right = prepare_right_for_contraction(right);
    let plan = MatmulPlan::new(left, &right)?;
    contract(left, &right, &plan)
}

fn prepare_right_for_contraction<R: Scalar>(right: &Array<R>) -> Array<R> {
    let ndim = right.ndim();
    if ndim >= 2 {
        let last_axis = ndim - 1;
        if right.shape()[last_axis] > 1 && right.strides()[last_axis] != 1 {
            return right.copy();
        }
    }
    right.clone()
}

/// Flatten two arrays in logical C-order and return their vector dot product.
///
/// The first operand is conjugated after dtype promotion.
pub fn vdot<L, R>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<Array<<L as Promote<R>>::Output>>
where
    L: Promote<R> + CastTo<<L as Promote<R>>::Output>,
    R: Scalar + CastTo<<L as Promote<R>>::Output>,
    <L as Promote<R>>::Output: ContractElement,
{
    if left.size() != right.size() {
        return Err(Error::InvalidArgument(format!(
            "vdot requires equal flattened sizes, got {} and {}",
            left.size(),
            right.size()
        )));
    }
    vector_dot(left, right, true)
}

/// Return the outer product of two arrays after logical C-order flattening.
pub fn outer<L, R>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<Array<<L as Promote<R>>::Output>>
where
    L: Promote<R> + CastTo<<L as Promote<R>>::Output>,
    R: Scalar + CastTo<<L as Promote<R>>::Output>,
    <L as Promote<R>>::Output: ContractElement,
{
    type Output<L, R> = <L as Promote<R>>::Output;

    let output_shape = [left.size(), right.size()];
    let output_len = checked_size_of_shape(&output_shape)?;
    let left_values = left.to_c_order_cow();
    let right_values = right.to_c_order_cow();
    let mut output = Vec::with_capacity(output_len);
    for &left_value in left_values.iter() {
        let left_value: Output<L, R> = left_value.cast_to();
        let mut chunks = right_values.chunks_exact(8);
        for chunk in &mut chunks {
            for &right_value in chunk {
                output.push(Output::<L, R>::multiply_add(
                    Output::<L, R>::zero(),
                    left_value,
                    right_value.cast_to(),
                ));
            }
        }
        for &right_value in chunks.remainder() {
            output.push(Output::<L, R>::multiply_add(
                Output::<L, R>::zero(),
                left_value,
                right_value.cast_to(),
            ));
        }
    }
    Array::from_vec(output, &output_shape)
}

/// Copy the selected diagonals to the final axis.
///
/// Axes may be negative. The output shape is all remaining axes in their
/// original order followed by the selected diagonal length.
pub fn diagonal<T: Scalar>(
    array: &Array<T>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<Array<T>> {
    let plan = DiagonalPlan::new(array, offset, axis1, axis2)?;
    gather_diagonal(array, &plan)
}

/// Sum along selected diagonals.
///
/// A two-dimensional input returns a 0-D array. Higher-dimensional inputs
/// return the shape formed by removing `axis1` and `axis2`.
pub fn trace<T: SumReduce>(
    array: &Array<T>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<Array<T::Acc>> {
    let plan = DiagonalPlan::new(array, offset, axis1, axis2)?;
    trace_diagonal(array, &plan)
}
