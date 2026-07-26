//! User-facing linear algebra operations.
//!
//! Each function validates operands, builds a geometry plan, and delegates
//! to the contraction or diagonal kernels. Dtype promotion follows the
//! shared `Promote` rules used elsewhere in the crate.

use crate::array::Array;
use crate::dtype::{CastTo, Promote, Scalar};
use crate::error::Result;
use crate::linalg::geometry::{plan_dot, DiagonalPlan, MatmulPlan};
use crate::linalg::kernels::{
    contract, diagonal_view, trace_diagonal, vector_dot,
};
use crate::linalg::traits::ContractElement;
use crate::reduction::SumReduce;
use crate::shape::{checked_allocation_len, checked_size_of_shape};

/// Inner product for 1-D and 2-D operands.
///
/// Supports vector-vector, matrix-vector, vector-matrix, and matrix-matrix
/// combinations. Unlike [`vdot`], complex values are not conjugated.
///
/// # Shape rules
///
/// * `(n,) · (n,)` → `(,)`
/// * `(m, n) · (n,)` → `(m,)`
/// * `(m, n) · (n, p)` → `(m, p)`
///
/// Higher-dimensional inputs are not supported; use [`matmul`] for batched
/// multiplication.
///
/// # Arguments
///
/// * `left` - Left operand array.
/// * `right` - Right operand array (last axis is the contraction axis).
///
/// # Returns
///
/// A promoted output array whose shape follows the rules above.
///
/// # Errors
///
/// Returns an error when inner dimensions are incompatible or shapes are
/// not 1-D or 2-D.
///
/// # Examples
///
/// ```
/// use sdnp::{dot, Array};
///
/// let a = Array::from_vec(vec![1_i64, 2], &[2]).unwrap();
/// let b = Array::from_vec(vec![3_i64, 4], &[2]).unwrap();
/// let c = dot(&a, &b).unwrap();
/// assert_eq!(c.item().unwrap(), 11);
/// ```
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

/// NumPy-style batched matrix multiplication (`@` operator semantics).
///
/// One-dimensional operands are promoted to row or column matrices; the
/// corresponding size-one output axis is removed afterward. Leading batch
/// axes are broadcast. Scalar operands are rejected.
///
/// # Shape rules
///
/// For the last two axes of each operand (after 1-D promotion):
///
/// * `(…, m, k) @ (…, k, n)` → `(…, m, n)`
/// * `(…, m, k) @ (…, k)` → `(…, m)` (1-D right promoted)
/// * `(…, k) @ (…, k, n)` → `(…, n)` (1-D left promoted)
///
/// Unlike [`dot`], batch dimensions are supported and broadcast.
///
/// # Arguments
///
/// * `left` - Left operand array.
/// * `right` - Right operand array.
///
/// # Returns
///
/// A promoted output array with batched matrix shape.
///
/// # Errors
///
/// Returns an error when inner dimensions mismatch, batch shapes cannot
/// be broadcast, or either operand is 0-D.
///
/// # Examples
///
/// ```
/// use sdnp::{matmul, Array};
///
/// let a = Array::from_vec(vec![1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// let b = Array::from_vec(vec![5_i64, 6, 7, 8], &[2, 2]).unwrap();
/// let c = matmul(&a, &b).unwrap();
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
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

// Matmul assumes a unit-stride last axis on the right operand; copy if not.
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

/// Flatten both arrays in C order and return their dot product.
///
/// The left operand is conjugated after dtype promotion, matching NumPy
/// `vdot`. For real dtypes conjugation is a no-op.
///
/// # Arguments
///
/// * `left` - Left operand (any shape; flattened in C order).
/// * `right` - Right operand (any shape; flattened in C order).
///
/// # Returns
///
/// A 0-D promoted output array containing the inner product.
///
/// # Errors
///
/// Returns an error when flattened lengths differ.
///
/// # Examples
///
/// ```
/// use sdnp::{vdot, Array, Complex64};
///
/// let a = Array::from_vec(
///     vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
///     &[2],
/// )
/// .unwrap();
/// let b = Array::from_vec(
///     vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
///     &[2],
/// )
/// .unwrap();
/// let c = vdot(&a, &b).unwrap();
/// assert_eq!(c.item().unwrap().re, 3.0);
/// ```
pub fn vdot<L, R>(
    left: &Array<L>,
    right: &Array<R>,
) -> Result<Array<<L as Promote<R>>::Output>>
where
    L: Promote<R> + CastTo<<L as Promote<R>>::Output>,
    R: Scalar + CastTo<<L as Promote<R>>::Output>,
    <L as Promote<R>>::Output: ContractElement,
{
    vector_dot(left, right, true)
}

/// Outer product after logical C-order flattening of both operands.
///
/// Each element of the flattened left vector pairs with every element of
/// the flattened right vector.
///
/// # Arguments
///
/// * `left` - Left operand (any shape).
/// * `right` - Right operand (any shape).
///
/// # Returns
///
/// A 2-D array with shape `(left.size(), right.size())`.
///
/// # Errors
///
/// Returns an error when the output allocation would overflow.
///
/// # Examples
///
/// ```
/// use sdnp::{outer, Array};
///
/// let a = Array::from_vec(vec![1_i64, 2], &[2]).unwrap();
/// let b = Array::from_vec(vec![10_i64, 20, 30], &[3]).unwrap();
/// let c = outer(&a, &b).unwrap();
/// assert_eq!(c.shape(), &[2, 3]);
/// ```
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
    checked_allocation_len::<<L as Promote<R>>::Output>(output_len)?;
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

/// View diagonals along two axes as the trailing output dimension.
///
/// Axes may be negative. Remaining axes keep their original order; the
/// diagonal length is appended last. This returns a zero-copy strided view
/// over the source buffer rather than gathering elements into a new vector.
/// Writes to the view detach via copy-on-write.
///
/// # Arguments
///
/// * `array` - Input array (at least 2-D).
/// * `offset` - Diagonal offset from the main diagonal.
/// * `axis1` - First axis defining the 2-D plane.
/// * `axis2` - Second axis defining the 2-D plane.
///
/// # Returns
///
/// A zero-copy strided view whose trailing dimension is the diagonal length.
///
/// # Errors
///
/// Returns an error when axes are invalid or the offset is out of range.
///
/// # Examples
///
/// ```
/// use sdnp::{diagonal, Array};
///
/// let a = Array::from_vec(vec![1_i64, 0, 0, 2], &[2, 2]).unwrap();
/// let d = diagonal(&a, 0, 0, 1).unwrap();
/// assert_eq!(d.to_vec(), vec![1, 2]);
/// ```
pub fn diagonal<T: Scalar>(
    array: &Array<T>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<Array<T>> {
    let plan = DiagonalPlan::new(array, offset, axis1, axis2)?;
    diagonal_view(array, &plan)
}

/// Sum elements along diagonals defined by two axes.
///
/// A 2-D input yields a 0-D array. Higher rank inputs drop `axis1` and
/// `axis2`, summing each diagonal into the remaining shape.
///
/// # Arguments
///
/// * `array` - Input array (at least 2-D).
/// * `offset` - Diagonal offset from the main diagonal.
/// * `axis1` - First axis defining the 2-D plane.
/// * `axis2` - Second axis defining the 2-D plane.
///
/// # Returns
///
/// An array of diagonal sums with dtype `T::Acc`.
///
/// # Errors
///
/// Returns an error when axes are invalid or the offset is out of range.
///
/// # Examples
///
/// ```
/// use sdnp::{trace, Array};
///
/// let a = Array::from_vec(vec![1_i64, 0, 0, 2], &[2, 2]).unwrap();
/// let t = trace(&a, 0, 0, 1).unwrap();
/// assert_eq!(t.item().unwrap(), 3);
/// ```
pub fn trace<T: SumReduce>(
    array: &Array<T>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<Array<T::Acc>> {
    let plan = DiagonalPlan::new(array, offset, axis1, axis2)?;
    trace_diagonal(array, &plan)
}
