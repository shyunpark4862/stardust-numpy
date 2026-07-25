//! Execution kernels for matrix contraction and diagonal operations.
//!
//! `contract` walks batch axes with `RunPlan`, then multiplies matrix tiles.
//! When the right operand has a unit column stride, an IKJ loop keeps the
//! output row and right row in registers. Otherwise a general path handles
//! arbitrary strides, with an inner fast path when both contraction axes are
//! contiguous.

use crate::array::Array;
use crate::dtype::{CastTo, Scalar};
use crate::error::Result;
use crate::linalg::geometry::{DiagonalPlan, MatmulPlan};
use crate::linalg::traits::ContractElement;
use crate::reduction::SumReduce;
use crate::shape::{checked_allocation_len, checked_size_of_shape};
use crate::traversal::RunPlan;

/// Fill a C-order output buffer from a prepared [`MatmulPlan`].
///
/// Iterates every batch tile via [`RunPlan`], then contracts one `(M, N)`
/// matrix per tile into the output vector. Output dtype is the promoted
/// [`ContractElement`] type.
///
/// # Arguments
///
/// * `left` — left contraction operand
/// * `right` — right contraction operand
/// * `plan` — geometry from [`MatmulPlan::new`]
///
/// # Returns
///
/// New C-order array with shape [`MatmulPlan::output_shape`].
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — output size or allocation exceeds limits
pub(crate) fn contract<L, R, Out>(
    left: &Array<L>,
    right: &Array<R>,
    plan: &MatmulPlan,
) -> Result<Array<Out>>
where
    L: Scalar + CastTo<Out>,
    R: Scalar + CastTo<Out>,
    Out: ContractElement,
{
    let output_shape = plan.output_shape();
    let output_len = checked_size_of_shape(&output_shape)?;
    checked_allocation_len::<Out>(output_len)?;
    let mut output = Vec::with_capacity(output_len);
    let batch_plan = RunPlan::<2>::new(
        &plan.batch_shape,
        [&plan.left_batch_strides, &plan.right_batch_strides],
    );

    batch_plan.for_each_element(
        [left.offset() as isize, right.offset() as isize],
        |[left_base, right_base]| {
            contract_matrix(
                left,
                right,
                plan,
                left_base,
                right_base,
                &mut output,
            );
        },
    );

    Array::from_vec(output, &output_shape)
}

/// Contract one `(M, N)` output tile into `output`.
///
/// Appends `plan.matrix_len` elements and fills them in place. Selects an
/// IKJ path when `right_column_stride == 1`, a vectorized inner dot when
/// both contraction strides are unit, or a fully general scalar loop.
///
/// # Arguments
///
/// * `left`, `right` — operand buffers
/// * `plan` — matrix face geometry for this batch element
/// * `left_base`, `right_base` — buffer offsets for the current batch tile
/// * `output` — growing output vector receiving the tile
fn contract_matrix<L, R, Out>(
    left: &Array<L>,
    right: &Array<R>,
    plan: &MatmulPlan,
    left_base: usize,
    right_base: usize,
    output: &mut Vec<Out>,
) where
    L: Scalar + CastTo<Out>,
    R: Scalar + CastTo<Out>,
    Out: ContractElement,
{
    let output_start = output.len();
    output.resize(output_start + plan.matrix_len, Out::zero());

    // IKJ: one left scalar broadcasts across a contiguous right row.
    if plan.right_column_stride == 1 {
        for row in 0..plan.rows {
            let left_row =
                left_base as isize + row as isize * plan.left_row_stride;
            let output_row = output_start + row * plan.columns;
            for inner in 0..plan.contraction_len {
                let left_position =
                    left_row + inner as isize * plan.left_contraction_stride;
                let right_row = right_base as isize
                    + inner as isize * plan.right_contraction_stride;
                let left_value: Out =
                    left.data[left_position as usize].cast_to();
                let right_start = right_row as usize;
                let right_row =
                    &right.data[right_start..right_start + plan.columns];
                let output_row =
                    &mut output[output_row..output_row + plan.columns];
                let mut output_chunks = output_row.chunks_exact_mut(8);
                let mut right_chunks = right_row.chunks_exact(8);
                for (output_chunk, right_chunk) in
                    output_chunks.by_ref().zip(right_chunks.by_ref())
                {
                    for lane in 0..8 {
                        output_chunk[lane] = Out::multiply_add(
                            output_chunk[lane],
                            left_value,
                            right_chunk[lane].cast_to(),
                        );
                    }
                }
                for (slot, &right_value) in output_chunks
                    .into_remainder()
                    .iter_mut()
                    .zip(right_chunks.remainder())
                {
                    *slot = Out::multiply_add(
                        *slot,
                        left_value,
                        right_value.cast_to(),
                    );
                }
            }
        }
        return;
    }

    // General layout: one output element per (row, column) tile.
    for row in 0..plan.rows {
        let left_row = left_base as isize + row as isize * plan.left_row_stride;
        for column in 0..plan.columns {
            let right_column = right_base as isize
                + column as isize * plan.right_column_stride;

            // Both contraction axes contiguous: vectorized inner dot.
            if plan.left_contraction_stride == 1
                && plan.right_contraction_stride == 1
            {
                let left_start = left_row as usize;
                let right_start = right_column as usize;
                let left_values =
                    &left.data[left_start..left_start + plan.contraction_len];
                let right_values = &right.data
                    [right_start..right_start + plan.contraction_len];
                let mut partials = [Out::zero(); 8];
                let paired_len = plan.contraction_len / 8 * 8;

                for (left_chunk, right_chunk) in left_values[..paired_len]
                    .chunks_exact(8)
                    .zip(right_values[..paired_len].chunks_exact(8))
                {
                    for lane in 0..8 {
                        partials[lane] = Out::multiply_add(
                            partials[lane],
                            left_chunk[lane].cast_to(),
                            right_chunk[lane].cast_to(),
                        );
                    }
                }

                let mut accumulator = Out::zero();
                for partial in partials {
                    accumulator = Out::add(accumulator, partial);
                }
                for (&left_value, &right_value) in left_values[paired_len..]
                    .iter()
                    .zip(&right_values[paired_len..])
                {
                    accumulator = Out::multiply_add(
                        accumulator,
                        left_value.cast_to(),
                        right_value.cast_to(),
                    );
                }
                output[output_start + row * plan.columns + column] =
                    accumulator;
                continue;
            }

            // Fully strided inner product: scalar loop over K.
            let mut accumulator = Out::zero();
            for inner in 0..plan.contraction_len {
                let left_position =
                    left_row + inner as isize * plan.left_contraction_stride;
                let right_position = right_column
                    + inner as isize * plan.right_contraction_stride;
                accumulator = Out::multiply_add(
                    accumulator,
                    left.data[left_position as usize].cast_to(),
                    right.data[right_position as usize].cast_to(),
                );
            }
            output[output_start + row * plan.columns + column] = accumulator;
        }
    }
}

/// C-order flatten of both operands, optionally conjugating the left side.
///
/// Materializes both vectors to C-order, then reduces with chunked
/// multiply-add. Used for `dot` / `vdot` after rank classification.
///
/// # Arguments
///
/// * `left` — left vector operand
/// * `right` — right vector operand (same logical length as `left`)
/// * `conjugate_left` — apply [`ContractElement::conjugate`] to left values
///
/// # Returns
///
/// 0-D array holding the scalar inner product.
///
/// # Errors
///
/// Never fails for valid same-length vectors (no allocation size check beyond
/// the scalar result).
pub(crate) fn vector_dot<L, R, Out>(
    left: &Array<L>,
    right: &Array<R>,
    conjugate_left: bool,
) -> Result<Array<Out>>
where
    L: Scalar + CastTo<Out>,
    R: Scalar + CastTo<Out>,
    Out: ContractElement,
{
    let left_values = left.to_c_order_cow();
    let right_values = right.to_c_order_cow();
    let paired_len = left_values.len() / 8 * 8;
    let mut partials = [Out::zero(); 8];

    for (left_chunk, right_chunk) in left_values[..paired_len]
        .chunks_exact(8)
        .zip(right_values[..paired_len].chunks_exact(8))
    {
        for lane in 0..8 {
            let mut left_value: Out = left_chunk[lane].cast_to();
            if conjugate_left {
                left_value = left_value.conjugate();
            }
            partials[lane] = Out::multiply_add(
                partials[lane],
                left_value,
                right_chunk[lane].cast_to(),
            );
        }
    }

    let mut accumulator = Out::zero();
    for partial in partials {
        accumulator = Out::add(accumulator, partial);
    }
    for (&left_value, &right_value) in left_values[paired_len..]
        .iter()
        .zip(&right_values[paired_len..])
    {
        let mut left_value: Out = left_value.cast_to();
        if conjugate_left {
            left_value = left_value.conjugate();
        }
        accumulator =
            Out::multiply_add(accumulator, left_value, right_value.cast_to());
    }
    Array::from_vec(vec![accumulator], &[])
}

/// Copy diagonal elements into a new C-order array.
///
/// Walks outer axes with [`RunPlan`], then reads `geometry.len` elements along
/// [`DiagonalPlan::diagonal_stride`] starting at
/// [`DiagonalPlan::diagonal_start_offset`].
///
/// # Arguments
///
/// * `array` — source array
/// * `plan` — geometry from [`DiagonalPlan::new`]
///
/// # Returns
///
/// C-order array with shape [`DiagonalPlan::diagonal_output_shape`].
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — output allocation exceeds limits
pub(crate) fn gather_diagonal<T: Scalar>(
    array: &Array<T>,
    plan: &DiagonalPlan,
) -> Result<Array<T>> {
    let output_shape = plan.diagonal_output_shape();
    let output_len = checked_size_of_shape(&output_shape)?;
    checked_allocation_len::<T>(output_len)?;
    let mut output = Vec::with_capacity(output_len);
    let outer_plan =
        RunPlan::<1>::new(&plan.outer_shape, [&plan.outer_strides]);
    outer_plan.for_each_element([array.offset() as isize], |[outer_base]| {
        let mut position = outer_base as isize + plan.diagonal_start_offset;
        if plan.geometry.len == 0 {
            return;
        }
        output.push(array.data[position as usize]);
        // Step along the diagonal: sum of row and column strides.
        for _ in 1..plan.geometry.len {
            position += plan.diagonal_stride;
            output.push(array.data[position as usize]);
        }
    });
    Array::from_vec(output, &output_shape)
}

/// Reduce each diagonal with the element type's sum fold.
///
/// Like [`gather_diagonal`], but accumulates with [`SumReduce::accumulate`]
/// instead of copying each element. Output rank equals `plan.outer_shape`.
///
/// # Arguments
///
/// * `array` — source array
/// * `plan` — geometry from [`DiagonalPlan::new`]
///
/// # Returns
///
/// C-order array with shape `plan.outer_shape` and accumulator dtype
/// `T::Acc`.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — output allocation exceeds limits
pub(crate) fn trace_diagonal<T: SumReduce>(
    array: &Array<T>,
    plan: &DiagonalPlan,
) -> Result<Array<T::Acc>> {
    let output_len = checked_size_of_shape(&plan.outer_shape)?;
    checked_allocation_len::<T::Acc>(output_len)?;
    let mut output = Vec::with_capacity(output_len);
    let outer_plan =
        RunPlan::<1>::new(&plan.outer_shape, [&plan.outer_strides]);
    outer_plan.for_each_element([array.offset() as isize], |[outer_base]| {
        let mut accumulator = T::identity();
        let mut position = outer_base as isize + plan.diagonal_start_offset;
        if plan.geometry.len > 0 {
            accumulator =
                T::accumulate(accumulator, array.data[position as usize]);
            for _ in 1..plan.geometry.len {
                position += plan.diagonal_stride;
                accumulator =
                    T::accumulate(accumulator, array.data[position as usize]);
            }
        }
        output.push(accumulator);
    });
    Array::from_vec(output, &plan.outer_shape)
}
