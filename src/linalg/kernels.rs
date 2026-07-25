//! Contiguous and strided contraction kernels.

use crate::array::Array;
use crate::dtype::{CastTo, Scalar};
use crate::error::Result;
use crate::linalg::geometry::{DiagonalPlan, MatmulPlan};
use crate::linalg::traits::ContractElement;
use crate::reduction::SumReduce;
use crate::shape::checked_size_of_shape;
use crate::traversal::RunPlan;

/// Execute a prepared contraction into a C-order output.
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

    debug_assert_eq!(output.len(), output_len);
    Array::from_vec(output, &output_shape)
}

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
    let matrix_len = plan.rows * plan.columns;
    let output_start = output.len();
    output.resize(output_start + matrix_len, Out::zero());

    // IKJ keeps a unit-stride right row and the output row hot.
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

    // General strided path computes one output element at a time.
    for row in 0..plan.rows {
        let left_row = left_base as isize + row as isize * plan.left_row_stride;
        for column in 0..plan.columns {
            let right_column = right_base as isize
                + column as isize * plan.right_column_stride;

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

/// Flatten both operands logically and compute a conjugating vector product.
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

/// Gather an N-dimensional diagonal into a C-order copy.
pub(crate) fn gather_diagonal<T: Scalar>(
    array: &Array<T>,
    plan: &DiagonalPlan,
) -> Result<Array<T>> {
    let output_shape = plan.diagonal_output_shape();
    let output_len = checked_size_of_shape(&output_shape)?;
    let mut output = Vec::with_capacity(output_len);
    let outer_plan =
        RunPlan::<1>::new(&plan.outer_shape, [&plan.outer_strides]);
    outer_plan.for_each_element([array.offset() as isize], |[outer_base]| {
        let mut position = outer_base as isize + plan.diagonal_start_offset;
        for _ in 0..plan.geometry.len {
            output.push(array.data[position as usize]);
            position += plan.diagonal_stride;
        }
    });
    Array::from_vec(output, &output_shape)
}

/// Sum each N-dimensional diagonal into the shape of the remaining axes.
pub(crate) fn trace_diagonal<T: SumReduce>(
    array: &Array<T>,
    plan: &DiagonalPlan,
) -> Result<Array<T::Acc>> {
    let output_len = checked_size_of_shape(&plan.outer_shape)?;
    let mut output = Vec::with_capacity(output_len);
    let outer_plan =
        RunPlan::<1>::new(&plan.outer_shape, [&plan.outer_strides]);
    outer_plan.for_each_element([array.offset() as isize], |[outer_base]| {
        let mut accumulator = T::identity();
        let mut position = outer_base as isize + plan.diagonal_start_offset;
        for _ in 0..plan.geometry.len {
            accumulator =
                T::accumulate(accumulator, array.data[position as usize]);
            position += plan.diagonal_stride;
        }
        output.push(accumulator);
    });
    Array::from_vec(output, &plan.outer_shape)
}
