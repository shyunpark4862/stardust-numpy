//! Same-type and cross-type element-wise map kernels.
//!
//! Fully-contiguous operands use a flat zip; otherwise operand layouts are
//! coalesced into [`crate::run::RunPlan`] and walked as fixed-stride runs.

use crate::array::Array;
use crate::broadcast::broadcast_shape;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::run::{collect_binary, collect_unary, try_collect_binary, RunPlan};
use crate::shape::size_of_shape;

/// Apply `f` to every element of `a` (C-order), producing a new contiguous array.
pub fn map_unary<A, Out, F>(a: &Array<A>, f: F) -> Result<Array<Out>>
where
    A: Scalar,
    Out: Scalar,
    F: FnMut(A) -> Out,
{
    if let Some(xs) = a.as_c_contiguous_slice() {
        let out = xs.iter().copied().map(f).collect();
        return Array::from_vec(out, a.shape());
    }

    let plan = RunPlan::new(a.shape(), [a.strides()]);
    let out = collect_unary(&plan, &a.data, a.offset(), f);
    Array::from_vec(out, a.shape())
}

/// Broadcast `a` and `b`, then apply infallible `f` element-wise.
pub fn map_binary<A, B, Out, F>(
    a: &Array<A>,
    b: &Array<B>,
    mut f: F,
) -> Result<Array<Out>>
where
    A: Scalar,
    B: Scalar,
    Out: Scalar,
    F: FnMut(A, B) -> Out,
{
    let aligned = align_binary(a, b)?;
    let left = aligned.left.as_ref().unwrap_or(a);
    let right = aligned.right.as_ref().unwrap_or(b);
    let shape = &aligned.shape;

    if let (Some(xs), Some(ys)) =
        (left.as_c_contiguous_slice(), right.as_c_contiguous_slice())
    {
        let out = xs
            .iter()
            .copied()
            .zip(ys.iter().copied())
            .map(|(x, y)| f(x, y))
            .collect();
        return Array::from_vec(out, shape);
    }

    let plan = RunPlan::new(shape, [left.strides(), right.strides()]);
    let out = collect_binary(
        &plan,
        &left.data,
        &right.data,
        [left.offset(), right.offset()],
        f,
    );
    Array::from_vec(out, shape)
}

/// Broadcast `a` and `b`, then apply fallible `f` element-wise.
pub fn try_map_binary<A, B, Out, F>(
    a: &Array<A>,
    b: &Array<B>,
    mut f: F,
) -> Result<Array<Out>>
where
    A: Scalar,
    B: Scalar,
    Out: Scalar,
    F: FnMut(A, B) -> Result<Out>,
{
    let aligned = align_binary(a, b)?;
    let left = aligned.left.as_ref().unwrap_or(a);
    let right = aligned.right.as_ref().unwrap_or(b);
    let shape = &aligned.shape;

    if let (Some(xs), Some(ys)) =
        (left.as_c_contiguous_slice(), right.as_c_contiguous_slice())
    {
        let mut out = Vec::with_capacity(size_of_shape(shape));
        for (&x, &y) in xs.iter().zip(ys) {
            out.push(f(x, y)?);
        }
        return Array::from_vec(out, shape);
    }
    let plan = RunPlan::new(shape, [left.strides(), right.strides()]);
    let out = try_collect_binary(
        &plan,
        &left.data,
        &right.data,
        [left.offset(), right.offset()],
        f,
    )?;
    Array::from_vec(out, shape)
}

/// Result of aligning two operands for a binary map.
///
/// `Some` is a broadcast view; `None` means the original already matches
/// `shape` and should be used as-is.
struct AlignedBinary<A: Scalar, B: Scalar> {
    left: Option<Array<A>>,
    right: Option<Array<B>>,
    shape: Vec<usize>,
}

fn align_binary<A: Scalar, B: Scalar>(
    a: &Array<A>,
    b: &Array<B>,
) -> Result<AlignedBinary<A, B>> {
    if a.shape() == b.shape() {
        return Ok(AlignedBinary {
            left: None,
            right: None,
            shape: a.shape().to_vec(),
        });
    }
    let shape = broadcast_shape(a.shape(), b.shape())?;
    let left = (a.shape() != shape.as_slice())
        .then(|| a.broadcast_to(&shape))
        .transpose()?;
    let right = (b.shape() != shape.as_slice())
        .then(|| b.broadcast_to(&shape))
        .transpose()?;
    Ok(AlignedBinary { left, right, shape })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_unary_contiguous() {
        let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
        let b = map_unary(&a, |x| x * 2).unwrap();
        assert_eq!(b.get(&[0]).unwrap(), 2);
        assert_eq!(b.get(&[2]).unwrap(), 6);
    }

    #[test]
    fn map_binary_broadcast() {
        let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
        let b = Array::from_slice(&[10_i64], &[1]).unwrap();
        let c = map_binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.get(&[1]).unwrap(), 12);
    }

    #[test]
    fn map_binary_noncontiguous() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let t = a.transpose();
        let ones = Array::from_vec(vec![1_i64; 6], &[3, 2]).unwrap();
        let c = map_binary(&t, &ones, |x, y| x + y).unwrap();
        assert_eq!(c.get(&[0, 1]).unwrap(), 5); // t[0,1]=4
        assert!(!t.is_c_contiguous());
        assert!(ones.is_c_contiguous());
    }

    #[test]
    fn map_binary_left_contiguous_right_not() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let t = a.transpose();
        let ones = Array::from_vec(vec![10_i64; 6], &[3, 2]).unwrap();
        // ones contiguous, t not — swapped argument order vs previous test
        let c = map_binary(&ones, &t, |x, y| x + y).unwrap();
        assert_eq!(c.get(&[0, 1]).unwrap(), 14);
    }

    #[test]
    fn try_map_binary_stops_on_strided_error() {
        use crate::error::Error;

        let values =
            Array::from_slice(&[6_i64, 12, 18, 24, 30, 36], &[2, 3]).unwrap();
        let divisors =
            Array::from_slice(&[1_i64, 1, 1, 0, 1, 1], &[2, 3]).unwrap();
        let mut calls = 0;
        let error = try_map_binary(
            &values.transpose(),
            &divisors.transpose(),
            |x, y| {
                calls += 1;
                if y == 0 {
                    Err(Error::DivideByZero)
                } else {
                    Ok(x / y)
                }
            },
        )
        .unwrap_err();

        assert_eq!(error, Error::DivideByZero);
        assert_eq!(calls, 2);
    }

    #[test]
    fn map_binary_mixed_types_on_transposed_operands() {
        let integers =
            Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let floats =
            Array::from_slice(&[0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0], &[2, 3])
                .unwrap();
        let out =
            map_binary(&integers.transpose(), &floats.transpose(), |x, y| {
                x as f64 + y
            })
            .unwrap();

        assert_eq!(out.to_vec(), vec![1.5, 6.0, 3.0, 7.5, 4.5, 9.0]);
    }

    #[test]
    fn map_unary_noncontiguous() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let t = a.transpose();
        let b = map_unary(&t, |x| x + 1).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(&[0, 1]).unwrap(), 5); // was 4
    }

    #[test]
    fn map_unary_contiguous_nonzero_offset() {
        use std::sync::Arc;
        // Buffer [0,1,2, 10,20,30,40,50,60] — logical 2x3 block at offset 3
        let data = Arc::new(vec![0_i64, 1, 2, 10, 20, 30, 40, 50, 60]);
        let a =
            Array::from_arc_raw_parts(data, vec![2, 3], vec![3, 1], 3, true)
                .unwrap();
        assert!(a.is_c_contiguous());
        assert_eq!(a.offset(), 3);
        assert_eq!(
            a.as_c_contiguous_slice().unwrap(),
            &[10_i64, 20, 30, 40, 50, 60]
        );
        let b = map_unary(&a, |x| x + 1).unwrap();
        assert_eq!(b.get(&[0, 0]).unwrap(), 11);
        assert_eq!(b.get(&[1, 2]).unwrap(), 61);
    }
}
