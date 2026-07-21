//! Same-type and cross-type element-wise map kernels.
//!
//! Contiguous arrays use a flat zip; otherwise elements are visited with
//! [`crate::stride_iter::StrideIter`].

use crate::array::Array;
use crate::broadcast::broadcast_shape;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::size_of_shape;
use crate::stride_iter::StrideIter;

/// Apply `f` to every element of `a` (C-order), producing a new contiguous array.
pub fn map_unary<A, Out, F>(a: &Array<A>, mut f: F) -> Result<Array<Out>>
where
    A: Scalar,
    Out: Scalar,
    F: FnMut(A) -> Out,
{
    try_map_unary(a, |x| Ok(f(x)))
}

/// Fallible unary map.
pub fn try_map_unary<A, Out, F>(a: &Array<A>, mut f: F) -> Result<Array<Out>>
where
    A: Scalar,
    Out: Scalar,
    F: FnMut(A) -> Result<Out>,
{
    let size = a.size();
    let mut out = Vec::with_capacity(size);

    if let Some(xs) = a.as_c_contiguous_slice() {
        for &x in xs {
            out.push(f(x)?);
        }
    } else {
        for buf_idx in StrideIter::new(a.shape(), a.strides(), a.offset()) {
            out.push(f(a.data[buf_idx])?);
        }
    }

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
    try_map_binary(a, b, |x, y| Ok(f(x, y)))
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

    let mut out = Vec::with_capacity(size_of_shape(shape));

    match (left.as_c_contiguous_slice(), right.as_c_contiguous_slice()) {
        (Some(xs), Some(ys)) => {
            for (&x, &y) in xs.iter().zip(ys) {
                out.push(f(x, y)?);
            }
        }
        (Some(xs), None) => {
            let right_it =
                StrideIter::new(right.shape(), right.strides(), right.offset());
            for (&x, j) in xs.iter().zip(right_it) {
                out.push(f(x, right.data[j])?);
            }
        }
        (None, Some(ys)) => {
            let left_it =
                StrideIter::new(left.shape(), left.strides(), left.offset());
            for (i, &y) in left_it.zip(ys.iter()) {
                out.push(f(left.data[i], y)?);
            }
        }
        (None, None) => {
            let left_it =
                StrideIter::new(left.shape(), left.strides(), left.offset());
            let right_it =
                StrideIter::new(right.shape(), right.strides(), right.offset());
            for (i, j) in left_it.zip(right_it) {
                out.push(f(left.data[i], right.data[j])?);
            }
        }
    }

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
