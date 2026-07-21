//! NumPy-style broadcasting: shape alignment and broadcast views.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

/// Broadcast-compatible result shape for two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    broadcast_shapes(&[a, b])
}

/// Broadcast-compatible result shape for any number of shapes.
///
/// Dimensions are aligned from the trailing axis. Size-`1` axes stretch;
/// missing leading axes are treated as size `1`.
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }

    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = Vec::with_capacity(max_ndim);

    for offset in 1..=max_ndim {
        let mut target = 1_usize;
        for shape in shapes {
            let dim = if offset <= shape.len() {
                shape[shape.len() - offset]
            } else {
                1
            };
            if dim != 1 {
                if target == 1 {
                    target = dim;
                } else if dim != target {
                    return Err(Error::Broadcast {
                        shapes: shapes.iter().map(|s| s.to_vec()).collect(),
                    });
                }
            }
        }
        result.push(target);
    }

    result.reverse();
    Ok(result)
}

/// Broadcast every array to a common shape.
pub fn broadcast_arrays<T: Scalar>(
    arrays: &[&Array<T>],
) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(Vec::new());
    }
    let shapes: Vec<&[usize]> = arrays.iter().map(|a| a.shape()).collect();
    let target = broadcast_shapes(&shapes)?;
    arrays.iter().map(|a| a.broadcast_to(&target)).collect()
}

impl<T: Scalar> Array<T> {
    /// Broadcast this array to `shape` as a view (stride `0` on stretched axes).
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Array<T>> {
        if shape.len() < self.ndim() {
            return Err(Error::Broadcast {
                shapes: vec![self.shape.clone(), shape.to_vec()],
            });
        }

        let pad = shape.len() - self.ndim();
        let mut new_strides = Vec::with_capacity(shape.len());

        for axis in 0..shape.len() {
            let (from_dim, from_stride) = if axis < pad {
                (1_usize, 0_isize)
            } else {
                let src = axis - pad;
                (self.shape[src], self.strides[src])
            };
            let to_dim = shape[axis];
            if from_dim == to_dim {
                new_strides.push(from_stride);
            } else if from_dim == 1 {
                new_strides.push(0);
            } else {
                return Err(Error::Broadcast {
                    shapes: vec![self.shape.clone(), shape.to_vec()],
                });
            }
        }

        Self::from_arc_raw_parts(
            Arc::clone(&self.data),
            shape.to_vec(),
            new_strides,
            self.offset,
            false, // NumPy: broadcast views are read-only
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create::zeros;

    #[test]
    fn broadcast_shape_basic() {
        assert_eq!(broadcast_shape(&[2, 1], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[3], &[3]).unwrap(), vec![3]);
        assert_eq!(broadcast_shape(&[], &[3]).unwrap(), vec![3]);
    }

    #[test]
    fn broadcast_shape_incompatible() {
        match broadcast_shape(&[2, 3], &[4]) {
            Err(Error::Broadcast { shapes }) => {
                assert_eq!(shapes, vec![vec![2, 3], vec![4]]);
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }
    }

    #[test]
    fn broadcast_to_view() {
        let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert!(a.shares_buffer_with(&b));
        assert!(!b.is_writable());
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.strides(), &[0, 1]);
        assert_eq!(b.get(&[0, 1]).unwrap(), 2);
        assert_eq!(b.get(&[1, 1]).unwrap(), 2);
    }

    #[test]
    fn broadcast_view_is_read_only() {
        let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
        let mut b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(b.set(&[0, 1], 99), Err(Error::ReadOnly));
        assert_eq!(a.get(&[1]).unwrap(), 2);

        let mut owned = b.copy();
        assert!(owned.is_writable());
        owned.set(&[0, 1], 99).unwrap();
        assert_eq!(owned.get(&[0, 1]).unwrap(), 99);
        assert_eq!(owned.get(&[1, 1]).unwrap(), 2);
    }

    #[test]
    fn broadcast_arrays_pair() {
        let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
        let b = zeros::<i64>(&[3, 1]).unwrap();
        let out = broadcast_arrays(&[&a, &b]).unwrap();
        assert_eq!(out[0].shape(), &[3, 3]);
        assert_eq!(out[1].shape(), &[3, 3]);
    }
}
