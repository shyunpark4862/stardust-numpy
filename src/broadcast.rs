//! NumPy-style broadcasting: shape alignment and zero-stride views.
//!
//! Broadcasting lets differently shaped arrays participate in the same
//! element-wise operation by virtually repeating length-one axes. This module
//! computes the common output shape and builds read-only views with stride
//! zero on stretched axes — the same model NumPy uses before ufunc dispatch.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

/// Compute the broadcast-compatible output shape for two input shapes.
///
/// Equivalent to [`broadcast_shapes`] with a two-element slice. Axes are
/// aligned from the trailing end; length-one axes stretch to match.
///
/// # Arguments
///
/// * `a` — first input shape
/// * `b` — second input shape
///
/// # Returns
///
/// The shape that both inputs can broadcast to.
///
/// # Errors
///
/// * [`Error::Broadcast`] — a pair of non-one axis lengths disagree
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    broadcast_shapes(&[a, b])
}

/// Compute the broadcast-compatible output shape for many input shapes.
///
/// Dimensions are aligned from the **trailing** axis outward. Missing
/// leading axes are treated as length one; two non-one lengths must match
/// or broadcast fails.
///
/// # Arguments
///
/// * `shapes` — slice of shape references to align (may be empty)
///
/// # Returns
///
/// The common output shape, or an empty vector when `shapes` is empty.
///
/// # Errors
///
/// * [`Error::Broadcast`] — incompatible non-one lengths on the same axis
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }

    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = Vec::with_capacity(max_ndim);

    // Compare the i-th axis from the end across all shapes.
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

/// Broadcast every array in `arrays` to their common output shape.
///
/// Each result is a read-only view with stride zero on stretched axes.
/// No element data is copied.
///
/// # Arguments
///
/// * `arrays` — arrays to align (may be empty)
///
/// # Returns
///
/// One broadcast view per input, all with the same shape.
///
/// # Errors
///
/// * [`Error::Broadcast`] — inputs cannot be aligned under NumPy rules
/// * [`Error::InvalidArgument`] — a broadcast view layout fails validation
///
/// # Examples
///
/// ```
/// use sdnp::Array;
/// use sdnp::broadcast_arrays;
///
/// let a = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
/// let b = Array::from_slice(&[10_i64], &[1]).unwrap();
/// let out = broadcast_arrays(&[&a, &b]).unwrap();
/// assert_eq!(out[0].shape(), &[3]);
/// assert_eq!(out[1].get(&[2]).unwrap(), 10);
/// ```
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
    /// Return a view broadcast to `shape` without copying data.
    ///
    /// Length-one source axes gain stride zero so the same element is read
    /// at every output coordinate along that axis. The result is read-only.
    ///
    /// # Arguments
    ///
    /// * `shape` — target output shape (rank ≥ this array's rank)
    ///
    /// # Returns
    ///
    /// A read-only view with the requested shape and zero strides on
    /// stretched axes.
    ///
    /// # Errors
    ///
    /// * [`Error::Broadcast`] — `shape` is not broadcast-compatible
    /// * [`Error::InvalidArgument`] — resulting layout exceeds buffer bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[5_i64], &[1]).unwrap();
    /// let b = a.broadcast_to(&[3, 1]).unwrap();
    /// assert_eq!(b.get(&[2, 0]).unwrap(), 5);
    /// assert!(!b.is_writable());
    /// ```
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Array<T>> {
        if shape.len() < self.ndim() {
            return Err(Error::Broadcast {
                shapes: vec![self.shape.clone(), shape.to_vec()],
            });
        }

        // Pad leading axes as implicit length-one dimensions.
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
                // Virtual repeat: advance zero bytes per step.
                new_strides.push(0);
            } else {
                return Err(Error::Broadcast {
                    shapes: vec![self.shape.clone(), shape.to_vec()],
                });
            }
        }

        Self::from_shared_parts(
            Arc::clone(&self.data),
            shape.to_vec(),
            new_strides,
            self.offset,
            false, // NumPy marks broadcast views read-only
        )
    }
}
