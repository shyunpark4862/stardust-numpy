//! Lockstep iteration over broadcast-compatible arrays.
//!
//! Like NumPy's `np.nditer`, each step yields one scalar from every operand
//! after implicit broadcasting. Contiguous operands share one linear index;
//! strided operands advance a shared multi-index instead.

use crate::array::Array;
use crate::broadcast::broadcast_arrays;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::index::advance_multi_index;
use crate::shape::{checked_size_of_shape, offset_at};

/// Read-only C-order iterator over broadcast-aligned operands.
///
/// Like NumPy's `np.nditer`, each step yields one scalar from every operand
/// after implicit broadcasting. The item type is `Vec<T>` with one entry per
/// operand, in input order. Operands must share the same scalar type in the
/// Rust core; mixed dtypes are handled in the Python binding layer.
///
/// Created by [`nditer`].
pub struct NdIter<T: Scalar> {
    operands: Vec<Array<T>>,
    shape: Vec<usize>,
    indices: Vec<usize>,
    linear: usize,
    remaining: usize,
    all_contiguous: bool,
}

/// Iterate one or more broadcast-compatible arrays in lockstep.
///
/// All operands are first aligned with [`broadcast_arrays`]. Each step yields
/// `Vec<T>` containing one scalar from each operand at the current C-order
/// coordinate. Contiguous operands share a fast linear index path; strided
/// operands advance a shared multi-index instead.
///
/// # Arguments
///
/// * `operands` - Slice of array references to iterate together.
///
/// # Returns
///
/// An [`NdIter`] over the broadcast shape of all operands.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when operand
/// shapes cannot be aligned under NumPy rules.
///
/// # Examples
///
/// ```rust
/// use sdnp::{nditer, Array};
///
/// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[10_i64], &[1]).unwrap();
/// let steps: Vec<_> = nditer(&[&a, &b]).unwrap().collect();
/// assert_eq!(steps[0], vec![1, 10]);
/// assert_eq!(steps[3], vec![4, 10]);
/// ```
pub fn nditer<T: Scalar>(operands: &[&Array<T>]) -> Result<NdIter<T>> {
    let operands = broadcast_arrays(operands)?;
    let shape = operands[0].shape().to_vec();
    let remaining = checked_size_of_shape(&shape)?;
    let all_contiguous = operands.iter().all(Array::is_c_contiguous);

    Ok(NdIter {
        operands,
        indices: vec![0; shape.len()],
        shape,
        linear: 0,
        remaining,
        all_contiguous,
    })
}

impl<T: Scalar> Iterator for NdIter<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let values = if self.all_contiguous {
            // Fast path: one linear index into each contiguous buffer.
            self.operands
                .iter()
                .map(|operand| {
                    operand.as_buffer()[operand.offset() + self.linear]
                })
                .collect()
        } else {
            self.operands
                .iter()
                .map(|operand| {
                    let offset = offset_at(
                        &self.indices,
                        operand.strides(),
                        operand.offset(),
                    );
                    operand.as_buffer()[offset]
                })
                .collect()
        };

        self.remaining -= 1;
        self.linear += 1;
        if self.remaining > 0 && !self.all_contiguous {
            advance_multi_index(&mut self.indices, &self.shape);
        }
        Some(values)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for NdIter<T> {}
