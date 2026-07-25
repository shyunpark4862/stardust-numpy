//! Read-only iterators over array values, coordinates, and axis-0 slices.
//!
//! These types mirror NumPy helpers like `arr.flat`, `np.ndenumerate`, and
//! iterating over `arr[i, ...]` views. Contiguous arrays use direct slice
//! iteration; strided layouts fall back to [`StrideIter`].

use std::slice;
use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::traversal::StrideIter;

use super::{ndindex, NdIndex};

enum FlatStorage<'a, T: Scalar> {
    Contiguous(slice::Iter<'a, T>),
    Strided {
        data: &'a [T],
        offsets: StrideIter<'a>,
    },
}

/// Read-only iterator over scalar values in logical C-order.
///
/// Yields one element per step, visiting the last axis fastest (row-major /
/// C-order). Contiguous arrays iterate a direct slice; strided layouts use
/// [`StrideIter`].
///
/// Created by [`Array::flat`]. Each item is a copied scalar (`T`).
pub struct FlatIter<'a, T: Scalar> {
    storage: FlatStorage<'a, T>,
}

impl<T: Scalar> Iterator for FlatIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.storage {
            FlatStorage::Contiguous(values) => values.next().copied(),
            FlatStorage::Strided { data, offsets } => {
                offsets.next().map(|offset| data[offset])
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = match &self.storage {
            FlatStorage::Contiguous(values) => values.len(),
            FlatStorage::Strided { offsets, .. } => offsets.len(),
        };
        (remaining, Some(remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for FlatIter<'_, T> {}

/// Iterator over `(coordinate, value)` pairs in logical C-order.
///
/// Yields `(Vec<usize>, T)` on each step: a multi-index coordinate followed
/// by the element at that location. Coordinates advance in the same order as
/// [`FlatIter`] and [`NdIndex`].
///
/// Created by [`ndenumerate`].
pub struct NdEnumerate<'a, T: Scalar> {
    indices: NdIndex,
    values: FlatIter<'a, T>,
}

/// Enumerate an array as `(coordinate, value)` pairs in C-order.
///
/// Like NumPy's `np.ndenumerate`: coordinates advance in the same order as
/// [`FlatIter`]. Each step yields `(Vec<usize>, T)` — the coordinate vector
/// and the scalar at that location.
///
/// # Arguments
///
/// * `array` - Array to enumerate.
///
/// # Returns
///
/// An [`NdEnumerate`] iterator over coordinate/value pairs.
///
/// # Errors
///
/// Never fails for arrays with valid shapes.
///
/// # Examples
///
/// ```rust
/// use sdnp::{ndenumerate, Array};
///
/// let a = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2]).unwrap();
/// let pairs: Vec<_> = ndenumerate(&a).collect();
/// assert_eq!(pairs[0], (vec![0, 0], 10));
/// assert_eq!(pairs[3], (vec![1, 1], 40));
/// ```
pub fn ndenumerate<T: Scalar>(array: &Array<T>) -> NdEnumerate<'_, T> {
    let indices = ndindex(array.shape())
        .expect("an existing array always has a valid shape size");
    NdEnumerate {
        indices,
        values: array.flat(),
    }
}

impl<T: Scalar> Iterator for NdEnumerate<'_, T> {
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().zip(self.values.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<T: Scalar> ExactSizeIterator for NdEnumerate<'_, T> {}

/// Iterator over shared-buffer views along axis 0.
///
/// Yields one [`Array`] view per step with shape `array.shape[1:]`, like
/// `array[i, ...]` for a matrix row. Views share the parent buffer and
/// inherit its writability flag.
///
/// Created by [`Array::iter_axis0`].
pub struct Axis0Iter<'a, T: Scalar> {
    parent: &'a Array<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    next_index: usize,
    remaining: usize,
}

impl<T: Scalar> Iterator for Axis0Iter<'_, T> {
    type Item = Array<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let offset = (self.parent.offset as isize
            + self.next_index as isize * self.parent.strides[0])
            as usize;
        self.next_index += 1;
        self.remaining -= 1;

        Some(
            Array::from_shared_parts(
                Arc::clone(&self.parent.data),
                self.shape.clone(),
                self.strides.clone(),
                offset,
                self.parent.writable,
            )
            .expect("axis-0 iteration preserves shape/strides rank"),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for Axis0Iter<'_, T> {}

impl<T: Scalar> Array<T> {
    /// Iterate scalar values in logical C-order without copying.
    ///
    /// Yields one `T` per step in row-major order. Contiguous storage uses a
    /// fast slice iterator; strided layouts fall back to [`StrideIter`].
    ///
    /// # Arguments
    ///
    /// None — only `self` is traversed.
    ///
    /// # Returns
    ///
    /// A [`FlatIter`] over this array's elements.
    ///
    /// # Errors
    ///
    /// This method never fails; invalid layouts are rejected at construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// let flat: Vec<_> = a.flat().collect();
    /// assert_eq!(flat, vec![1, 2, 3, 4]);
    /// ```
    pub fn flat(&self) -> FlatIter<'_, T> {
        let storage = match self.as_c_contiguous_slice() {
            Some(values) => FlatStorage::Contiguous(values.iter()),
            None => FlatStorage::Strided {
                data: self.as_buffer(),
                offsets: StrideIter::new(
                    self.shape(),
                    self.strides(),
                    self.offset(),
                ),
            },
        };
        FlatIter { storage }
    }

    /// Length of axis 0 (`shape[0]`).
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// The size of the leading dimension.
    ///
    /// # Errors
    ///
    /// Never fails for arrays with at least one dimension.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// assert_eq!(a.axis0_len(), 2);
    /// ```
    pub fn axis0_len(&self) -> usize {
        self.shape()[0]
    }

    /// Iterate shared-buffer views along axis 0.
    ///
    /// Yields one sub-array view per index along the leading axis. Each view
    /// has shape `self.shape()[1..]` and shares this array's buffer.
    ///
    /// # Arguments
    ///
    /// None — only `self` is traversed.
    ///
    /// # Returns
    ///
    /// An [`Axis0Iter`] yielding `axis0_len()` views.
    ///
    /// # Errors
    ///
    /// This method never fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// let rows: Vec<_> = a.iter_axis0().collect();
    /// assert_eq!(rows.len(), 2);
    /// assert_eq!(rows[0].get(&[1]).unwrap(), 2);
    /// ```
    pub fn iter_axis0(&self) -> Axis0Iter<'_, T> {
        let remaining = self.axis0_len();
        Axis0Iter {
            parent: self,
            shape: self.shape[1..].to_vec(),
            strides: self.strides[1..].to_vec(),
            next_index: 0,
            remaining,
        }
    }
}
