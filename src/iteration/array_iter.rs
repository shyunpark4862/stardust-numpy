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

/// Read-only iterator over an array's scalar values in logical C-order.
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
pub struct NdEnumerate<'a, T: Scalar> {
    indices: NdIndex,
    values: FlatIter<'a, T>,
}

/// Enumerate an array as `(coordinate, value)` pairs in logical C-order.
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

/// Iterator over shared-buffer views selected along axis 0.
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
    /// Iterate over scalar values in logical C-order without materializing.
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

    /// Return the length of axis 0.
    pub fn axis0_len(&self) -> usize {
        debug_assert!(
            self.ndim() > 0,
            "axis-0 length is undefined for a 0-D array"
        );
        self.shape()[0]
    }

    /// Iterate over shared-buffer views selected along axis 0.
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
