//! Element get/set with copy-on-write.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::{c_order_strides, offset_at};

impl<T: Scalar> Array<T> {
    /// Read one element by multi-index.
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        let buf_idx = self.checked_offset(indices)?;
        Ok(self.data[buf_idx])
    }

    /// Write one element by multi-index.
    ///
    /// Broadcast views are read-only and return [`Error::ReadOnly`](crate::error::Error::ReadOnly).
    ///
    /// If other views share this buffer, the buffer is cloned first
    /// (copy-on-write).
    ///
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if !self.writable {
            return Err(crate::error::Error::ReadOnly);
        }
        self.checked_offset(indices)?;
        let _ = self.ensure_unique_storage_for_write();
        let buf_idx = offset_at(indices, &self.strides, self.offset);
        Arc::make_mut(&mut self.data)[buf_idx] = value;
        Ok(())
    }

    /// Detach shared storage before a write.
    ///
    /// A non-trivial shared view materializes only its logical elements in
    /// C-order instead of cloning unrelated elements from the backing buffer.
    /// Returns whether detaching also changed the array's layout.
    #[must_use]
    pub(crate) fn ensure_unique_storage_for_write(&mut self) -> bool {
        debug_assert!(self.writable);
        if Arc::strong_count(&self.data) == 1 {
            return false;
        }

        let covers_entire_buffer = self.offset == 0
            && self.is_c_contiguous()
            && self.size() == self.data.len();
        if covers_entire_buffer {
            Arc::make_mut(&mut self.data);
            return false;
        }

        let data = self.to_vec_c_order();
        self.data = Arc::new(data);
        self.strides = c_order_strides(&self.shape);
        self.offset = 0;
        true
    }

    fn checked_offset(&self, indices: &[usize]) -> Result<usize> {
        debug_assert_eq!(
            indices.len(),
            self.ndim(),
            "expected {} indices, got {}",
            self.ndim(),
            indices.len()
        );
        for (&i, &dim) in indices.iter().zip(self.shape.iter()) {
            if i >= dim {
                return Err(crate::error::Error::IndexOutOfBounds {
                    index: i as i64,
                    axis_len: dim,
                });
            }
        }
        Ok(offset_at(indices, &self.strides, self.offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_set() {
        let mut a = Array::from_vec(vec![0_i64; 6], &[2, 3]).unwrap();
        a.set(&[1, 2], 42).unwrap();
        assert_eq!(a.get(&[1, 2]).unwrap(), 42);
    }

    #[test]
    fn set_cow_does_not_mutate_shared_view() {
        let mut a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
        let b = a.transpose();
        assert!(a.shares_buffer_with(&b));
        a.set(&[0, 0], 99).unwrap();
        assert!(!a.shares_buffer_with(&b));
        assert_eq!(a.get(&[0, 0]).unwrap(), 99);
        assert_eq!(b.get(&[0, 0]).unwrap(), 1);
    }
}
