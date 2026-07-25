//! Single-element read/write and copy-on-write (CoW) storage detachment.
//!
//! Indexed access maps logical coordinates through strides to a flat buffer
//! offset. Writes on shared buffers clone only the logical elements when the
//! view does not cover the entire allocation — a common ndarray optimization
//! pattern worth studying alongside NumPy's buffer protocol rules.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::{c_order_strides_unchecked, offset_at};

impl<T: Scalar> Array<T> {
    /// Read one element at the given multi-dimensional index.
    ///
    /// Maps `indices` through strides to a flat buffer offset without
    /// copying storage.
    ///
    /// # Arguments
    ///
    /// * `indices` — one coordinate per axis, in axis order
    ///
    /// # Returns
    ///
    /// A copy of the scalar at the resolved buffer location.
    ///
    /// # Errors
    ///
    /// * [`Error::IndexOutOfBounds`] — an index is ≥ its axis length
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2]).unwrap();
    /// assert_eq!(a.get(&[0, 1]).unwrap(), 20);
    /// ```
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        let buf_idx = self.checked_offset(indices)?;
        Ok(self.data[buf_idx])
    }

    /// Write one element at the given multi-dimensional index.
    ///
    /// When other arrays share this buffer, storage is detached first via
    /// copy-on-write so peers are not mutated.
    ///
    /// # Arguments
    ///
    /// * `indices` — one coordinate per axis, in axis order
    /// * `value` — scalar to store at the resolved location
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; the array remains writable afterward.
    ///
    /// # Errors
    ///
    /// * [`Error::ReadOnly`] — this is a broadcast (read-only) view
    /// * [`Error::IndexOutOfBounds`] — an index is out of range for its axis
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let mut a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// a.set(&[1, 0], 99).unwrap();
    /// assert_eq!(a.get(&[1, 0]).unwrap(), 99);
    /// ```
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

    /// Ensure this array owns its storage before an in-place mutation.
    ///
    /// When the view covers the entire contiguous buffer, only the `Arc`
    /// reference count is decremented. Otherwise logical elements are
    /// materialized in C-order into a fresh buffer. Returns `true` when the
    /// layout (strides/offset) was normalized as part of detaching.
    ///
    /// **CoW note:** Shared buffers with narrow or strided views cannot use
    /// the cheap `Arc::make_mut` path; they copy logical elements only,
    /// then reset to C-contiguous layout with offset zero.
    ///
    /// # Arguments
    ///
    /// None — only `self` may be modified.
    ///
    /// # Returns
    ///
    /// `true` when strides and offset were reset after a partial copy;
    /// `false` when storage was already unique or only the `Arc` was split.
    ///
    /// # Errors
    ///
    /// Never fails; allocation failure panics like ordinary `Vec` growth.
    #[must_use]
    pub(crate) fn ensure_unique_storage_for_write(&mut self) -> bool {
        if Arc::strong_count(&self.data) == 1 {
            return false;
        }

        let covers_entire_buffer = self.offset == 0
            && self.is_c_contiguous()
            && self.size() == self.data.len();
        if covers_entire_buffer {
            // Cheap path: clone the Vec in place via Arc::make_mut later.
            Arc::make_mut(&mut self.data);
            return false;
        }

        // Narrow view: copy only logical elements, then reset to C-order.
        let data = self.to_vec_c_order();
        self.data = Arc::new(data);
        self.strides = c_order_strides_unchecked(&self.shape);
        self.offset = 0;
        true
    }

    /// Bounds-check indices and return the flat buffer offset.
    ///
    /// Combines per-axis range checks with stride-based address arithmetic.
    /// Does not touch storage; safe to call on shared read-only views.
    ///
    /// # Arguments
    ///
    /// * `indices` — one coordinate per axis, in axis order
    ///
    /// # Returns
    ///
    /// The flat buffer index for the given multi-dimensional coordinate.
    ///
    /// # Errors
    ///
    /// * [`Error::IndexOutOfBounds`] — an index is ≥ its axis length
    fn checked_offset(&self, indices: &[usize]) -> Result<usize> {
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
