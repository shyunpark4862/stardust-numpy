//! Core [`Array`] type, element access, and view helpers.

mod element;
mod view;

use std::sync::Arc;

use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::shape::{c_order_strides, is_c_contiguous, size_of_shape};

/// N-dimensional array with element type `T`.
///
/// Storage is a shared [`Arc<Vec<T>>`] so views can share a buffer;
/// writes use copy-on-write via [`Arc::make_mut`]. Strides are in
/// **element** units (not bytes). An empty `shape` (`[]`) is a valid
/// 0-D array with one logical element.
///
/// [`Clone`] increments the buffer reference count (cheap). Use
/// [`Array::copy`] for a deep contiguous copy.
#[derive(Clone, Debug)]
pub struct Array<T: Scalar> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: usize,
    /// `false` for broadcast views (NumPy-style read-only).
    pub(crate) writable: bool,
}

impl<T: Scalar> Array<T> {
    /// Construct a C-contiguous array by **moving** a [`Vec`] buffer.
    ///
    /// Prefer [`Array::from_slice`] when you have an array/slice literal.
    /// Strides and offset are computed internally (C-order, offset 0).
    ///
    /// `shape == []` creates a 0-D array; `data` must then have length 1.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let size = size_of_shape(shape);
        if data.len() != size {
            return Err(Error::BufferSizeMismatch {
                buffer_len: data.len(),
                size,
            });
        }
        let strides = c_order_strides(shape);
        Ok(Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            writable: true,
        })
    }

    /// Construct a C-contiguous array by **copying** from a slice (or fixed array).
    ///
    /// ```
    /// use sdnp::Array;
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// assert_eq!(a.get(&[1, 0]).unwrap(), 3);
    /// ```
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Internal: build a (possibly shared) view with explicit layout.
    pub(crate) fn from_arc_raw_parts(
        data: Arc<Vec<T>>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
        writable: bool,
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(Error::ShapeStridesMismatch {
                shape_ndim: shape.len(),
                strides_ndim: strides.len(),
            });
        }
        Ok(Self {
            data,
            shape,
            strides,
            offset,
            writable,
        })
    }

    /// Return the single element of a 0-D array.
    pub fn item(&self) -> Result<T> {
        if self.ndim() != 0 {
            return Err(Error::InvalidArgument(format!(
                "item() requires a 0-D array, got ndim={}",
                self.ndim()
            )));
        }
        Ok(self.data[self.offset])
    }

    /// Array dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element-unit strides.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of logical elements.
    #[inline]
    pub fn size(&self) -> usize {
        size_of_shape(&self.shape)
    }

    /// Whether this array may be written via [`Array::set`].
    ///
    /// Broadcast views are read-only (NumPy-compatible).
    #[inline]
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// Buffer offset of the logical origin.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Borrow the full backing buffer (may include elements outside this view).
    ///
    /// Prefer [`Array::get`], [`Array::as_c_contiguous_slice`], or iteration
    /// helpers for logical elements only. For Python buffer export, do **not**
    /// expose this whole slice as writable zero-copy unless the view owns the
    /// entire buffer and is C-contiguous from offset 0.
    #[inline]
    pub fn as_buffer(&self) -> &[T] {
        &self.data
    }

    /// Borrow the logical elements when this array is C-contiguous.
    ///
    /// Returns a subslice of the backing buffer starting at [`offset`](Self::offset).
    /// Useful for ufunc fast-paths and for read-only Python/NumPy buffer export.
    #[inline]
    pub fn as_c_contiguous_slice(&self) -> Option<&[T]> {
        if !self.is_c_contiguous() {
            return None;
        }
        let start = self.offset;
        let end = start + self.size();
        Some(&self.data[start..end])
    }

    /// Whether this array's logical elements are packed in C-order in memory.
    ///
    /// A non-zero [`offset`](Array::offset) is allowed; the contiguous block
    /// then starts at that index in the backing buffer.
    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.shape, &self.strides)
    }

    /// Whether `self` and `other` share the same backing buffer.
    #[inline]
    pub fn shares_buffer_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// Deep copy into a new C-contiguous array.
    pub fn copy(&self) -> Array<T> {
        let data = self.to_vec_c_order();
        let shape = self.shape.clone();
        let strides = c_order_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            writable: true,
        }
    }

    /// Collect logical elements in C-order into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        self.to_vec_c_order()
    }

    /// Collect logical elements in C-order into a new `Vec`.
    pub(crate) fn to_vec_c_order(&self) -> Vec<T> {
        if let Some(slice) = self.as_c_contiguous_slice() {
            return slice.to_vec();
        }

        use crate::stride_iter::StrideIter;

        StrideIter::new(&self.shape, &self.strides, self.offset)
            .map(|physical| self.data[physical])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_ok() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.strides(), &[3, 1]);
        assert_eq!(a.size(), 6);
    }

    #[test]
    fn from_slice_0d() {
        let a = Array::from_slice(&[42_i64], &[]).unwrap();
        assert_eq!(a.ndim(), 0);
        assert_eq!(a.size(), 1);
        assert_eq!(a.strides(), &[]);
        assert_eq!(a.item().unwrap(), 42);
        assert_eq!(a.get(&[]).unwrap(), 42);
    }

    #[test]
    fn from_slice_size_mismatch() {
        assert!(Array::from_slice(&[1_i64, 2], &[2, 2]).is_err());
    }

    #[test]
    fn from_vec_ok() {
        let a = Array::from_vec(vec![1_i64, 2, 3, 4], &[2, 2]).unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.get(&[1, 0]).unwrap(), 3);
    }
}
