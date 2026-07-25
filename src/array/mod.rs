//! Core [`Array`] type, element access, and view helpers.

mod element;
mod view;

pub(crate) use view::insert_axis_view;

use std::borrow::Cow;
use std::sync::Arc;

use crate::dtype::{ArrayCast, Scalar};
use crate::error::{Error, Result};
use crate::shape::{
    c_order_strides, checked_size_of_shape, is_c_contiguous, size_of_shape,
};

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
        let size = checked_size_of_shape(shape)?;
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
    pub(crate) fn from_shared_parts(
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
        validate_layout_bounds(data.len(), &shape, &strides, offset)?;
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
        debug_assert_eq!(self.ndim(), 0, "item() requires a 0-D array");
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

    /// Cast every element to `Out` and return a new C-contiguous array.
    ///
    /// All conversions among `bool`, `i64`, `f64`, and
    /// [`Complex64`](crate::Complex64) are supported. Narrowing follows Rust
    /// `as` semantics; complex-to-real conversions use the real component.
    /// Casting to the same dtype still returns a deep copy.
    pub fn astype<Out>(&self) -> Result<Array<Out>>
    where
        Out: Scalar,
        T: ArrayCast<Out>,
    {
        crate::ufunc::kernels::map_unary(self, T::array_cast)
    }

    /// Collect logical elements in C-order into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        self.to_vec_c_order()
    }

    /// Collect logical elements in C-order into a new `Vec`.
    pub(crate) fn to_vec_c_order(&self) -> Vec<T> {
        self.to_c_order_cow().into_owned()
    }

    /// Borrow logical C-order elements when packed, otherwise materialize them.
    pub(crate) fn to_c_order_cow(&self) -> Cow<'_, [T]> {
        if let Some(slice) = self.as_c_contiguous_slice() {
            return Cow::Borrowed(slice);
        }

        let plan = crate::traversal::RunPlan::new(&self.shape, [&self.strides]);
        Cow::Owned(crate::traversal::collect_unary(
            &plan,
            &self.data,
            self.offset,
            |value| value,
        ))
    }
}

fn validate_layout_bounds(
    buffer_len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: usize,
) -> Result<()> {
    if shape.contains(&0) {
        if offset > buffer_len {
            return Err(Error::InvalidArgument(
                "empty array offset exceeds backing buffer".into(),
            ));
        }
        return Ok(());
    }

    let mut minimum = offset as i128;
    let mut maximum = offset as i128;
    for (&length, &stride) in shape.iter().zip(strides) {
        let extent = (length - 1) as i128 * stride as i128;
        if extent < 0 {
            minimum += extent;
        } else {
            maximum += extent;
        }
    }
    if minimum < 0 || maximum >= buffer_len as i128 {
        return Err(Error::InvalidArgument(format!(
            "array layout [{minimum}, {maximum}] exceeds backing buffer of length {buffer_len}"
        )));
    }
    Ok(())
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
        assert!(a.strides().is_empty());
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

    #[test]
    fn to_vec_handles_negative_stride_and_nonzero_offset() {
        let a = Array::from_shared_parts(
            Arc::new(vec![0_i64, 1, 2, 3, 4, 5, 6, 7]),
            vec![2, 2],
            vec![3, -1],
            4,
            true,
        )
        .unwrap();

        assert_eq!(a.to_vec(), vec![4, 3, 7, 6]);
    }

    #[test]
    fn shared_layout_must_stay_inside_backing_buffer() {
        let data = Arc::new(vec![1_i64, 2, 3]);
        assert!(Array::from_shared_parts(
            Arc::clone(&data),
            vec![3],
            vec![1],
            1,
            true,
        )
        .is_err());
        assert!(
            Array::from_shared_parts(data, vec![3], vec![-1], 1, true).is_err()
        );
    }
}
