//! The [`Array`] type: shared storage, strides, and layout invariants.
//!
//! An array is a shape/strides/offset triple over an `Arc`-backed buffer.
//! Views share memory; writes trigger copy-on-write when needed. Strides are
//! in element units. An empty shape denotes a 0-D array with one logical
//! element. Element access and view operations live in sibling modules.

mod element;
mod view;

pub(crate) use view::insert_axis_view;

use std::borrow::Cow;
use std::sync::Arc;

use crate::dtype::{ArrayCast, Scalar};
use crate::error::{Error, Result};
use crate::shape::{
    c_order_strides, c_order_strides_unchecked, checked_size_of_shape,
    is_c_contiguous, size_of_shape_unchecked, validate_shape_geometry,
};

/// N-dimensional array whose elements have scalar type `T`.
///
/// Storage is reference-counted (`Arc`) so views alias the same buffer
/// cheaply. Mutating through a shared view triggers copy-on-write via
/// [`Arc::make_mut`](std::sync::Arc::make_mut). [`Clone`] increments the
/// reference count; use [`Array::copy`] for an independent C-contiguous
/// copy. Broadcast views set [`Array::is_writable`] to `false`.
#[derive(Clone, Debug)]
pub struct Array<T: Scalar> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: usize,
    /// `false` for broadcast views, which NumPy treats as read-only.
    pub(crate) writable: bool,
}

impl<T: Scalar> Array<T> {
    /// Build a C-contiguous [`Array`] by taking ownership of `data`.
    ///
    /// The buffer length must equal the product of `shape`. A 0-D shape
    /// `[]` requires exactly one element. Strides are computed in
    /// C-order (row-major) element units.
    ///
    /// # Arguments
    ///
    /// * `data` — backing storage; ownership is transferred to the array
    /// * `shape` — axis lengths; may be empty for a 0-D scalar array
    ///
    /// # Returns
    ///
    /// A writable, C-contiguous array with offset zero.
    ///
    /// # Errors
    ///
    /// * [`Error::BufferSizeMismatch`] — `data.len()` ≠ shape product
    /// * [`Error::InvalidArgument`] — invalid shape geometry or stride
    ///   overflow
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let a = Array::from_vec(vec![1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// assert_eq!(a.shape(), &[2, 2]);
    /// ```
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        validate_shape_geometry(shape)?;
        let size = checked_size_of_shape(shape)?;
        if data.len() != size {
            return Err(Error::BufferSizeMismatch {
                buffer_len: data.len(),
                size,
            });
        }
        let strides = c_order_strides(shape)?;
        Ok(Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            writable: true,
        })
    }

    /// Build a C-contiguous [`Array`] by copying from a slice.
    ///
    /// Convenience wrapper around [`Array::from_vec`] for literal and
    /// borrowed inputs.
    ///
    /// # Arguments
    ///
    /// * `data` — elements to copy into a new owned buffer
    /// * `shape` — axis lengths (see [`Array::from_vec`])
    ///
    /// # Returns
    ///
    /// A writable, C-contiguous array (same as [`Array::from_vec`]).
    ///
    /// # Errors
    ///
    /// Same as [`Array::from_vec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// assert_eq!(a.get(&[1, 0]).unwrap(), 3);
    /// ```
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Construct an array from shared parts after layout validation.
    ///
    /// Low-level constructor for views: callers supply an existing `Arc`
    /// buffer plus shape, strides, offset, and writability. Every reachable
    /// flat index must lie inside the buffer. Views created here share
    /// storage until copy-on-write detaches on mutation.
    ///
    /// # Arguments
    ///
    /// * `data` — `Arc`-shared backing buffer
    /// * `shape` — axis lengths; rank must match `strides.len()`
    /// * `strides` — element-unit strides, one per axis
    /// * `offset` — flat buffer index of logical element `[0, …, 0]`
    /// * `writable` — `false` for broadcast (read-only) views
    ///
    /// # Returns
    ///
    /// A validated [`Array`] referencing the supplied storage and layout.
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeStridesMismatch`] — `shape` and `strides` differ in
    ///   length
    /// * [`Error::InvalidArgument`] — invalid shape geometry, stride/offset
    ///   overflow, or reachable indices exceed the buffer
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

    /// Return the sole element of a 0-D (scalar) array.
    ///
    /// Reads the logical origin at [`Array::offset`]; no axis bounds
    /// check is performed because 0-D arrays have no dimensions.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// A copy of the scalar at the array's logical origin.
    ///
    /// # Errors
    ///
    /// Never fails for arrays with valid layout; an out-of-range offset is
    /// rejected at construction time.
    pub fn item(&self) -> Result<T> {
        Ok(self.data[self.offset])
    }

    /// Borrow the shape vector (length of each axis).
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// A slice of axis lengths; empty for 0-D arrays.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow element-unit strides (one entry per axis).
    ///
    /// Strides describe how many buffer elements to skip when an index
    /// along one axis increases by one. Zero strides appear on broadcast
    /// or `newaxis` dimensions.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// Strides in **element** units (not bytes); empty for 0-D arrays.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return the number of dimensions (rank).
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// `0` for scalar (0-D) arrays; otherwise the length of
    /// [`shape`](Self::shape).
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return the total number of logical elements.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// The product of all axis lengths; `1` for 0-D arrays, `0` when any
    /// axis has length zero.
    ///
    /// # Errors
    ///
    /// Never fails; overflow is prevented at construction time.
    #[inline]
    pub fn size(&self) -> usize {
        size_of_shape_unchecked(&self.shape)
    }

    /// Return whether in-place writes (e.g. [`Array::set`]) are allowed.
    ///
    /// Broadcast views are read-only, matching NumPy behavior.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// `true` for writable arrays; `false` for broadcast views.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// Return the buffer index of the logical origin (first element).
    ///
    /// Combined with strides, this locates element `[0, …, 0]` without
    /// copying storage.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// Flat offset into [`as_buffer`](Self::as_buffer) for index
    /// `[0, …, 0]`.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Borrow the full backing buffer, including elements outside this view.
    ///
    /// Prefer [`Array::get`], [`Array::as_c_contiguous_slice`], or iteration
    /// helpers for logical elements. Do not export this slice as a writable
    /// zero-copy Python buffer unless the view owns the whole buffer and is
    /// C-contiguous from offset zero.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// The entire `Arc`-shared allocation, not just the logical elements.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn as_buffer(&self) -> &[T] {
        &self.data
    }

    /// Borrow logical elements as a contiguous slice when layout allows.
    ///
    /// When strides pack logical elements without gaps, this avoids a copy.
    /// Non-contiguous or broadcast-stretched views return `None`; use
    /// [`to_c_order_cow`](Self::to_c_order_cow) to borrow or materialize.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// `Some(slice)` when the view is C-contiguous; the slice starts at
    /// [`offset`](Self::offset) and spans [`size`](Self::size) elements.
    /// `None` for non-contiguous or broadcast-stretched layouts.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn as_c_contiguous_slice(&self) -> Option<&[T]> {
        if !self.is_c_contiguous() {
            return None;
        }
        let start = self.offset;
        let end = start + self.size();
        Some(&self.data[start..end])
    }

    /// Return whether logical elements are packed in C-order in memory.
    ///
    /// C-contiguity means the last axis has stride 1 and each outer stride
    /// equals the product of inner shape and stride — the layout
    /// [`Array::reshape`] needs for a zero-copy view.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// `true` when traversing axes in reverse order with the expected
    /// stride products reaches every logical element without gaps.
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.shape, &self.strides)
    }

    /// Return whether `self` and `other` alias the same backing allocation.
    ///
    /// # Arguments
    ///
    /// * `other` — array to compare for buffer identity
    ///
    /// # Returns
    ///
    /// `true` when both arrays reference the same `Arc` allocation
    /// (copy-on-write may still detach on write).
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn shares_buffer_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// Deep-copy logical elements into a new writable C-contiguous array.
    ///
    /// Unlike [`Clone`], which only bumps the `Arc` reference count, this
    /// always allocates fresh storage with normalized offset and strides.
    ///
    /// # Arguments
    ///
    /// None — only `self` is consulted.
    ///
    /// # Returns
    ///
    /// An independent array with offset zero, C-order strides, and no
    /// shared buffer with `self`.
    ///
    /// # Errors
    ///
    /// Never fails; allocation failure panics like ordinary `Vec` growth.
    pub fn copy(&self) -> Array<T> {
        let data = self.to_vec_c_order();
        let shape = self.shape.clone();
        let strides = c_order_strides_unchecked(&shape);
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
    /// [`Complex64`](crate::Complex64) are supported. Narrowing uses Rust `as`
    /// rules; complex-to-real uses the real part. Same-dtype casts still
    /// produce a deep copy.
    ///
    /// # Arguments
    ///
    /// None — only `self` is converted.
    ///
    /// # Returns
    ///
    /// A new writable array with dtype `Out` and C-contiguous layout.
    ///
    /// # Errors
    ///
    /// Propagates layout or allocation errors from the underlying ufunc
    /// materialization path (rare for well-formed inputs).
    pub fn astype<Out>(&self) -> Result<Array<Out>>
    where
        Out: Scalar,
        T: ArrayCast<Out>,
    {
        crate::ufunc::kernels::map_unary(self, T::array_cast)
    }

    /// Collect logical elements in C-order into a new `Vec`.
    ///
    /// Always materializes owned data; for a borrowed view when contiguous,
    /// prefer [`as_c_contiguous_slice`](Self::as_c_contiguous_slice) or
    /// [`to_c_order_cow`](Self::to_c_order_cow).
    ///
    /// # Arguments
    ///
    /// None — only `self` is traversed.
    ///
    /// # Returns
    ///
    /// A length-[`size`](Self::size) vector with elements in row-major order.
    ///
    /// # Errors
    ///
    /// Never fails; allocation failure panics like ordinary `Vec` growth.
    pub fn to_vec(&self) -> Vec<T> {
        self.to_vec_c_order()
    }

    /// Internal helper: always materialize C-order element data.
    ///
    /// Delegates to [`to_c_order_cow`](Self::to_c_order_cow) and takes
    /// ownership, so strided or broadcast layouts pay a full copy cost.
    ///
    /// # Arguments
    ///
    /// None — only `self` is traversed.
    ///
    /// # Returns
    ///
    /// An owned `Vec` of logical elements in row-major order.
    ///
    /// # Errors
    ///
    /// Never fails; allocation failure panics like ordinary `Vec` growth.
    pub(crate) fn to_vec_c_order(&self) -> Vec<T> {
        self.to_c_order_cow().into_owned()
    }

    /// Borrow C-order data when contiguous; otherwise materialize a copy.
    ///
    /// This is the copy-on-write friendly path for kernels that need a
    /// linear element slice: contiguous arrays return `Cow::Borrowed`
    /// without allocation; strided layouts walk a [`RunPlan`] and build
    /// `Cow::Owned`.
    ///
    /// # Arguments
    ///
    /// None — only `self` is traversed.
    ///
    /// # Returns
    ///
    /// `Cow::Borrowed` when [`as_c_contiguous_slice`](Self::as_c_contiguous_slice)
    /// succeeds; otherwise `Cow::Owned` with materialized C-order data.
    ///
    /// # Errors
    ///
    /// Never fails; allocation failure panics like ordinary `Vec` growth.
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

/// Verify that every index reachable via `(shape, strides, offset)` lies in
/// the backing buffer.
///
/// This is the core layout safety check for views. It computes the min and
/// max flat index reachable from the logical origin and rejects layouts
/// that would read outside `[0, buffer_len)`.
///
/// # Arguments
///
/// * `buffer_len` — length of the backing allocation in elements
/// * `shape` — axis lengths to validate
/// * `strides` — element-unit strides, same rank as `shape`
/// * `offset` — flat index of logical element `[0, …, 0]`
///
/// # Returns
///
/// `Ok(())` when every reachable index lies in the buffer.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — invalid shape geometry, size overflow,
///   buffer/offset exceeds `isize` range, or reachable index range exceeds
///   `buffer_len`
fn validate_layout_bounds(
    buffer_len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: usize,
) -> Result<()> {
    validate_shape_geometry(shape)?;
    checked_size_of_shape(shape)?;
    if buffer_len > isize::MAX as usize || offset > isize::MAX as usize {
        return Err(Error::InvalidArgument(
            "array buffer or offset exceeds isize address range".into(),
        ));
    }
    // Empty arrays have no reachable indices beyond the origin check.
    if shape.contains(&0) {
        if offset > buffer_len {
            return Err(Error::InvalidArgument(
                "empty array offset exceeds backing buffer".into(),
            ));
        }
        return Ok(());
    }

    // Track the min/max flat index reachable from the logical origin.
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
