//! Zero-copy view operations: transpose, reshape, squeeze, and axis insert.
//!
//! Views reinterpret the same backing buffer with new shape/strides metadata.
//! Reshape returns a view when the source is C-contiguous; otherwise it copies.
//! These operations mirror NumPy's `reshape`, `transpose`, and `squeeze`
//! without changing the underlying educational storage model.

use std::sync::Arc;

use crate::array::Array;
use crate::axis::{
    resolve_axis, resolve_axis_mask, resolve_insert_axis,
    visit_resolved_permutation,
};
use crate::dtype::Scalar;
use crate::error::{Error, Result};

/// Insert a length-one axis without copying the backing buffer.
///
/// The new axis receives stride zero, matching NumPy's `np.newaxis` behavior.
/// Logical elements are unchanged; only shape/strides metadata grows.
///
/// # Arguments
///
/// * `array` — source array whose buffer is shared by the result
/// * `axis` — insertion position; negative values count from the end
///
/// # Returns
///
/// A view with rank increased by one; shares `array`'s `Arc` buffer.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — `axis` out of range after normalization,
///   or the resulting layout fails bounds validation in
///   [`Array::from_shared_parts`]
pub(crate) fn insert_axis_view<T: Scalar>(
    array: &Array<T>,
    axis: isize,
) -> Result<Array<T>> {
    let axis = resolve_insert_axis(axis, array.ndim())?;
    let mut shape = array.shape().to_vec();
    let mut strides = array.strides().to_vec();
    shape.insert(axis, 1);
    strides.insert(axis, 0);
    Array::from_shared_parts(
        Arc::clone(&array.data),
        shape,
        strides,
        array.offset(),
        array.is_writable(),
    )
}

impl<T: Scalar> Array<T> {
    /// Build a view after only reordering an existing valid layout's axes.
    ///
    /// A complete permutation preserves the source layout's reachable buffer
    /// indices, rank agreement, shape product, offset, and writability. The
    /// caller must therefore provide `shape` and `strides` produced by the
    /// same validated permutation; no unchecked constructor is exposed.
    #[inline]
    fn permuted_layout_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> Self {
        debug_assert_eq!(shape.len(), self.ndim());
        debug_assert_eq!(strides.len(), self.ndim());
        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
            offset: self.offset,
            writable: self.writable,
        }
    }

    /// Build a view after dropping only length-one axes from a valid layout.
    ///
    /// Removing such axes preserves every reachable backing-buffer index, so
    /// the general layout bounds validation would be redundant.
    #[inline]
    fn squeezed_layout_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> Self {
        debug_assert_eq!(shape.len(), strides.len());
        debug_assert!(shape.len() <= self.ndim());
        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
            offset: self.offset,
            writable: self.writable,
        }
    }

    /// Return a view with axes reversed (matrix transpose for 2-D).
    ///
    /// Shares the backing buffer. For 0-D and 1-D arrays, returns an
    /// equivalent-layout view with the same shape and strides.
    ///
    /// # Arguments
    ///
    /// None — only `self` is reinterpreted.
    ///
    /// # Returns
    ///
    /// A view with `shape` and `strides` reversed; preserves writability.
    ///
    /// # Errors
    ///
    /// Never fails; layout invariants are preserved by axis reversal.
    pub fn transpose(&self) -> Array<T> {
        if self.ndim() <= 1 {
            return self.view();
        }
        let shape: Vec<usize> = self.shape.iter().rev().copied().collect();
        let strides: Vec<isize> = self.strides.iter().rev().copied().collect();
        Self::from_shared_parts(
            Arc::clone(&self.data),
            shape,
            strides,
            self.offset,
            self.writable,
        )
        .expect("transpose preserves shape/strides rank")
    }

    /// Shorthand for [`Array::transpose`].
    ///
    /// # Arguments
    ///
    /// None — only `self` is reinterpreted.
    ///
    /// # Returns
    ///
    /// Same as [`Array::transpose`].
    ///
    /// # Errors
    ///
    /// Never fails.
    #[inline]
    pub fn t(&self) -> Array<T> {
        self.transpose()
    }

    /// Return a view with axes permuted according to `axes`.
    ///
    /// `axes` must be a permutation of `0..ndim`; negative values count from
    /// the end, NumPy-style. The backing buffer is not copied.
    ///
    /// # Arguments
    ///
    /// * `axes` — new axis order; length must equal [`ndim`](Array::ndim)
    ///
    /// # Returns
    ///
    /// A view sharing storage with reordered shape and strides.
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfBounds`] — an axis is outside the array rank
    /// * [`Error::NotPermutation`] — axes are duplicated or incomplete
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// let b = a.permute_axes(&[1, 0]).unwrap();
    /// assert_eq!(b.get(&[0, 1]).unwrap(), 3);
    /// ```
    pub fn permute_axes(&self, axes: &[isize]) -> Result<Array<T>> {
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        visit_resolved_permutation(axes, self.ndim(), |axis| {
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
        })?;
        Ok(self.permuted_layout_view(shape, strides))
    }

    /// Change the array's shape without changing its element order.
    ///
    /// One dimension may be `-1` to infer its length from the total size.
    /// Returns a zero-copy view when this array is C-contiguous; otherwise
    /// copies logical elements into a new owned buffer.
    ///
    /// # Arguments
    ///
    /// * `shape` — target axis lengths; at most one entry may be `-1`
    ///
    /// # Returns
    ///
    /// An array with the requested shape and the same elements in C-order.
    /// Views share the buffer when the source is C-contiguous.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — incompatible size, multiple `-1`
    ///   entries, invalid negative dimensions, or stride overflow
    /// * [`Error::BufferSizeMismatch`] — only when a copy path is taken and
    ///   materialized data does not match the resolved shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sdnp::Array;
    ///
    /// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// let b = a.reshape(&[4]).unwrap();
    /// assert_eq!(b.shape(), &[4]);
    /// ```
    pub fn reshape(&self, shape: &[isize]) -> Result<Array<T>> {
        let new_shape = resolve_reshape(shape, self.size())?;
        if self.is_c_contiguous() {
            let strides = crate::shape::c_order_strides(&new_shape)?;
            Self::from_shared_parts(
                Arc::clone(&self.data),
                new_shape,
                strides,
                self.offset,
                self.writable,
            )
        } else {
            let data = self.to_vec_c_order();
            Array::from_vec(data, &new_shape)
        }
    }

    /// Drop length-one axes as a zero-copy view.
    ///
    /// Pass `None` to remove every axis of length one. Pass `Some(axes)` to
    /// remove only the listed axes (negative indices allowed). Every listed
    /// axis must have length one. Removing all axes yields a valid 0-D array.
    ///
    /// # Arguments
    ///
    /// * `axes` — axes to squeeze, or `None` to squeeze all length-1 axes
    ///
    /// # Returns
    ///
    /// A view with fewer dimensions; shares the backing buffer.
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfBounds`] — an axis is outside the array rank
    /// * [`Error::DuplicateAxes`] — the same axis is listed more than once
    /// * [`Error::CannotSqueezeAxis`] — a listed axis has length other than one
    /// * [`Error::InvalidArgument`] — an explicit axis list is empty
    pub fn squeeze(&self, axes: Option<&[isize]>) -> Result<Array<T>> {
        let remove = match axes {
            None => None,
            Some([]) => {
                return Err(Error::InvalidArgument(
                    "squeeze axes must be a non-empty sequence".into(),
                ));
            }
            Some(axes) => {
                let remove = resolve_axis_mask(axes, self.ndim())?;
                // Resolve all axes before checking lengths to preserve error
                // precedence for bounds and duplicate failures.
                for &axis in axes {
                    let axis = resolve_axis(axis, self.ndim())?;
                    if self.shape[axis] != 1 {
                        return Err(Error::CannotSqueezeAxis {
                            axis,
                            axis_len: self.shape[axis],
                        });
                    }
                }
                Some(remove)
            }
        };
        let removed = axes.map_or_else(
            || self.shape.iter().filter(|&&length| length == 1).count(),
            <[isize]>::len,
        );
        let output_ndim = self.ndim() - removed;
        let mut shape = Vec::with_capacity(output_ndim);
        let mut strides = Vec::with_capacity(output_ndim);
        for (axis, (&length, &stride)) in
            self.shape.iter().zip(&self.strides).enumerate()
        {
            let should_remove = remove
                .as_ref()
                .map_or(length == 1, |mask| mask.contains(axis));
            if !should_remove {
                shape.push(length);
                strides.push(stride);
            }
        }
        Ok(self.squeezed_layout_view(shape, strides))
    }

    /// Return a cheap alias sharing buffer, shape, strides, and offset.
    ///
    /// Increments the `Arc` reference count only; no element copy occurs.
    ///
    /// # Arguments
    ///
    /// None — only `self` is aliased.
    ///
    /// # Returns
    ///
    /// A new [`Array`] handle referencing the same storage and layout.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub fn view(&self) -> Array<T> {
        Self::from_shared_parts(
            Arc::clone(&self.data),
            self.shape.clone(),
            self.strides.clone(),
            self.offset,
            self.writable,
        )
        .expect("view preserves shape/strides rank")
    }
}

/// Resolve a NumPy-style reshape spec, including at most one `-1` inference.
///
/// Converts signed dimensions to `usize`, infers a single `-1` from the
/// remaining element count, and verifies the product matches `size`.
///
/// # Arguments
///
/// * `shape` — target axis lengths; at most one entry may be `-1`
/// * `size` — total logical element count of the source array
///
/// # Returns
///
/// A concrete unsigned shape vector suitable for layout construction.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — multiple `-1` entries, negative dimension
///   other than `-1`, size mismatch, ambiguous inference when a zero axis
///   is present, or shape product overflow
fn resolve_reshape(shape: &[isize], size: usize) -> Result<Vec<usize>> {
    if shape.is_empty() {
        return Err(Error::InvalidArgument(
            "reshape target shape must be non-empty".into(),
        ));
    }
    let mut inferred = None;
    let mut known = 1_usize;
    let mut out = Vec::with_capacity(shape.len());

    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if inferred.is_some() {
                return Err(Error::InvalidArgument(
                    "only one reshape dimension may be -1".into(),
                ));
            }
            inferred = Some(i);
            out.push(0);
        } else {
            if d < 0 {
                return Err(Error::InvalidArgument(format!(
                    "invalid reshape dimension {d}"
                )));
            }
            let d = d as usize;
            known = known.checked_mul(d).ok_or_else(|| {
                Error::InvalidArgument(
                    "reshape shape size overflows usize".into(),
                )
            })?;
            out.push(d);
        }
    }

    if let Some(idx) = inferred {
        // Cannot infer when a zero dimension makes the product ambiguous.
        if known == 0 {
            return Err(Error::InvalidArgument(
                "cannot infer reshape dimension when another is 0".into(),
            ));
        }
        if size % known != 0 {
            return Err(Error::InvalidArgument(format!(
                "cannot reshape array of size {size} into shape {shape:?}"
            )));
        }
        out[idx] = size / known;
    } else if known != size {
        return Err(Error::InvalidArgument(format!(
            "cannot reshape array of size {size} into shape {shape:?}"
        )));
    }

    Ok(out)
}
