//! Gather (read) and scatter (write) implementations for array indexing.
//!
//! Basic indexing builds strided views when possible, like NumPy. Fancy and
//! boolean indexing always copy data into a new C-contiguous array. Scatter
//! copies assignment sources before writing so overlapping reads stay safe.

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::index::advance_multi_index;
use crate::index::prepare::{
    prepare_index, FancyLayout, PreparedEntry, PreparedIndex,
};
use crate::index::spec::IndexSpec;
use crate::shape::{
    checked_allocation_len, is_c_contiguous, offset_at, size_of_shape_unchecked,
};
use crate::traversal::{RunKind, RunPlan, StrideIter};

/// Read elements selected by `index`.
///
/// **Basic indexing** (`Index`, `Slice`, `NewAxis`, `Ellipsis`) returns a
/// shared-buffer **view** when the result is a strided slice of the original
/// buffer. **Fancy indexing** (integer or boolean arrays) always copies
/// selected elements into a new C-contiguous array.
///
/// Output shape rules:
/// - Integer indices along an axis collapse that axis (0-D when every axis is
///   an integer).
/// - Slices and `NewAxis` preserve or insert axes according to NumPy rules.
/// - Fancy integer arrays are broadcast together; their broadcast shape
///   contributes trailing (or adjacent) output axes.
/// - Boolean masks replace `ndim` source axes with the mask shape.
///
/// # Arguments
///
/// * `array` - Source array to read from.
/// * `index` - Normalized index tuple (see [`IndexSpec`]).
///
/// # Returns
///
/// A new array or view containing the selected elements. Fancy paths always
/// allocate a dense copy; basic paths may alias the source buffer.
///
/// # Errors
///
/// * [`Error::IndexOutOfBounds`](crate::error::Error::IndexOutOfBounds) —
///   resolved index outside an axis length.
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — fancy integer
///   arrays cannot be broadcast together.
/// * [`Error::InvalidArgument`](crate::error::Error::InvalidArgument) —
///   malformed index (e.g. duplicate ellipsis, zero step).
///
/// # Examples
///
/// ```rust
/// use sdnp::{gather, Array, IndexSpec};
///
/// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// let row = gather(&a, &[IndexSpec::index(0)]).unwrap();
/// assert_eq!(row.shape(), &[2]);
/// assert_eq!(row.get(&[1]).unwrap(), 2);
/// ```
pub fn gather<T: Scalar>(
    array: &Array<T>,
    index: &[IndexSpec],
) -> Result<Array<T>> {
    let prepared = prepare_index(array.shape(), index)?;
    if prepared.has_fancy() {
        gather_fancy(array, &prepared)
    } else {
        gather_basic(array, &prepared)
    }
}

/// Write a scalar to every location selected by `index`.
///
/// Uses the same basic vs fancy dispatch as [`gather`]. Basic paths update a
/// strided sub-region in place; fancy paths walk resolved buffer offsets.
/// The target array must be writable — broadcast views are read-only and
/// return [`Error::ReadOnly`](crate::error::Error::ReadOnly).
///
/// # Arguments
///
/// * `array` - Mutable destination array.
/// * `index` - Normalized index tuple (see [`IndexSpec`]).
/// * `value` - Scalar written to every selected location.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * [`Error::ReadOnly`](crate::error::Error::ReadOnly) — destination is a
///   read-only view (e.g. a broadcast view).
/// * [`Error::IndexOutOfBounds`](crate::error::Error::IndexOutOfBounds) —
///   resolved index outside an axis length.
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — fancy integer
///   arrays cannot be broadcast together.
///
/// # Examples
///
/// ```rust
/// use sdnp::{scatter, Array, IndexSpec};
///
/// let mut a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// scatter(&mut a, &[IndexSpec::index(0)], 0).unwrap();
/// assert_eq!(a.get(&[0, 0]).unwrap(), 0);
/// assert_eq!(a.get(&[1, 1]).unwrap(), 4);
/// ```
pub fn scatter<T: Scalar>(
    array: &mut Array<T>,
    index: &[IndexSpec],
    value: T,
) -> Result<()> {
    if !array.writable {
        return Err(Error::ReadOnly);
    }
    let prepared = prepare_index(array.shape(), index)?;
    if prepared.has_fancy() {
        scatter_fancy_scalar(array, &prepared, value)
    } else {
        scatter_basic_scalar(array, &prepared, value)
    }
}

/// Write broadcast `values` into locations selected by `index`.
///
/// Uses the same basic vs fancy dispatch as [`gather`]. Assignment values are
/// [`broadcast_to`](crate::array::Array::broadcast_to) the indexed region
/// shape before writing. When the destination shares a buffer with `values`,
/// `values` is copied first so overlapping reads cannot corrupt the write.
///
/// The target array must be writable — broadcast views are read-only and
/// return [`Error::ReadOnly`](crate::error::Error::ReadOnly).
///
/// # Arguments
///
/// * `array` - Mutable destination array.
/// * `index` - Normalized index tuple (see [`IndexSpec`]).
/// * `values` - Source array broadcast to the indexed output shape.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * [`Error::ReadOnly`](crate::error::Error::ReadOnly) — destination is a
///   read-only view (e.g. a broadcast view).
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — `values` cannot
///   be aligned to the indexed output shape, or fancy arrays conflict.
/// * [`Error::IndexOutOfBounds`](crate::error::Error::IndexOutOfBounds) —
///   resolved index outside an axis length.
///
/// # Examples
///
/// ```rust
/// use sdnp::{scatter_array, Array, IndexSpec};
///
/// let mut a = Array::from_slice(&[0_i64; 4], &[2, 2]).unwrap();
/// let src = Array::from_slice(&[9_i64, 8], &[2]).unwrap();
/// scatter_array(&mut a, &[IndexSpec::index(1)], &src).unwrap();
/// assert_eq!(a.get(&[1, 0]).unwrap(), 9);
/// ```
pub fn scatter_array<T: Scalar>(
    array: &mut Array<T>,
    index: &[IndexSpec],
    values: &Array<T>,
) -> Result<()> {
    if !array.writable {
        return Err(Error::ReadOnly);
    }
    let prepared = prepare_index(array.shape(), index)?;
    if prepared.has_fancy() {
        scatter_fancy_array(array, &prepared, values)
    } else {
        scatter_basic_array(array, &prepared, values)
    }
}

impl<T: Scalar> Array<T> {
    /// Read elements selected by `index`.
    ///
    /// Convenience wrapper for [`gather`]. See that function for indexing
    /// paths, output shape rules, and error conditions.
    ///
    /// # Arguments
    ///
    /// * `index` - Normalized index tuple (see [`IndexSpec`]).
    ///
    /// # Returns
    ///
    /// A new array or view containing the selected elements.
    ///
    /// # Errors
    ///
    /// Same as [`gather`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::{Array, IndexSpec};
    ///
    /// let a = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2]).unwrap();
    /// let col = a.gather(&[IndexSpec::full(), IndexSpec::index(0)]).unwrap();
    /// assert_eq!(col.shape(), &[2]);
    /// ```
    #[inline]
    pub fn gather(&self, index: &[IndexSpec]) -> Result<Array<T>> {
        gather(self, index)
    }

    /// Write a scalar to every location selected by `index`.
    ///
    /// Convenience wrapper for [`scatter`]. The array must be writable.
    ///
    /// # Arguments
    ///
    /// * `index` - Normalized index tuple (see [`IndexSpec`]).
    /// * `value` - Scalar written to every selected location.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Same as [`scatter`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::{Array, IndexSpec};
    ///
    /// let mut a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
    /// a.scatter(&[IndexSpec::full()], 0).unwrap();
    /// assert_eq!(a.get(&[1, 1]).unwrap(), 0);
    /// ```
    #[inline]
    pub fn scatter(&mut self, index: &[IndexSpec], value: T) -> Result<()> {
        scatter(self, index, value)
    }

    /// Write broadcast `values` into locations selected by `index`.
    ///
    /// Convenience wrapper for [`scatter_array`]. The array must be writable.
    ///
    /// # Arguments
    ///
    /// * `index` - Normalized index tuple (see [`IndexSpec`]).
    /// * `values` - Source array broadcast to the indexed output shape.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Same as [`scatter_array`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::{Array, IndexSpec};
    ///
    /// let mut a = Array::from_slice(&[0_i64; 4], &[2, 2]).unwrap();
    /// let src = Array::from_slice(&[9_i64, 8], &[2]).unwrap();
    /// a.scatter_array(&[IndexSpec::index(1)], &src).unwrap();
    /// assert_eq!(a.get(&[1, 0]).unwrap(), 9);
    /// ```
    #[inline]
    pub fn scatter_array(
        &mut self,
        index: &[IndexSpec],
        values: &Array<T>,
    ) -> Result<()> {
        scatter_array(self, index, values)
    }
}

/// View metadata shared by basic gather and basic scatter.
///
/// Captures the output shape, strides, and buffer offset of a basic index
/// without copying data. Integer indices collapse axes into the offset;
/// slices and `NewAxis` contribute output axes.
struct BasicViewMeta {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

/// Compute view shape, strides, and offset for a basic prepared index.
///
/// Walks [`PreparedEntry`] slots that do not involve fancy arrays. Integer
/// indices advance `offset`; slices append an output axis with strided step;
/// `NewAxis` inserts a length-1 zero-stride axis.
///
/// # Arguments
///
/// * `array` — source array being indexed
/// * `prepared` — index plan with no fancy layout
///
/// # Returns
///
/// [`BasicViewMeta`] describing the indexed view.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — index offset arithmetic overflows or yields
///   a negative buffer offset
fn basic_view_meta<T: Scalar>(
    array: &Array<T>,
    prepared: &PreparedIndex,
) -> Result<BasicViewMeta> {
    let mut offset = array.offset() as isize;
    let mut shape = Vec::new();
    let mut strides = Vec::new();
    let mut source_axis = 0usize;

    for entry in &prepared.entries {
        match entry {
            PreparedEntry::NewAxis => {
                // Inserted axis: length 1, stride 0 (no buffer movement).
                shape.push(1);
                strides.push(0);
            }
            PreparedEntry::Index(idx) => {
                // Integer index collapses an axis; advance offset only.
                let delta = isize::try_from(*idx)
                    .ok()
                    .and_then(|index| {
                        index.checked_mul(array.strides()[source_axis])
                    })
                    .ok_or_else(|| {
                        Error::InvalidArgument(
                            "index offset overflows isize".into(),
                        )
                    })?;
                offset = offset.checked_add(delta).ok_or_else(|| {
                    Error::InvalidArgument(
                        "index offset overflows isize".into(),
                    )
                })?;
                source_axis += 1;
            }
            PreparedEntry::Slice { start, len, step } => {
                let source_stride = array.strides()[source_axis];
                let delta =
                    start.checked_mul(source_stride).ok_or_else(|| {
                        Error::InvalidArgument(
                            "slice start offset overflows isize".into(),
                        )
                    })?;
                offset = offset.checked_add(delta).ok_or_else(|| {
                    Error::InvalidArgument(
                        "slice start offset overflows isize".into(),
                    )
                })?;
                shape.push(*len);
                // Output stride = source stride × slice step (0 if len ≤ 1).
                let stride = source_stride
                    .checked_mul(*step)
                    .or_else(|| (*len <= 1).then_some(0))
                    .ok_or_else(|| {
                        Error::InvalidArgument(
                            "slice stride overflows isize".into(),
                        )
                    })?;
                strides.push(stride);
                source_axis += 1;
            }
            PreparedEntry::IntegerArray(_) => {}
        }
    }

    Ok(BasicViewMeta {
        shape,
        strides,
        offset: usize::try_from(offset).map_err(|_| {
            Error::InvalidArgument(
                "indexing produced a negative buffer offset".into(),
            )
        })?,
    })
}

/// Return a shared-buffer view for a basic (non-fancy) index.
///
/// # Arguments
///
/// * `array` — source array
/// * `prepared` — basic prepared index (`fancy` is `None`)
///
/// # Returns
///
/// View array sharing `array`'s storage.
///
/// # Errors
///
/// Propagates errors from [`basic_view_meta`].
fn gather_basic<T: Scalar>(
    array: &Array<T>,
    prepared: &PreparedIndex,
) -> Result<Array<T>> {
    let meta = basic_view_meta(array, prepared)?;
    Array::from_shared_parts(
        Arc::clone(&array.data),
        meta.shape,
        meta.strides,
        meta.offset,
        array.writable,
    )
}

/// Write one scalar into every element selected by a basic index.
///
/// Ensures unique storage before mutation. Uses contiguous fill fast paths
/// when the indexed view is C-contiguous.
///
/// # Arguments
///
/// * `array` — destination array (mutated in place)
/// * `prepared` — basic prepared index
/// * `value` — scalar to broadcast into the selection
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Propagates errors from [`basic_view_meta`].
fn scatter_basic_scalar<T: Scalar>(
    array: &mut Array<T>,
    prepared: &PreparedIndex,
    value: T,
) -> Result<()> {
    let mut meta = basic_view_meta(array, prepared)?;
    if array.ensure_unique_storage_for_write() {
        meta = basic_view_meta(array, prepared)?;
    }
    let data = Arc::make_mut(&mut array.data);

    if is_c_contiguous(&meta.shape, &meta.strides) {
        let size = size_of_shape_unchecked(&meta.shape);
        data[meta.offset..meta.offset + size].fill(value);
        return Ok(());
    }

    let plan = RunPlan::new(&meta.shape, [&meta.strides]);
    plan.for_each([meta.offset as isize], |run| {
        if run.kinds[0] == RunKind::UnitStride {
            data[run.bases[0]..run.bases[0] + run.len].fill(value);
        } else {
            let mut pos = run.bases[0] as isize;
            for _ in 0..run.len {
                data[pos as usize] = value;
                pos += run.strides[0];
            }
        }
    });
    Ok(())
}

/// Write an array into every element selected by a basic index.
///
/// Broadcasts `values` to the indexed view shape, then copies element-wise
/// (with run-based fast paths for contiguous layouts).
///
/// # Arguments
///
/// * `array` — destination array (mutated in place)
/// * `prepared` — basic prepared index
/// * `values` — source values (broadcast to view shape)
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Propagates broadcast, view, or copy errors.
fn scatter_basic_array<T: Scalar>(
    array: &mut Array<T>,
    prepared: &PreparedIndex,
    values: &Array<T>,
) -> Result<()> {
    let mut meta = basic_view_meta(array, prepared)?;
    let aligned = prepare_scatter_source(array, values, &meta.shape)?;

    if array.ensure_unique_storage_for_write() {
        meta = basic_view_meta(array, prepared)?;
    }
    let dest_c_contiguous = is_c_contiguous(&meta.shape, &meta.strides);
    let src_slice = aligned.as_c_contiguous_slice();
    let data = Arc::make_mut(&mut array.data);

    if dest_c_contiguous {
        if let Some(src) = src_slice {
            let size = size_of_shape_unchecked(&meta.shape);
            data[meta.offset..meta.offset + size].copy_from_slice(src);
            return Ok(());
        }
    }

    let plan = RunPlan::new(&meta.shape, [&meta.strides, aligned.strides()]);
    plan.for_each([meta.offset as isize, aligned.offset() as isize], |run| {
        match (run.kinds[0], run.kinds[1]) {
            (RunKind::UnitStride, RunKind::UnitStride) => {
                let source =
                    &aligned.data[run.bases[1]..run.bases[1] + run.len];
                data[run.bases[0]..run.bases[0] + run.len]
                    .copy_from_slice(source);
            }
            (RunKind::UnitStride, RunKind::Repeated) => {
                // Broadcast source: one value written across a contiguous run.
                data[run.bases[0]..run.bases[0] + run.len]
                    .fill(aligned.data[run.bases[1]]);
            }
            _ => {
                let mut dst = run.bases[0] as isize;
                let mut src = run.bases[1] as isize;
                for _ in 0..run.len {
                    data[dst as usize] = aligned.data[src as usize];
                    dst += run.strides[0];
                    src += run.strides[1];
                }
            }
        }
    });
    Ok(())
}

/// Materialize a fancy-indexed copy of `array`.
///
/// Iterates source buffer offsets in fancy result order and collects values
/// into a new C-order vector.
///
/// # Arguments
///
/// * `array` — source array
/// * `prepared` — prepared index with [`PreparedIndex::fancy`] set
///
/// # Returns
///
/// New array with shape `layout.result_shape`.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — output allocation exceeds limits
///
/// # Panics
///
/// Panics if `prepared.fancy` is `None` (caller must branch on fancy first).
fn gather_fancy<T: Scalar>(
    array: &Array<T>,
    prepared: &PreparedIndex,
) -> Result<Array<T>> {
    let layout = prepared.fancy.as_ref().unwrap();
    let output_len = size_of_shape_unchecked(&layout.result_shape);
    checked_allocation_len::<T>(output_len)?;
    let mut out = Vec::with_capacity(output_len);
    for buffer_index in
        iter_fancy_source_offsets(array.strides(), array.offset(), prepared)
    {
        out.push(array.data[buffer_index]);
    }
    Array::from_vec(out, &layout.result_shape)
}

/// Write one scalar to every source element selected by a fancy index.
///
/// # Arguments
///
/// * `array` — destination array (mutated in place)
/// * `prepared` — prepared index with fancy layout
/// * `value` — scalar to write at each selected offset
///
/// # Returns
///
/// `Ok(())` on success.
fn scatter_fancy_scalar<T: Scalar>(
    array: &mut Array<T>,
    prepared: &PreparedIndex,
    value: T,
) -> Result<()> {
    let _ = array.ensure_unique_storage_for_write();
    let strides = array.strides().to_vec();
    let base_offset = array.offset();
    let data = Arc::make_mut(&mut array.data);
    for buffer_index in
        iter_fancy_source_offsets(&strides, base_offset, prepared)
    {
        data[buffer_index] = value;
    }
    Ok(())
}

/// Write broadcast assignment values through a fancy index.
///
/// Aligns `values` to the fancy result shape, then zips source buffer
/// offsets with assignment elements.
///
/// # Arguments
///
/// * `array` — destination array (mutated in place)
/// * `prepared` — prepared index with fancy layout
/// * `values` — assignment array (broadcast to result shape)
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Propagates broadcast errors from [`prepare_scatter_source`].
fn scatter_fancy_array<T: Scalar>(
    array: &mut Array<T>,
    prepared: &PreparedIndex,
    values: &Array<T>,
) -> Result<()> {
    let result_shape = &prepared.fancy.as_ref().unwrap().result_shape;
    let aligned = prepare_scatter_source(array, values, result_shape)?;

    let _ = array.ensure_unique_storage_for_write();
    let strides = array.strides().to_vec();
    let base_offset = array.offset();
    let data = Arc::make_mut(&mut array.data);
    let buffer_indices =
        iter_fancy_source_offsets(&strides, base_offset, prepared);

    if let Some(src) = aligned.as_c_contiguous_slice() {
        for (buffer_index, &value) in buffer_indices.zip(src.iter()) {
            data[buffer_index] = value;
        }
    } else {
        let source_indices = StrideIter::new(
            aligned.shape(),
            aligned.strides(),
            aligned.offset(),
        );
        for (buffer_index, source_index) in buffer_indices.zip(source_indices) {
            data[buffer_index] = aligned.data[source_index];
        }
    }
    Ok(())
}

/// Copy/broadcast assignment values before COW may detach shared storage.
///
/// When destination and values share a buffer, copies first so later
/// mutation does not affect the source view of `values`.
///
/// # Arguments
///
/// * `destination` — array being written into
/// * `values` — assignment source
/// * `target_shape` — indexed view or fancy result shape
///
/// # Returns
///
/// Broadcast (and possibly copied) array aligned to `target_shape`.
///
/// # Errors
///
/// Propagates broadcast failures.
fn prepare_scatter_source<T: Scalar>(
    destination: &Array<T>,
    values: &Array<T>,
    target_shape: &[usize],
) -> Result<Array<T>> {
    let aligned = if destination.shares_buffer_with(values) {
        values.copy().broadcast_to(target_shape)?
    } else {
        values.broadcast_to(target_shape)?
    };
    Ok(aligned)
}

/// Iterator over source buffer offsets for each fancy-result element.
///
/// Walks result coordinates in C-order and maps each to one source buffer
/// index via [`FancyOffsetIter::current_offset`].
///
/// # Arguments
///
/// * `source_strides` — strides of the array being indexed
/// * `base_offset` — buffer offset of the source view
/// * `prepared` — prepared index with fancy layout
///
/// # Returns
///
/// Exact-size iterator yielding one source buffer index per result element.
///
/// # Panics
///
/// Panics if `prepared.fancy` is `None`.
fn iter_fancy_source_offsets<'a>(
    source_strides: &'a [isize],
    base_offset: usize,
    prepared: &'a PreparedIndex,
) -> FancyOffsetIter<'a> {
    let layout = prepared.fancy.as_ref().unwrap();
    let remaining = size_of_shape_unchecked(&layout.result_shape);
    let ndim = layout.result_shape.len();
    FancyOffsetIter {
        source_strides,
        base_offset,
        prepared,
        layout,
        result_indices: vec![0; ndim],
        remaining,
    }
}

/// C-order walk over fancy result coordinates mapped to source offsets.
struct FancyOffsetIter<'a> {
    source_strides: &'a [isize],
    base_offset: usize,
    prepared: &'a PreparedIndex,
    layout: &'a FancyLayout,
    result_indices: Vec<usize>,
    remaining: usize,
}

impl Iterator for FancyOffsetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.current_offset();
        self.remaining -= 1;
        if self.remaining > 0 {
            advance_multi_index(
                &mut self.result_indices,
                &self.layout.result_shape,
            );
        }
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for FancyOffsetIter<'_> {}

impl FancyOffsetIter<'_> {
    /// Map the current fancy result coordinate to a source buffer index.
    ///
    /// Combines integer indices, slice positions, and fancy array lookups
    /// using `slot_to_result_axis` and the broadcast fancy coordinate block.
    ///
    /// # Arguments
    ///
    /// * `self` - Iterator state with `result_coord` at the current output
    ///   cell and resolved fancy broadcast metadata.
    ///
    /// # Returns
    ///
    /// Linear index into the source array buffer.
    fn current_offset(&self) -> usize {
        let mut offset = self.base_offset as isize;
        let mut source_axis = 0usize;
        // Fancy coordinates live in a contiguous block of result axes.
        let fancy_coords = &self.result_indices[self.layout.fancy_axis_start
            ..self.layout.fancy_axis_start + self.layout.fancy_axis_len];

        for (slot, entry) in self.prepared.entries.iter().enumerate() {
            match entry {
                PreparedEntry::NewAxis => {}
                PreparedEntry::Index(idx) => {
                    offset += *idx as isize * self.source_strides[source_axis];
                    source_axis += 1;
                }
                PreparedEntry::Slice { start, step, .. } => {
                    let axis = self.layout.slot_to_result_axis[slot]
                        .expect("missing basic result axis for slice");
                    let local = self.result_indices[axis];
                    offset += (*start + local as isize * *step)
                        * self.source_strides[source_axis];
                    source_axis += 1;
                }
                PreparedEntry::IntegerArray(fancy) => {
                    let idx = read_fancy_usize(fancy, fancy_coords);
                    offset += idx as isize * self.source_strides[source_axis];
                    source_axis += 1;
                }
            }
        }
        offset as usize
    }
}

/// Read one normalized fancy index value at a broadcast result coordinate.
///
/// # Arguments
///
/// * `array` — fancy integer index array for one source axis
/// * `indices` — coordinate within the broadcast fancy shape block
///
/// # Returns
///
/// Non-negative source axis index stored at `indices`.
fn read_fancy_usize(array: &Array<i64>, indices: &[usize]) -> usize {
    let buffer_index = offset_at(indices, array.strides(), array.offset());
    array.data[buffer_index] as usize
}
