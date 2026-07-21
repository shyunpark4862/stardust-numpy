//! gather / scatter implementation (basic views + fancy copies).

use std::sync::Arc;

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::index::advance_multi_index;
use crate::index::prepare::{
    prepare_index, FancyLayout, PreparedEntry, PreparedIndex,
};
use crate::index::spec::IndexSpec;
use crate::shape::{buffer_index, is_c_contiguous, size_of_shape};
use crate::stride_iter::StrideIter;

/// Select elements by `index`.
///
/// Basic indexing (`Index` / `Slice` / `NewAxis` / `Ellipsis`) returns a
/// **view** when possible. Fancy / boolean indexing always returns a new
/// C-contiguous copy. A fully integer index with no remaining axes yields a
/// **0-D** array (size 1).
pub fn gather<T: Scalar>(
    a: &Array<T>,
    index: &[IndexSpec],
) -> Result<Array<T>> {
    let prepared = prepare_index(a.shape(), index)?;
    if prepared.has_fancy() {
        gather_fancy(a, &prepared)
    } else {
        gather_basic(a, &prepared)
    }
}

/// Assign a scalar to every location selected by `index`.
pub fn scatter<T: Scalar>(
    a: &mut Array<T>,
    index: &[IndexSpec],
    value: T,
) -> Result<()> {
    if !a.writable {
        return Err(Error::ReadOnly);
    }
    let prepared = prepare_index(a.shape(), index)?;
    if prepared.has_fancy() {
        scatter_fancy_scalar(a, &prepared, value)
    } else {
        scatter_basic_scalar(a, &prepared, value)
    }
}

/// Assign `values` (broadcast to the indexed shape) into locations selected
/// by `index`.
///
/// Fancy assignment copies `values` first so overlapping reads/writes are safe.
pub fn scatter_array<T: Scalar>(
    a: &mut Array<T>,
    index: &[IndexSpec],
    values: &Array<T>,
) -> Result<()> {
    if !a.writable {
        return Err(Error::ReadOnly);
    }
    let prepared = prepare_index(a.shape(), index)?;
    if prepared.has_fancy() {
        scatter_fancy_array(a, &prepared, values)
    } else {
        scatter_basic_array(a, &prepared, values)
    }
}

impl<T: Scalar> Array<T> {
    /// [`gather`] with [`IndexSpec`].
    #[inline]
    pub fn gather(&self, index: &[IndexSpec]) -> Result<Array<T>> {
        gather(self, index)
    }

    /// [`scatter`] with [`IndexSpec`].
    #[inline]
    pub fn scatter(&mut self, index: &[IndexSpec], value: T) -> Result<()> {
        scatter(self, index, value)
    }

    /// [`scatter_array`] with [`IndexSpec`].
    #[inline]
    pub fn scatter_array(
        &mut self,
        index: &[IndexSpec],
        values: &Array<T>,
    ) -> Result<()> {
        scatter_array(self, index, values)
    }
}

/// Layout of a basic-index selection (shared by gather view and scatter writes).
struct BasicViewMeta {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

fn basic_view_meta<T: Scalar>(
    a: &Array<T>,
    prepared: &PreparedIndex,
) -> Result<BasicViewMeta> {
    let mut offset = a.offset() as isize;
    let mut shape = Vec::new();
    let mut strides = Vec::new();
    let mut source_axis = 0usize;

    for entry in &prepared.entries {
        match entry {
            PreparedEntry::NewAxis => {
                shape.push(1);
                strides.push(0);
            }
            PreparedEntry::Index(idx) => {
                offset += *idx as isize * a.strides()[source_axis];
                source_axis += 1;
            }
            PreparedEntry::Slice { start, len, step } => {
                offset += *start * a.strides()[source_axis];
                shape.push(*len);
                strides.push(a.strides()[source_axis] * *step);
                source_axis += 1;
            }
            PreparedEntry::Fancy(_) => {
                return Err(Error::InvalidArgument(
                    "internal error: fancy entry on basic path".into(),
                ));
            }
        }
    }

    if offset < 0 {
        return Err(Error::InvalidArgument(
            "indexing produced a negative buffer offset".into(),
        ));
    }

    Ok(BasicViewMeta {
        shape,
        strides,
        offset: offset as usize,
    })
}

fn gather_basic<T: Scalar>(
    a: &Array<T>,
    prepared: &PreparedIndex,
) -> Result<Array<T>> {
    let meta = basic_view_meta(a, prepared)?;
    Array::from_arc_raw_parts(
        Arc::clone(&a.data),
        meta.shape,
        meta.strides,
        meta.offset,
        a.writable,
    )
}

fn scatter_basic_scalar<T: Scalar>(
    a: &mut Array<T>,
    prepared: &PreparedIndex,
    value: T,
) -> Result<()> {
    basic_view_meta(a, prepared)?;
    a.ensure_unique_storage_for_write();
    let meta = basic_view_meta(a, prepared)?;
    let data = Arc::make_mut(&mut a.data);

    if is_c_contiguous(&meta.shape, &meta.strides) {
        let size = size_of_shape(&meta.shape);
        data[meta.offset..meta.offset + size].fill(value);
        return Ok(());
    }

    for buf_idx in StrideIter::new(&meta.shape, &meta.strides, meta.offset) {
        data[buf_idx] = value;
    }
    Ok(())
}

fn scatter_basic_array<T: Scalar>(
    a: &mut Array<T>,
    prepared: &PreparedIndex,
    values: &Array<T>,
) -> Result<()> {
    let meta = basic_view_meta(a, prepared)?;
    let aligned = prepare_scatter_source(a, values, &meta.shape)?;

    a.ensure_unique_storage_for_write();
    let meta = basic_view_meta(a, prepared)?;
    let dest_c_contiguous = is_c_contiguous(&meta.shape, &meta.strides);
    let src_slice = aligned.as_c_contiguous_slice();
    let data = Arc::make_mut(&mut a.data);

    match (dest_c_contiguous, src_slice) {
        (true, Some(src)) => {
            let size = size_of_shape(&meta.shape);
            data[meta.offset..meta.offset + size].copy_from_slice(src);
        }
        (false, Some(src)) => {
            for (buf_idx, &value) in
                StrideIter::new(&meta.shape, &meta.strides, meta.offset)
                    .zip(src.iter())
            {
                data[buf_idx] = value;
            }
        }
        (true, None) => {
            let size = size_of_shape(&meta.shape);
            let dest = &mut data[meta.offset..meta.offset + size];
            for (dst, src_idx) in dest.iter_mut().zip(StrideIter::new(
                aligned.shape(),
                aligned.strides(),
                aligned.offset(),
            )) {
                *dst = aligned.data[src_idx];
            }
        }
        (false, None) => {
            let dest_it =
                StrideIter::new(&meta.shape, &meta.strides, meta.offset);
            let src_it = StrideIter::new(
                aligned.shape(),
                aligned.strides(),
                aligned.offset(),
            );
            for (dst_idx, src_idx) in dest_it.zip(src_it) {
                data[dst_idx] = aligned.data[src_idx];
            }
        }
    }
    Ok(())
}

fn gather_fancy<T: Scalar>(
    a: &Array<T>,
    prepared: &PreparedIndex,
) -> Result<Array<T>> {
    let layout = prepared.fancy.as_ref().unwrap();
    let mut out = Vec::with_capacity(size_of_shape(&layout.result_shape));
    for buf_idx in iter_fancy_source_offsets(a.strides(), a.offset(), prepared)
    {
        out.push(a.data[buf_idx]);
    }
    Array::from_vec(out, &layout.result_shape)
}

fn scatter_fancy_scalar<T: Scalar>(
    a: &mut Array<T>,
    prepared: &PreparedIndex,
    value: T,
) -> Result<()> {
    a.ensure_unique_storage_for_write();
    let strides = a.strides().to_vec();
    let base_offset = a.offset();
    let data = Arc::make_mut(&mut a.data);
    for buf_idx in iter_fancy_source_offsets(&strides, base_offset, prepared) {
        data[buf_idx] = value;
    }
    Ok(())
}

fn scatter_fancy_array<T: Scalar>(
    a: &mut Array<T>,
    prepared: &PreparedIndex,
    values: &Array<T>,
) -> Result<()> {
    let result_shape = &prepared.fancy.as_ref().unwrap().result_shape;
    let aligned = prepare_scatter_source(a, values, result_shape)?;

    a.ensure_unique_storage_for_write();
    let strides = a.strides().to_vec();
    let base_offset = a.offset();
    let data = Arc::make_mut(&mut a.data);
    let buf_idxs = iter_fancy_source_offsets(&strides, base_offset, prepared);

    if let Some(src) = aligned.as_c_contiguous_slice() {
        for (buf_idx, &value) in buf_idxs.zip(src.iter()) {
            data[buf_idx] = value;
        }
    } else {
        let src_idxs = StrideIter::new(
            aligned.shape(),
            aligned.strides(),
            aligned.offset(),
        );
        for (buf_idx, src_idx) in buf_idxs.zip(src_idxs) {
            data[buf_idx] = aligned.data[src_idx];
        }
    }
    Ok(())
}

/// Prepare an array assignment source before destination COW can detach or
/// relayout shared storage.
fn prepare_scatter_source<T: Scalar>(
    destination: &Array<T>,
    values: &Array<T>,
    target_shape: &[usize],
) -> Result<Array<T>> {
    let aligned = if destination.shares_buffer_with(values) {
        values.copy().broadcast_to(target_shape)
    } else {
        values.broadcast_to(target_shape)
    };
    aligned.map_err(|_| {
        Error::InvalidArgument(format!(
            "could not broadcast assignment from {:?} onto {:?}",
            values.shape(),
            target_shape
        ))
    })
}

/// Yields buffer offsets into the source array for each fancy-result element.
fn iter_fancy_source_offsets<'a>(
    source_strides: &'a [isize],
    base_offset: usize,
    prepared: &'a PreparedIndex,
) -> FancyOffsetIter<'a> {
    let layout = prepared.fancy.as_ref().unwrap();
    let remaining = size_of_shape(&layout.result_shape);
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
    fn current_offset(&self) -> usize {
        let mut offset = self.base_offset as isize;
        let mut source_axis = 0usize;
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
                PreparedEntry::Fancy(fancy) => {
                    let idx = read_fancy_usize(fancy, fancy_coords);
                    offset += idx as isize * self.source_strides[source_axis];
                    source_axis += 1;
                }
            }
        }

        debug_assert!(
            offset >= 0,
            "fancy indexing produced a negative buffer offset"
        );
        offset as usize
    }
}

fn read_fancy_usize(arr: &Array<i64>, indices: &[usize]) -> usize {
    debug_assert_eq!(indices.len(), arr.ndim());
    let buf = buffer_index(indices, arr.strides(), arr.offset());
    arr.data[buf] as usize
}
