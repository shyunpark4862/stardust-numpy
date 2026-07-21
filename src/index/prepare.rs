//! Expand / broadcast index expressions into a prepared form.

use crate::array::Array;
use crate::broadcast::broadcast_shapes;
use crate::error::{Error, Result};
use crate::index::bounds::{
    advance_multi_index, normalize_index, resolve_slice, slice_length,
};
use crate::index::spec::IndexSpec;

/// Index entry after ellipsis/bool expansion (no `Ellipsis` / `BoolArray`).
///
/// Integer and slice bounds are already resolved against the source shape.
/// Fancy arrays are broadcast; result axis layout is cached in
/// [`PreparedIndex::fancy`].
#[derive(Clone, Debug)]
pub(crate) enum PreparedEntry {
    /// Normalized non-negative axis index.
    Index(usize),
    /// Resolved slice: first index, output length, and step.
    Slice {
        start: isize,
        len: usize,
        step: isize,
    },
    NewAxis,
    /// Integer fancy index for one source axis (broadcast; values already in
    /// `0..axis_len`).
    Fancy(Array<i64>),
}

/// Cached layout of a fancy-indexed gather/scatter result.
#[derive(Clone, Debug)]
pub(crate) struct FancyLayout {
    /// Full result shape (basic dims + fancy dims, order depends on adjacency).
    pub result_shape: Vec<usize>,
    /// Start axis of the fancy block inside [`Self::result_shape`].
    pub fancy_axis_start: usize,
    /// Number of fancy axes (broadcast fancy ndim).
    pub fancy_axis_len: usize,
    /// Index-entry slot → result axis for basic `NewAxis` / `Slice` (else `None`).
    pub slot_to_result_axis: Vec<Option<usize>>,
}

/// Fully prepared index ready for basic or fancy gather/scatter.
pub(crate) struct PreparedIndex {
    pub entries: Vec<PreparedEntry>,
    /// `Some` when any fancy entries are present.
    pub fancy: Option<FancyLayout>,
}

impl PreparedIndex {
    pub fn has_fancy(&self) -> bool {
        self.fancy.is_some()
    }
}

pub(crate) fn prepare_index(
    shape: &[usize],
    index: &[IndexSpec],
) -> Result<PreparedIndex> {
    let index = expand_ellipsis(shape, index)?;
    let index = expand_boolean_masks(shape, &index)?;
    let mut entries = resolve_entries(shape, &index)?;

    let fancy_shapes: Vec<&[usize]> = entries
        .iter()
        .filter_map(|e| match e {
            PreparedEntry::Fancy(arr) => Some(arr.shape()),
            _ => None,
        })
        .collect();

    if fancy_shapes.is_empty() {
        return Ok(PreparedIndex {
            entries,
            fancy: None,
        });
    }

    let fancy_shape = broadcast_shapes(&fancy_shapes)?;
    for entry in &mut entries {
        if let PreparedEntry::Fancy(arr) = entry {
            *arr = arr.broadcast_to(&fancy_shape)?;
        }
    }

    let fancy = build_fancy_layout(&entries, &fancy_shape);
    Ok(PreparedIndex {
        entries,
        fancy: Some(fancy),
    })
}

fn axes_consumed(item: &IndexSpec) -> usize {
    match item {
        IndexSpec::NewAxis | IndexSpec::Ellipsis => 0,
        IndexSpec::BoolArray(m) => m.ndim(),
        IndexSpec::Index(_)
        | IndexSpec::Slice { .. }
        | IndexSpec::IntegerArray(_) => 1,
    }
}

fn expand_ellipsis(
    shape: &[usize],
    index: &[IndexSpec],
) -> Result<Vec<IndexSpec>> {
    let ellipsis_count = index
        .iter()
        .filter(|i| matches!(i, IndexSpec::Ellipsis))
        .count();
    if ellipsis_count > 1 {
        return Err(Error::InvalidArgument(
            "an index can only have a single ellipsis".into(),
        ));
    }

    let used: usize = index
        .iter()
        .filter(|i| !matches!(i, IndexSpec::Ellipsis))
        .map(axes_consumed)
        .sum();

    if used > shape.len() {
        return Err(Error::InvalidArgument(format!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            shape.len(),
            used
        )));
    }

    let missing = shape.len() - used;
    let mut out = Vec::new();
    if ellipsis_count == 1 {
        for item in index {
            match item {
                IndexSpec::Ellipsis => {
                    for _ in 0..missing {
                        out.push(IndexSpec::full());
                    }
                }
                other => out.push(other.clone()),
            }
        }
    } else {
        out.extend(index.iter().cloned());
        for _ in 0..missing {
            out.push(IndexSpec::full());
        }
    }
    Ok(out)
}

/// Replace boolean masks with per-axis integer coordinate arrays.
///
/// Does not resolve negative indices or slices; that is [`resolve_entries`].
fn expand_boolean_masks(
    shape: &[usize],
    index: &[IndexSpec],
) -> Result<Vec<IndexSpec>> {
    let mut out = Vec::new();
    let mut source_axis = 0usize;

    for item in index {
        match item {
            IndexSpec::NewAxis => {
                out.push(IndexSpec::NewAxis);
            }
            IndexSpec::Index(i) => {
                out.push(IndexSpec::Index(*i));
                source_axis += 1;
            }
            IndexSpec::Slice { start, stop, step } => {
                out.push(IndexSpec::Slice {
                    start: *start,
                    stop: *stop,
                    step: *step,
                });
                source_axis += 1;
            }
            IndexSpec::IntegerArray(arr) => {
                out.push(IndexSpec::IntegerArray(arr.clone()));
                source_axis += 1;
            }
            IndexSpec::BoolArray(mask) => {
                let end = source_axis + mask.ndim();
                if end > shape.len() {
                    return Err(Error::InvalidArgument(format!(
                        "too many indices for array: array is {}-dimensional",
                        shape.len()
                    )));
                }
                let expected = &shape[source_axis..end];
                if mask.shape() != expected {
                    return Err(Error::InvalidArgument(format!(
                        "boolean index did not match indexed array along dimension; got {:?}, expected {:?}",
                        mask.shape(),
                        expected
                    )));
                }
                let coords = boolean_mask_to_integer_arrays(mask)?;
                out.extend(coords.into_iter().map(IndexSpec::IntegerArray));
                source_axis += mask.ndim();
            }
            IndexSpec::Ellipsis => {
                return Err(Error::InvalidArgument(
                    "internal error: ellipsis should have been expanded".into(),
                ));
            }
        }
    }

    if source_axis != shape.len() {
        return Err(Error::InvalidArgument(format!(
            "index does not cover all axes: covered {source_axis}, ndim {}",
            shape.len()
        )));
    }

    Ok(out)
}

/// Resolve integer / slice bounds against `shape` into [`PreparedEntry`].
///
/// Input must already have ellipsis and boolean masks expanded.
/// Fancy arrays are normalized here; broadcast to a common shape happens in
/// [`prepare_index`].
fn resolve_entries(
    shape: &[usize],
    index: &[IndexSpec],
) -> Result<Vec<PreparedEntry>> {
    let mut out = Vec::new();
    let mut source_axis = 0usize;

    for item in index {
        match item {
            IndexSpec::NewAxis => {
                out.push(PreparedEntry::NewAxis);
            }
            IndexSpec::Index(i) => {
                let idx = normalize_index(*i, shape[source_axis])?;
                out.push(PreparedEntry::Index(idx));
                source_axis += 1;
            }
            IndexSpec::Slice { start, stop, step } => {
                let (s0, s1, st) =
                    resolve_slice(*start, *stop, *step, shape[source_axis])?;
                let len = slice_length(s0, s1, st)?;
                out.push(PreparedEntry::Slice {
                    start: s0,
                    len,
                    step: st,
                });
                source_axis += 1;
            }
            IndexSpec::IntegerArray(arr) => {
                let normalized =
                    normalize_fancy_array(arr, shape[source_axis])?;
                out.push(PreparedEntry::Fancy(normalized));
                source_axis += 1;
            }
            IndexSpec::BoolArray(_) | IndexSpec::Ellipsis => {
                return Err(Error::InvalidArgument(
                    "internal error: index must be expanded before resolve"
                        .into(),
                ));
            }
        }
    }

    if source_axis != shape.len() {
        return Err(Error::InvalidArgument(format!(
            "index does not cover all axes: covered {source_axis}, ndim {}",
            shape.len()
        )));
    }

    Ok(out)
}

fn normalize_fancy_array(
    arr: &Array<i64>,
    axis_len: usize,
) -> Result<Array<i64>> {
    let mut out = Vec::with_capacity(arr.size());
    if let Some(xs) = arr.as_c_contiguous_slice() {
        for &raw in xs {
            out.push(normalize_index(raw, axis_len)? as i64);
        }
    } else {
        use crate::stride_iter::StrideIter;
        for buf_idx in StrideIter::new(arr.shape(), arr.strides(), arr.offset())
        {
            out.push(normalize_index(arr.data[buf_idx], axis_len)? as i64);
        }
    }
    Array::from_vec(out, arr.shape())
}

fn boolean_mask_to_integer_arrays(
    mask: &Array<bool>,
) -> Result<Vec<Array<i64>>> {
    let ndim = mask.ndim();
    let mut coords: Vec<Vec<i64>> = vec![Vec::new(); ndim];

    if let Some(xs) = mask.as_c_contiguous_slice() {
        let mut indices = vec![0usize; ndim];
        for &value in xs {
            if value {
                for (axis, &c) in indices.iter().enumerate() {
                    coords[axis].push(c as i64);
                }
            }
            advance_multi_index(&mut indices, mask.shape());
        }
    } else {
        use crate::stride_iter::StrideIter;
        StrideIter::new(mask.shape(), mask.strides(), mask.offset()).for_each(
            |buf_idx, indices| {
                if mask.data[buf_idx] {
                    for (axis, &c) in indices.iter().enumerate() {
                        coords[axis].push(c as i64);
                    }
                }
            },
        );
    }

    let n = coords.first().map(|c| c.len()).unwrap_or(0);
    coords
        .into_iter()
        .map(|c| {
            debug_assert_eq!(c.len(), n);
            Array::from_vec(c, &[n])
        })
        .collect()
}

/// Whether all fancy slots form a contiguous block of slots.
fn fancy_slots_adjacent(entries: &[PreparedEntry]) -> bool {
    let mut first = None;
    let mut last = None;
    for (slot, entry) in entries.iter().enumerate() {
        if matches!(entry, PreparedEntry::Fancy(_)) {
            if first.is_none() {
                first = Some(slot);
            }
            last = Some(slot);
        }
    }
    let (Some(first), Some(last)) = (first, last) else {
        return true;
    };
    (first..=last).all(|slot| matches!(entries[slot], PreparedEntry::Fancy(_)))
}

fn build_fancy_layout(
    entries: &[PreparedEntry],
    fancy_shape: &[usize],
) -> FancyLayout {
    let first_fancy = entries
        .iter()
        .position(|e| matches!(e, PreparedEntry::Fancy(_)))
        .expect("build_fancy_layout requires at least one Fancy entry");
    let adjacent = fancy_slots_adjacent(entries);
    let fancy_axis_len = fancy_shape.len();

    let mut basic_before_dims = Vec::new();
    let mut basic_after_dims = Vec::new();
    let mut basic_before_slots = Vec::new();
    let mut basic_after_slots = Vec::new();

    for (slot, entry) in entries.iter().enumerate() {
        let (dims, slots) = if slot < first_fancy {
            (&mut basic_before_dims, &mut basic_before_slots)
        } else {
            (&mut basic_after_dims, &mut basic_after_slots)
        };
        match entry {
            PreparedEntry::NewAxis => {
                dims.push(1);
                slots.push(slot);
            }
            PreparedEntry::Slice { len, .. } => {
                dims.push(*len);
                slots.push(slot);
            }
            PreparedEntry::Index(_) | PreparedEntry::Fancy(_) => {}
        }
    }

    let (fancy_axis_start, before_axis_start, after_axis_start) = if adjacent {
        (
            basic_before_slots.len(),
            0,
            basic_before_slots.len() + fancy_axis_len,
        )
    } else {
        (0, fancy_axis_len, fancy_axis_len + basic_before_slots.len())
    };

    let mut slot_to_result_axis = vec![None; entries.len()];
    for (i, &slot) in basic_before_slots.iter().enumerate() {
        slot_to_result_axis[slot] = Some(before_axis_start + i);
    }
    for (i, &slot) in basic_after_slots.iter().enumerate() {
        slot_to_result_axis[slot] = Some(after_axis_start + i);
    }

    let result_shape = if adjacent {
        [&basic_before_dims[..], fancy_shape, &basic_after_dims[..]].concat()
    } else {
        [fancy_shape, &basic_before_dims[..], &basic_after_dims[..]].concat()
    };

    FancyLayout {
        result_shape,
        fancy_axis_start,
        fancy_axis_len,
        slot_to_result_axis,
    }
}
