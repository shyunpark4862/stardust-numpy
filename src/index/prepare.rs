//! Turn a raw index tuple into a shape-aware, executable plan.
//!
//! Preparation expands ellipsis, converts boolean masks to integer coordinate
//! arrays, normalizes negative indices, resolves slices, and broadcasts fancy
//! index arrays to a common shape. The result drives both basic views and
//! fancy copies in gather/scatter.

use crate::array::Array;
use crate::broadcast::broadcast_shapes;
use crate::error::{Error, Result};
use crate::index::bounds::{
    advance_multi_index, normalize_element_index, resolve_slice, slice_length,
};
use crate::index::spec::IndexSpec;
use crate::shape::validate_shape_geometry;

/// One resolved index slot after ellipsis and boolean expansion.
///
/// Ellipsis and boolean masks are gone. Integer and slice bounds are resolved
/// against the source shape. Fancy arrays are broadcast; output layout lives in
/// [`PreparedIndex::fancy`].
#[derive(Clone, Debug)]
pub(crate) enum PreparedEntry {
    /// Non-negative element index along one source axis.
    Index(usize),
    /// Resolved slice: first selected index, output length, and step.
    Slice {
        start: isize,
        len: usize,
        step: isize,
    },
    /// Length-1 axis inserted by `NewAxis`.
    NewAxis,
    /// Fancy integer index for one source axis (values in `0..axis_len`).
    IntegerArray(Array<i64>),
}

/// Cached shape metadata for a fancy-indexed result.
///
/// NumPy places basic `NewAxis` / `Slice` dimensions before or after the
/// broadcast fancy block depending on whether fancy slots are adjacent in the
/// index tuple. This struct records that ordering for gather/scatter.
#[derive(Clone, Debug)]
pub(crate) struct FancyLayout {
    /// Full output shape (basic dims + fancy dims; order depends on adjacency).
    pub(crate) result_shape: Vec<usize>,
    /// First output axis that belongs to the fancy block.
    pub(crate) fancy_axis_start: usize,
    /// Number of output axes contributed by the broadcast fancy shape.
    pub(crate) fancy_axis_len: usize,
    /// Maps each index slot to its output axis for basic `NewAxis` / `Slice`.
    pub(crate) slot_to_result_axis: Vec<Option<usize>>,
}

/// Fully prepared index ready for gather or scatter.
///
/// Basic indexing uses [`PreparedEntry`] alone to build a view. Fancy
/// indexing additionally stores [`FancyLayout`] describing the copied
/// result shape and axis placement.
pub(crate) struct PreparedIndex {
    pub(crate) entries: Vec<PreparedEntry>,
    /// Present when any fancy integer arrays appear in the index.
    pub(crate) fancy: Option<FancyLayout>,
}

impl PreparedIndex {
    /// Whether this index requires fancy (copying) gather/scatter.
    ///
    /// Basic indices (`Index`, `Slice`, `NewAxis`) can return a shared view.
    /// Any [`PreparedEntry::IntegerArray`] forces a materializing copy because
    /// source elements are gathered in broadcast fancy order.
    ///
    /// # Arguments
    ///
    /// * `self` - A fully prepared index after ellipsis and boolean expansion.
    ///
    /// # Returns
    ///
    /// `true` when [`PreparedIndex::fancy`] is present.
    pub(crate) fn has_fancy(&self) -> bool {
        self.fancy.is_some()
    }
}

/// Expand, normalize, and broadcast an index against `shape`.
///
/// Runs the full NumPy-style preparation pipeline:
///
/// 1. **Ellipsis** — [`expand_ellipsis`] replaces `...` with full slices for
///    every unconsumed axis, or appends trailing full slices when ellipsis is
///    absent.
/// 2. **Boolean masks** — [`expand_boolean_masks`] turns each `BoolArray`
///    into one integer coordinate array per mask axis (length = number of
///    `True` entries).
/// 3. **Resolution** — [`resolve_entries`] normalizes integers, resolves
///    slices, and bounds-checks fancy arrays against the source shape.
/// 4. **Fancy broadcast** — all [`PreparedEntry::IntegerArray`] operands are
///    broadcast to a common shape; [`build_fancy_layout`] records output axis
///    order (adjacent vs separated fancy slots).
///
/// # Arguments
///
/// * `shape` — source array shape used for bound checks and ellipsis fill
/// * `index` — raw index tuple from the caller
///
/// # Returns
///
/// A [`PreparedIndex`] ready for [`crate::index::gather`] or scatter.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — duplicate ellipsis, too many index slots,
///   slice/broadcast/shape validation failures
/// * [`Error::IndexOutOfBounds`] — integer or fancy index out of range
pub(crate) fn prepare_index(
    shape: &[usize],
    index: &[IndexSpec],
) -> Result<PreparedIndex> {
    let index = expand_ellipsis(shape, index)?;
    let index = expand_boolean_masks(&index)?;
    let mut entries = resolve_entries(shape, &index)?;

    let fancy_shapes: Vec<&[usize]> = entries
        .iter()
        .filter_map(|e| match e {
            PreparedEntry::IntegerArray(arr) => Some(arr.shape()),
            _ => None,
        })
        .collect();

    if fancy_shapes.is_empty() {
        return Ok(PreparedIndex {
            entries,
            fancy: None,
        });
    }

    // NumPy broadcasts all fancy arrays together before indexing.
    let fancy_shape = broadcast_shapes(&fancy_shapes)?;
    for entry in &mut entries {
        if let PreparedEntry::IntegerArray(arr) = entry {
            *arr = arr.broadcast_to(&fancy_shape)?;
        }
    }

    let fancy = build_fancy_layout(&entries, &fancy_shape);
    validate_shape_geometry(&fancy.result_shape)?;
    Ok(PreparedIndex {
        entries,
        fancy: Some(fancy),
    })
}

/// Count source axes consumed by one index slot before ellipsis expansion.
///
/// `NewAxis` and `Ellipsis` consume zero axes; boolean masks consume one axis
/// per mask dimension; all other slots consume exactly one source axis.
///
/// # Arguments
///
/// * `item` — one slot from the raw index tuple
///
/// # Returns
///
/// Number of source axes this slot advances the source-axis cursor.
fn axes_consumed(item: &IndexSpec) -> usize {
    match item {
        IndexSpec::NewAxis | IndexSpec::Ellipsis => 0,
        IndexSpec::BoolArray(m) => m.ndim(),
        IndexSpec::Index(_)
        | IndexSpec::Slice { .. }
        | IndexSpec::IntegerArray(_) => 1,
    }
}

/// Replace ellipsis and append implicit trailing slices.
///
/// At most one `Ellipsis` is allowed. The number of missing axes is
/// `shape.len() - sum(axes_consumed)` for non-ellipsis slots. When ellipsis
/// is present it expands inline to that many full slices; otherwise missing
/// axes become trailing full slices at the end of the tuple.
///
/// # Arguments
///
/// * `shape` — source array shape
/// * `index` — raw index tuple (may contain `Ellipsis`)
///
/// # Returns
///
/// An index tuple with ellipsis removed and every source axis accounted for.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — more than one ellipsis, or too many index
///   slots for `shape.len()`
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
                    // `...` expands to one full slice per missing axis.
                    for _ in 0..missing {
                        out.push(IndexSpec::full());
                    }
                }
                other => out.push(other.clone()),
            }
        }
    } else {
        out.extend(index.iter().cloned());
        // Trailing axes omitted from the index tuple become full slices.
        for _ in 0..missing {
            out.push(IndexSpec::full());
        }
    }
    Ok(out)
}

/// Replace boolean masks with per-axis integer coordinate arrays.
///
/// Each `BoolArray` of rank `r` becomes `r` length-`n` integer arrays, where
/// `n` is the count of `True` entries. Coordinates are emitted in C-order
/// over the mask. Non-boolean slots pass through unchanged.
///
/// # Arguments
///
/// * `index` — index tuple after ellipsis expansion
///
/// # Returns
///
/// Index tuple with only non-boolean slot kinds (plus any remaining slots
/// that were never boolean).
///
/// # Errors
///
/// Propagates allocation or traversal errors from coordinate extraction.
fn expand_boolean_masks(index: &[IndexSpec]) -> Result<Vec<IndexSpec>> {
    let mut out = Vec::new();

    for item in index {
        match item {
            IndexSpec::NewAxis => {
                out.push(IndexSpec::NewAxis);
            }
            IndexSpec::Index(i) => {
                out.push(IndexSpec::Index(*i));
            }
            IndexSpec::Slice { start, stop, step } => {
                out.push(IndexSpec::Slice {
                    start: *start,
                    stop: *stop,
                    step: *step,
                });
            }
            IndexSpec::IntegerArray(arr) => {
                out.push(IndexSpec::IntegerArray(arr.clone()));
            }
            IndexSpec::BoolArray(mask) => {
                // One coordinate array per mask axis, all length = True count.
                let coords = boolean_mask_to_integer_arrays(mask)?;
                out.extend(coords.into_iter().map(IndexSpec::IntegerArray));
            }
            IndexSpec::Ellipsis => {}
        }
    }

    Ok(out)
}

/// Resolve integer and slice bounds against `shape`.
///
/// Walks the index tuple in lockstep with source axes. Integer and fancy
/// entries are normalized to non-negative in-bounds values. Slices become
/// `(start, len, step)` triples. `NewAxis` does not advance the source axis.
///
/// # Arguments
///
/// * `shape` — source array shape
/// * `index` — fully expanded index tuple (no ellipsis or boolean masks)
///
/// # Returns
///
/// One [`PreparedEntry`] per index slot.
///
/// # Errors
///
/// * [`Error::IndexOutOfBounds`] — integer or fancy value out of range
/// * [`Error::InvalidArgument`] — slice resolution failures
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
                let idx = normalize_element_index(*i, shape[source_axis])?;
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
                out.push(PreparedEntry::IntegerArray(normalized));
                source_axis += 1;
            }
            IndexSpec::BoolArray(_) | IndexSpec::Ellipsis => {}
        }
    }

    Ok(out)
}

/// Bounds-check and normalize every element of a fancy index array.
///
/// Each raw `i64` is mapped through [`normalize_element_index`] for the
/// given axis length. Layout (shape/strides) is preserved.
///
/// # Arguments
///
/// * `array` — fancy integer index array for one source axis
/// * `axis_len` — length of that source axis
///
/// # Returns
///
/// A new array with the same shape and normalized index values.
///
/// # Errors
///
/// * [`Error::IndexOutOfBounds`] — any element falls outside the axis
fn normalize_fancy_array(
    array: &Array<i64>,
    axis_len: usize,
) -> Result<Array<i64>> {
    let mut output = Vec::with_capacity(array.size());
    if let Some(values) = array.as_c_contiguous_slice() {
        for &raw in values {
            output.push(normalize_element_index(raw, axis_len)? as i64);
        }
    } else {
        use crate::traversal::StrideIter;
        for buffer_index in
            StrideIter::new(array.shape(), array.strides(), array.offset())
        {
            output.push(normalize_element_index(
                array.data[buffer_index],
                axis_len,
            )? as i64);
        }
    }
    Array::from_vec(output, array.shape())
}

/// Extract C-order `(axis0, axis1, …)` coordinates for every `True` mask entry.
///
/// Produces one 1-D integer array per mask axis, each of length equal to the
/// number of selected elements. All arrays share the same length.
///
/// # Arguments
///
/// * `mask` — boolean index array (any rank)
///
/// # Returns
///
/// One coordinate array per mask axis, in axis order.
///
/// # Errors
///
/// Propagates allocation errors from [`Array::from_vec`].
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
        use crate::traversal::StrideIter;
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
        .map(|c| Array::from_vec(c, &[n]))
        .collect()
}

/// True when all fancy slots form one contiguous block in the index tuple.
///
/// NumPy uses adjacent fancy slots for "inner" indexing (basic dims before
/// and after the fancy block) and separated slots for "outer" indexing (fancy
/// block leads the result shape).
///
/// # Arguments
///
/// * `entries` — prepared index slots
///
/// # Returns
///
/// `true` when there are no fancy slots, or every slot between the first and
/// last fancy slot is also fancy.
fn fancy_slots_adjacent(entries: &[PreparedEntry]) -> bool {
    let mut first = None;
    let mut last = None;
    for (slot, entry) in entries.iter().enumerate() {
        if matches!(entry, PreparedEntry::IntegerArray(_)) {
            if first.is_none() {
                first = Some(slot);
            }
            last = Some(slot);
        }
    }
    let (Some(first), Some(last)) = (first, last) else {
        return true;
    };
    (first..=last)
        .all(|slot| matches!(entries[slot], PreparedEntry::IntegerArray(_)))
}

/// Compute output shape and axis maps for a broadcast fancy index block.
///
/// Collects basic `NewAxis` / `Slice` dimensions before and after the first
/// fancy slot, then orders `[before, fancy, after]` or `[fancy, before,
/// after]` depending on [`fancy_slots_adjacent`].
///
/// # Arguments
///
/// * `entries` — prepared entries containing at least one fancy array
/// * `fancy_shape` — broadcast shape shared by all fancy operands
///
/// # Returns
///
/// [`FancyLayout`] describing the gather/scatter result geometry.
///
/// # Panics
///
/// Panics if `entries` contains no [`PreparedEntry::IntegerArray`] (caller
/// must only invoke this after detecting fancy indices).
fn build_fancy_layout(
    entries: &[PreparedEntry],
    fancy_shape: &[usize],
) -> FancyLayout {
    let first_fancy = entries
        .iter()
        .position(|e| matches!(e, PreparedEntry::IntegerArray(_)))
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
            PreparedEntry::Index(_) | PreparedEntry::IntegerArray(_) => {}
        }
    }

    // Adjacent fancy indices keep basic dims before/after; separated ones
    // place the fancy block first (NumPy "outer" vs "inner" ordering).
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
