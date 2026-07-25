//! Low-level helpers for slice bounds and C-order coordinate stepping.
//!
//! These functions mirror Python/NumPy slice semantics: negative indices count
//! from the end, omitted slice bounds pick axis defaults, and a zero step is
//! rejected. They are used while turning [`IndexSpec`] into prepared entries.

use crate::error::{Error, Result};

/// Count elements selected by a Python-style slice.
///
/// Computes the length of `range(start, stop, step)` the way NumPy does.
/// Empty ranges return zero; a zero `step` is rejected.
///
/// # Arguments
///
/// * `start` — resolved slice start (may be negative before resolution)
/// * `stop` — resolved slice stop (exclusive, like Python)
/// * `step` — slice step; must be non-zero
///
/// # Returns
///
/// The number of indices visited by the slice walk.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — `step == 0` or the length overflows
///   `usize`
pub(crate) fn slice_length(
    start: isize,
    stop: isize,
    step: isize,
) -> Result<usize> {
    if step == 0 {
        return Err(Error::InvalidArgument("slice step cannot be zero".into()));
    }
    let start = start as i128;
    let stop = stop as i128;
    let step = step as i128;
    // Empty when start/stop do not advance in the step direction.
    if (stop - start) * step <= 0 {
        return Ok(0);
    }
    let numer = stop - start + step - if step > 0 { 1 } else { -1 };
    usize::try_from(numer / step).map_err(|_| {
        Error::InvalidArgument("slice length overflows usize".into())
    })
}

/// Map a possibly-negative element index into `0..axis_len`.
///
/// Negative values count backward from the end of the axis, matching NumPy.
/// This is an index **along one axis**, not an axis number in `0..ndim`.
///
/// # Arguments
///
/// * `index` — raw element index (may be negative)
/// * `axis_len` — length of the axis being indexed
///
/// # Returns
///
/// A non-negative index in `0..axis_len`.
///
/// # Errors
///
/// * [`Error::IndexOutOfBounds`] — the normalized index falls outside the
///   axis
pub(crate) fn normalize_element_index(
    index: i64,
    axis_len: usize,
) -> Result<usize> {
    let len = axis_len as i64;
    let mut idx = index;
    if idx < 0 {
        idx += len;
    }
    if idx < 0 || idx >= len {
        return Err(Error::IndexOutOfBounds { index, axis_len });
    }
    Ok(idx as usize)
}

/// Advance a C-order multi-index in place, like an odometer.
///
/// Increments the rightmost axis first and carries left on overflow. When
/// every axis wraps, the index becomes all zeros (the caller typically stops
/// before that sentinel state).
///
/// # Arguments
///
/// * `indices` — current coordinate, updated in place
/// * `shape` — axis lengths bounding each coordinate component
///
/// # Returns
///
/// Nothing; `indices` is advanced by one C-order step.
pub(crate) fn advance_multi_index(indices: &mut [usize], shape: &[usize]) {
    if indices.is_empty() {
        return;
    }
    // Rightmost axis increments first; carry left on overflow.
    for axis in (0..indices.len()).rev() {
        indices[axis] += 1;
        if indices[axis] < shape[axis] {
            return;
        }
        indices[axis] = 0;
    }
}

/// Resolve slice bounds against an axis of length `axis_len`.
///
/// Applies NumPy/Python `slice.indices(length)` rules: omitted bounds pick
/// axis defaults that depend on step sign, and negative bounds count from
/// the end. The triple is suitable for [`slice_length`] and offset math.
///
/// # Arguments
///
/// * `start` — optional raw slice start (`None` → axis default)
/// * `stop` — optional raw slice stop (`None` → axis default)
/// * `step` — optional slice step (`None` → `1`)
/// * `axis_len` — length of the axis being sliced
///
/// # Returns
///
/// `(start, stop, step)` as signed integers ready for length and stride
/// computation.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — zero step, or any bound does not fit in
///   `isize`
pub(crate) fn resolve_slice(
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    axis_len: usize,
) -> Result<(isize, isize, isize)> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(Error::InvalidArgument("slice step cannot be zero".into()));
    }
    let len = i64::try_from(axis_len).map_err(|_| {
        Error::InvalidArgument("axis length exceeds i64 range".into())
    })?;

    let (start, stop) = if step > 0 {
        // Forward slice: default start=0, default stop=axis length.
        let start = match start {
            None => 0,
            Some(s) if s < 0 => (s + len).max(0),
            Some(s) => s.min(len),
        };
        let stop = match stop {
            None => len,
            Some(s) if s < 0 => (s + len).max(0),
            Some(s) => s.min(len),
        };
        (start, stop)
    } else {
        // Negative step: defaults mirror Python's reverse-slice rules.
        let start = match start {
            None => len - 1,
            Some(s) if s < 0 => (s + len).max(-1),
            Some(s) => s.min(len - 1),
        };
        let stop = match stop {
            None => -1,
            Some(s) if s < 0 => (s + len).max(-1),
            Some(s) => s.min(len - 1),
        };
        (start, stop)
    };

    Ok((
        isize::try_from(start).map_err(|_| {
            Error::InvalidArgument("slice start exceeds isize range".into())
        })?,
        isize::try_from(stop).map_err(|_| {
            Error::InvalidArgument("slice stop exceeds isize range".into())
        })?,
        isize::try_from(step).map_err(|_| {
            Error::InvalidArgument("slice step exceeds isize range".into())
        })?,
    ))
}
