//! Axis / slice bound helpers used by indexing prepare.

use crate::error::{Error, Result};

/// Number of elements produced by `range(start, stop, step)`.
///
/// Matches Python/`slice` length semantics. `step` must be non-zero.
pub(crate) fn slice_length(
    start: isize,
    stop: isize,
    step: isize,
) -> Result<usize> {
    if step == 0 {
        return Err(Error::InvalidArgument("slice step cannot be zero".into()));
    }
    if (stop - start) * step <= 0 {
        return Ok(0);
    }
    let numer = stop - start + step - if step > 0 { 1 } else { -1 };
    Ok((numer / step) as usize)
}

/// Normalize a possibly-negative index into `0..axis_len`.
///
/// This is an **element** index along an axis of length `axis_len`
/// (not an axis number in `0..ndim`).
pub(crate) fn normalize_index(index: i64, axis_len: usize) -> Result<usize> {
    let len = axis_len as i64;
    let mut idx = index;
    if idx < 0 {
        idx += len;
    }
    if idx < 0 || idx >= len {
        return Err(Error::IndexOutOfBounds {
            index: idx as isize,
            axis_len,
        });
    }
    Ok(idx as usize)
}

/// Advance a C-order multi-index in place (odometer).
pub(crate) fn advance_multi_index(indices: &mut [usize], shape: &[usize]) {
    if indices.is_empty() {
        return;
    }
    for axis in (0..indices.len()).rev() {
        indices[axis] += 1;
        if indices[axis] < shape[axis] {
            return;
        }
        indices[axis] = 0;
    }
}

/// Resolve a Python-style slice against an axis of length `axis_len`.
///
/// Returns `(start, stop, step)` suitable for [`slice_length`] and for
/// computing the first selected index / stride multiplier (like
/// `slice.indices` in Python).
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
    let len = axis_len as i64;

    let (start, stop) = if step > 0 {
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

    Ok((start as isize, stop as isize, step as isize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_length_basic() {
        assert_eq!(slice_length(0, 5, 1).unwrap(), 5);
        assert_eq!(slice_length(3, 1, 1).unwrap(), 0);
        assert!(slice_length(0, 5, 0).is_err());
    }

    #[test]
    fn normalize_negative() {
        assert_eq!(normalize_index(-1, 3).unwrap(), 2);
        assert!(normalize_index(3, 3).is_err());
    }

    #[test]
    fn resolve_full_and_reverse() {
        assert_eq!(resolve_slice(None, None, None, 5).unwrap(), (0, 5, 1));
        assert_eq!(
            resolve_slice(None, None, Some(-1), 4).unwrap(),
            (3, -1, -1)
        );
    }
}
