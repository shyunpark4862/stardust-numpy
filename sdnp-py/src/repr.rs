//! NumPy-style string formatting for `Array.__repr__`.
//!
//! Builds a nested bracket display with dtype annotation. Long 1-D arrays
//! are truncated with `...` so interactive sessions stay readable. This is
//! presentation only; it does not affect array semantics.

use sdnp::Complex64;

use crate::inner::ArrayInner;

/// Maximum 1-D elements printed before ellipsis truncation.
const MAX_ITEMS: usize = 1000;

/// Elements shown at each end when truncating a long 1-D array.
const EDGE_ITEMS: usize = 3;

/// Format `array([...], dtype=...)` for the given storage.
///
/// Entry point for NumPy-style `__repr__`. Delegates nested bracket layout
/// to [`format_body`] and appends the dtype name annotation.
///
/// # Arguments
///
/// * `inner` - Typed storage to display.
///
/// # Returns
///
/// A complete `array(...)` display string.
///
/// # Errors
///
/// * Propagates gather/scalar formatting failures from nested slices.
pub fn array_repr(inner: &ArrayInner) -> pyo3::PyResult<String> {
    let dtype = inner.dtype().name();
    let body = format_body(inner, inner.shape(), 0)?;
    Ok(format!("array({body}, dtype={dtype})"))
}

/// Recursively format nested brackets for multidimensional arrays.
///
/// Walks axis 0, indenting nested rows to mimic NumPy's multiline layout.
///
/// # Arguments
///
/// * `inner` - Typed storage at the current depth.
/// * `shape` - Remaining shape suffix from this depth onward.
/// * `depth` - Nesting depth (controls newline indentation).
///
/// # Returns
///
/// A bracket string for this sub-array.
///
/// # Errors
///
/// * Propagates gather or scalar formatting failures.
fn format_body(
    inner: &ArrayInner,
    shape: &[usize],
    depth: usize,
) -> pyo3::PyResult<String> {
    if shape.is_empty() {
        return Ok(format_scalar(inner.item_scalar()?));
    }
    if shape.len() == 1 {
        return format_1d(inner);
    }
    let n = shape[0];
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let sub = slice_index(inner, i)?;
        parts.push(format_body(&sub, &shape[1..], depth + 1)?);
    }
    Ok(format!(
        "[{}]",
        parts.join(",\n ".repeat(depth + 1).trim_end())
    ))
}

/// Format a 1-D row, truncating with `...` when longer than [`MAX_ITEMS`].
///
/// Shows [`EDGE_ITEMS`] elements at each end when truncating.
///
/// # Arguments
///
/// * `inner` - 1-D typed storage.
///
/// # Returns
///
/// A single-line bracket string.
///
/// # Errors
///
/// * Propagates element formatting failures.
fn format_1d(inner: &ArrayInner) -> pyo3::PyResult<String> {
    let n = inner.size();
    if n <= MAX_ITEMS {
        return Ok(format!("[{}]", format_elements(inner, 0, n)?.join(", ")));
    }
    let mut parts = format_elements(inner, 0, EDGE_ITEMS)?;
    parts.push("...".to_string());
    parts.extend(format_elements(inner, n - EDGE_ITEMS, n)?);
    Ok(format!("[{}]", parts.join(", ")))
}

/// Format a contiguous flat range `[start, stop)` as strings.
///
/// Each element is rendered with dtype-appropriate formatting (`True`/`False`,
/// NumPy-like `nan`/`inf`, complex `(re+imj)` notation).
///
/// # Arguments
///
/// * `inner` - Typed storage (treated as flat for the range).
/// * `start` - Inclusive flat start index.
/// * `stop` - Exclusive flat stop index.
///
/// # Returns
///
/// A vector of formatted element strings.
///
/// # Errors
///
/// None; flat indexing is bounded by caller.
fn format_elements(
    inner: &ArrayInner,
    start: usize,
    stop: usize,
) -> pyo3::PyResult<Vec<String>> {
    let mut out = Vec::new();
    match inner {
        ArrayInner::Bool(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(if *v { "True".into() } else { "False".into() });
            }
        }
        ArrayInner::I64(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(v.to_string());
            }
        }
        ArrayInner::F64(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(format_float(*v));
            }
        }
        ArrayInner::C64(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(format_complex(*v));
            }
        }
    }
    Ok(out)
}

/// Take axis-0 index `i` via gather for recursive repr slicing.
///
/// # Arguments
///
/// * `inner` - Source typed storage.
/// * `i` - Zero-based axis-0 index.
///
/// # Returns
///
/// A sub-array with the leading dimension removed.
///
/// # Errors
///
/// * `IndexError` / `ValueError` — out-of-bounds or gather failure.
fn slice_index(inner: &ArrayInner, i: usize) -> pyo3::PyResult<ArrayInner> {
    use sdnp::{gather, IndexSpec};

    use crate::error::map_sdnp;
    let spec = vec![IndexSpec::Index(i as i64)];
    Ok(match inner {
        ArrayInner::Bool(a) => ArrayInner::Bool(map_sdnp(gather(a, &spec))?),
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(gather(a, &spec))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(gather(a, &spec))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(gather(a, &spec))?),
    })
}

/// Format one scalar for embedding in bracket notation.
///
/// # Arguments
///
/// * `s` - Typed scalar value.
///
/// # Returns
///
/// A display string for one array element.
///
/// # Errors
///
/// None.
fn format_scalar(s: crate::unwrap::PyScalar) -> String {
    use crate::unwrap::PyScalar;
    match s {
        PyScalar::Bool(v) => {
            if v {
                "True".into()
            } else {
                "False".into()
            }
        }
        PyScalar::I64(v) => v.to_string(),
        PyScalar::F64(v) => format_float(v),
        PyScalar::C64(v) => format_complex(v),
    }
}

/// Format `f64` with NumPy-like names for non-finite values.
///
/// # Arguments
///
/// * `v` - Floating-point element.
///
/// # Returns
///
/// `"nan"`, `"inf"`, `"-inf"`, or the default `Display` string.
///
/// # Errors
///
/// None.
fn format_float(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "inf".into()
        } else {
            "-inf".into()
        }
    } else {
        v.to_string()
    }
}

/// Format `Complex64` as `(re+imj)` with real-part special cases.
///
/// Purely real values use `(re+0j)`; negative imaginary parts omit `+`.
///
/// # Arguments
///
/// * `v` - Complex element.
///
/// # Returns
///
/// A parenthesized complex literal string.
///
/// # Errors
///
/// None.
fn format_complex(v: Complex64) -> String {
    if v.im == 0.0 {
        format!("({}+0j)", format_float(v.re))
    } else if v.im.is_sign_positive() {
        format!("({}+{}j)", format_float(v.re), format_float(v.im))
    } else {
        format!("({}{}j)", format_float(v.re), format_float(v.im))
    }
}
