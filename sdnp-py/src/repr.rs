//! Python dtype dispatch for core `repr` and `str` formatting.

use sdnp::{format_array_repr, format_array_str};

use crate::inner::ArrayInner;

/// Dispatch the diagnostic core formatter for the active dtype.
///
/// # Arguments
///
/// * `inner` - Typed storage to display.
/// * `address` - Address of the surrounding Python object.
///
/// # Returns
///
/// A six-line, 80-column diagnostic representation.
///
/// # Errors
///
/// Never fails.
pub fn array_repr(
    inner: &ArrayInner,
    address: usize,
) -> pyo3::PyResult<String> {
    Ok(match inner {
        ArrayInner::Bool(array) => format_array_repr(array, address),
        ArrayInner::I64(array) => format_array_repr(array, address),
        ArrayInner::F64(array) => format_array_repr(array, address),
        ArrayInner::C64(array) => format_array_repr(array, address),
    })
}

/// Dispatch the R-style core string formatter for the active dtype.
///
/// The final two axes form a matrix. Leading axes paginate in logical order,
/// with zero-based labels such as `[0, ,]`.
///
/// # Arguments
///
/// * `inner` - Typed storage to display.
///
/// # Returns
///
/// An R-style, 80-column display string.
///
/// # Errors
///
/// Never fails.
pub fn array_str(inner: &ArrayInner) -> pyo3::PyResult<String> {
    Ok(match inner {
        ArrayInner::Bool(array) => format_array_str(array),
        ArrayInner::I64(array) => format_array_str(array),
        ArrayInner::F64(array) => format_array_str(array),
        ArrayInner::C64(array) => format_array_str(array),
    })
}
