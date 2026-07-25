//! Map `sdnp::Error` values to Python exceptions.
//!
//! User-facing argument checks belong in [`crate::validate`] before the core
//! runs. Errors returned from the core are domain invariants (broadcast
//! failure, out-of-bounds index, read-only buffer, …) and are translated here
//! into the closest standard Python exception type.

use pyo3::exceptions::{
    PyIndexError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use sdnp::Error;

/// Convert a core [`Error`] into the appropriate Python exception.
///
/// After the generic Rust core validates domain invariants, failures surface
/// as `sdnp::Error`. This function is the single mapping point from those
/// variants to standard Python exception types, matching NumPy conventions
/// where possible (e.g. index failures → [`PyIndexError`]).
///
/// # Arguments
///
/// * `err` — Core error returned from an `sdnp` operation.
///
/// # Returns
///
/// A [`PyErr`] ready to propagate across the PyO3 boundary.
///
/// # Errors
///
/// This function constructs errors rather than returning `Result`; the
/// returned [`PyErr`] types are:
///
/// * [`PyValueError`] — shape, broadcast, read-only buffer, invalid argument.
/// * [`PyIndexError`] — out-of-bounds indexing.
/// * [`PyZeroDivisionError`] — division by zero in numeric kernels.
///
/// # Examples
///
/// ```rust
/// use sdnp::Error;
/// use sdnp_py::error::sdnp_err;
///
/// let py_err = sdnp_err(Error::DivideByZero);
/// assert!(py_err.to_string().contains("DivideByZero"));
/// ```
pub(crate) fn sdnp_err(err: Error) -> PyErr {
    match err {
        // Shape/layout invariants → ValueError (NumPy convention).
        Error::BufferSizeMismatch { .. }
        | Error::ShapeStridesMismatch { .. }
        | Error::Broadcast { .. }
        | Error::ReadOnly => PyValueError::new_err(err.to_string()),
        Error::InvalidArgument(msg) => PyValueError::new_err(msg),
        Error::IndexOutOfBounds { .. } => {
            PyIndexError::new_err(err.to_string())
        }
        Error::DivideByZero => PyZeroDivisionError::new_err(err.to_string()),
    }
}

/// Lift `sdnp::Result<T>` into [`PyResult<T>`] via [`sdnp_err`].
///
/// Binding code that calls the generic core should use this at return sites
/// instead of manual `match` on `Error`. Argument validation errors are
/// raised earlier via [`value_error`], [`index_error`], etc.
///
/// # Arguments
///
/// * `result` — Outcome of a core operation.
///
/// # Returns
///
/// `Ok(value)` on success, or `Err(PyErr)` after [`sdnp_err`] mapping.
///
/// # Errors
///
/// Any [`Error`] variant maps through [`sdnp_err`] to [`PyValueError`],
/// [`PyIndexError`], or [`PyZeroDivisionError`].
///
/// # Examples
///
/// ```rust
/// use sdnp_py::error::map_sdnp;
/// use sdnp_py::inner::ArrayInner;
///
/// fn add(a: ArrayInner, b: ArrayInner) -> pyo3::PyResult<ArrayInner> {
///     map_sdnp(sdnp::ufunc::add(&a, &b))
/// }
/// ```
pub(crate) fn map_sdnp<T>(result: sdnp::Result<T>) -> PyResult<T> {
    result.map_err(sdnp_err)
}

/// Shorthand for [`PyValueError`] with a custom message.
///
/// Use at the Python boundary for user-input mistakes: invalid keyword
/// combinations, incompatible shapes checked before the core, empty
/// sequences, and similar API contract violations.
///
/// # Arguments
///
/// * `msg` — Human-readable message (often matching NumPy wording).
///
/// # Returns
///
/// A constructed [`PyErr`] wrapping [`PyValueError`].
///
/// # Errors
///
/// Always produces [`PyValueError`]; this helper does not return `Result`.
///
/// # Examples
///
/// ```rust
/// use sdnp_py::error::value_error;
///
/// let err = value_error("reshape target shape must be non-empty");
/// ```
pub(crate) fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Shorthand for [`PyIndexError`] with a custom message.
///
/// Axis and index bounds checked at the binding layer (before the core
/// canonicalizes negative indices) should use this for NumPy-compatible
/// exception types.
///
/// # Arguments
///
/// * `msg` — Human-readable index error message.
///
/// # Returns
///
/// A constructed [`PyErr`] wrapping [`PyIndexError`].
///
/// # Errors
///
/// Always produces [`PyIndexError`].
pub(crate) fn index_error(msg: impl Into<String>) -> PyErr {
    PyIndexError::new_err(msg.into())
}

/// Shorthand for [`PyTypeError`] with a custom message.
///
/// Dtype mismatches, wrong operand kinds, and unsupported Python types
/// map to [`PyTypeError`] rather than [`PyValueError`].
///
/// # Arguments
///
/// * `msg` — Human-readable type error message.
///
/// # Returns
///
/// A constructed [`PyErr`] wrapping [`PyTypeError`].
///
/// # Errors
///
/// Always produces [`PyTypeError`].
///
/// # Examples
///
/// ```rust
/// use sdnp_py::error::type_error;
///
/// let err = type_error("condition must be a bool array");
/// ```
pub(crate) fn type_error(msg: impl Into<String>) -> PyErr {
    PyTypeError::new_err(msg.into())
}

/// Shorthand for [`PyZeroDivisionError`] with a custom message.
///
/// Used when binding code detects division by zero before calling the core,
/// or when constructing errors that mirror numeric kernel failures.
///
/// # Arguments
///
/// * `msg` — Human-readable message.
///
/// # Returns
///
/// A constructed [`PyErr`] wrapping [`PyZeroDivisionError`].
///
/// # Errors
///
/// Always produces [`PyZeroDivisionError`].
pub(crate) fn zero_division_error(msg: impl Into<String>) -> PyErr {
    PyZeroDivisionError::new_err(msg.into())
}
