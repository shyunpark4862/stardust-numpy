//! Map `sdnp::Error` to Python exceptions.
//!
//! User-input validation should happen in [`crate::validate`] before calling
//! the core. Errors from the core are domain invariants (safe to rely on as a
//! second line of defense).

use pyo3::exceptions::{
    PyIndexError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use sdnp::Error;

pub(crate) fn sdnp_err(err: Error) -> PyErr {
    match err {
        Error::BufferSizeMismatch { .. }
        | Error::ShapeStridesMismatch { .. }
        | Error::Broadcast { .. }
        | Error::ReadOnly => PyValueError::new_err(err.to_string()),
        Error::InvalidArgument(msg) => PyValueError::new_err(msg),
        Error::IndexOutOfBounds { .. } | Error::AxisOutOfBounds { .. } => {
            PyIndexError::new_err(err.to_string())
        }
        Error::DivideByZero => PyZeroDivisionError::new_err(err.to_string()),
    }
}

pub(crate) fn map_sdnp<T>(result: sdnp::Result<T>) -> PyResult<T> {
    result.map_err(sdnp_err)
}

pub(crate) fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

pub(crate) fn index_error(msg: impl Into<String>) -> PyErr {
    PyIndexError::new_err(msg.into())
}

pub(crate) fn type_error(msg: impl Into<String>) -> PyErr {
    PyTypeError::new_err(msg.into())
}

pub(crate) fn zero_division_error(msg: impl Into<String>) -> PyErr {
    PyZeroDivisionError::new_err(msg.into())
}
