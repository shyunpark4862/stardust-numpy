//! Zero-dimensional unwrap policy at the Python boundary.
//!
//! NumPy returns Python scalars for 0-D results; this crate mirrors that UX.
//! Internally, scalar results still live in typed 0-D [`ArrayInner`] values
//! until [`finish`] converts them to `bool`, `int`, `float`, or `complex`.

use pyo3::prelude::*;
use pyo3::types::PyComplex;
use sdnp::Complex64;

use crate::inner::{finish_array, ArrayInner};

/// Typed scalar extracted from a 0-D array or Python literal.
///
/// Ufunc and reduction paths may produce a logical 0-D [`ArrayInner`]. Before
/// returning to Python, the element is unpacked into this enum so the correct
/// builtin scalar type can be constructed without exposing 0-D array objects.
#[derive(Clone, Debug)]
pub enum PyScalar {
    /// Boolean scalar (`True` / `False`).
    Bool(bool),
    /// Signed 64-bit integer scalar.
    I64(i64),
    /// IEEE double scalar.
    F64(f64),
    /// Complex double scalar.
    C64(Complex64),
}

impl PyScalar {
    /// Map this scalar to its [`PyDType`] tag.
    ///
    /// Used when a scalar participates in dtype promotion or dispatch alongside
    /// array operands without allocating a 0-D wrapper array.
    ///
    /// # Arguments
    ///
    /// None — inspects the active variant of `self`.
    ///
    /// # Returns
    ///
    /// The [`PyDType`] variant matching the scalar's storage kind.
    pub fn dtype(&self) -> crate::dtype::PyDType {
        use crate::dtype::PyDType;
        match self {
            PyScalar::Bool(_) => PyDType::Bool,
            PyScalar::I64(_) => PyDType::I64,
            PyScalar::F64(_) => PyDType::F64,
            PyScalar::C64(_) => PyDType::C64,
        }
    }
}

/// Build the native Python object for one typed scalar.
///
/// Converts an internal [`PyScalar`] into the corresponding builtin Python
/// object under the GIL. Called from [`finish`] and other return paths that
/// must not leak 0-D [`PyArray`] instances to callers.
///
/// # Arguments
///
/// * `py` — GIL token for object allocation.
/// * `scalar` — Typed value to expose in Python.
///
/// # Returns
///
/// `Ok(PyObject)` owning a `bool`, `int`, `float`, or `complex` instance.
///
/// # Errors
///
/// PyO3 allocation errors when boxing integers or floats (rare).
///
/// # Examples
///
/// ```rust
/// use sdnp_py::unwrap::{scalar_from_item, PyScalar};
///
/// Python::with_gil(|py| {
///     let obj = scalar_from_item(py, PyScalar::I64(42)).unwrap();
///     assert_eq!(obj.extract::<i64>(py).unwrap(), 42);
/// });
/// ```
pub(crate) fn scalar_from_item(
    py: Python<'_>,
    scalar: PyScalar,
) -> PyResult<PyObject> {
    match scalar {
        PyScalar::Bool(v) => {
            Ok(pyo3::types::PyBool::new(py, v).to_owned().unbind().into())
        }
        PyScalar::I64(v) => {
            let bound = v.into_pyobject(py)?;
            Ok(bound.into_any().unbind())
        }
        PyScalar::F64(v) => {
            let bound = v.into_pyobject(py)?;
            Ok(bound.into_any().unbind())
        }
        PyScalar::C64(v) => Ok(PyComplex::from_doubles(py, v.re, v.im).into()),
    }
}

/// Return a Python scalar or `Array` — never expose 0-D arrays to callers.
///
/// This is the public exit hook for operations that may yield rank-0 results.
/// NumPy users expect `np.sum(np.eye(3))` to be a Python float, not a 0-D
/// ndarray; [`finish_array`] unwraps typed 0-D buffers via [`scalar_from_item`]
/// and passes higher-rank results through as [`PyArray`] wrappers.
///
/// # Arguments
///
/// * `py` — GIL token.
/// * `inner` — Result array, possibly 0-D.
///
/// # Returns
///
/// `Ok(PyObject)` that is either a builtin scalar or a [`PyArray`].
///
/// # Errors
///
/// * [`PyValueError`] or other errors from [`finish_array`] on invalid state.
/// * PyO3 errors from scalar boxing.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// assert isinstance(np.sum(np.eye(3)), float)
/// assert hasattr(np.sum(np.eye(3), axis=0), "shape")
/// ```
pub fn finish(py: Python<'_>, inner: ArrayInner) -> PyResult<PyObject> {
    finish_array(py, inner)
}
