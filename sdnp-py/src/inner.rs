//! Tagged array storage bridging Rust generics and Python dispatch.
//!
//! The core library uses monomorphized `Array<T>` for each element type.
//! Python cannot express that generic, so every `PyArray` holds an
//! [`ArrayInner`] enum and matches on it at operation boundaries. Helpers
//! here expose shape metadata and typed downcasts without leaking the enum
//! to Python users.

use pyo3::prelude::*;
use sdnp::Array;
use sdnp::Complex64;

use crate::dtype::PyDType;
use crate::error::{map_sdnp, value_error};
use crate::unwrap::{scalar_from_item, PyScalar};

/// Erased storage for one of the four supported element types.
///
/// Each variant wraps a core `Array<T>` with identical layout metadata.
/// Dispatch layers (`dispatch`, `coerce`, reductions) match on this tag
/// instead of monomorphizing entire PyO3 APIs four times.
#[derive(Clone)]
pub enum ArrayInner {
    Bool(Array<bool>),
    I64(Array<i64>),
    F64(Array<f64>),
    C64(Array<Complex64>),
}

impl ArrayInner {
    /// Return the runtime dtype tag for this array.
    ///
    /// Mirrors the Python `dtype` property without constructing a PyO3
    /// object. Used for promotion (`PyDType::promote`) before ufunc dispatch.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// The [`PyDType`] variant matching the active enum arm.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub fn dtype(&self) -> PyDType {
        match self {
            ArrayInner::Bool(_) => PyDType::Bool,
            ArrayInner::I64(_) => PyDType::I64,
            ArrayInner::F64(_) => PyDType::F64,
            ArrayInner::C64(_) => PyDType::C64,
        }
    }

    /// Borrow the shape slice from the underlying typed array.
    ///
    /// All variants delegate to the core `Array::shape()` view. The slice
    /// lives as long as `self` and reflects logical dimensions (not bytes).
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Shared slice of dimension lengths, outermost first.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayInner::Bool(a) => a.shape(),
            ArrayInner::I64(a) => a.shape(),
            ArrayInner::F64(a) => a.shape(),
            ArrayInner::C64(a) => a.shape(),
        }
    }

    /// Number of dimensions (`shape.len()`).
    ///
    /// Convenience for dispatch and the 0-D unwrap policy. Internal 0-D
    /// arrays (scalar storage) report `ndim == 0` even though Python rarely
    /// surfaces them as `Array` objects.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Count of shape entries (0 for scalar storage).
    ///
    /// # Errors
    ///
    /// Never fails.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Total element count (product of shape extents).
    ///
    /// Matches NumPy `size`. For empty arrays, returns 0 regardless of
    /// dtype variant.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Number of elements in the underlying buffer.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub fn size(&self) -> usize {
        match self {
            ArrayInner::Bool(a) => a.size(),
            ArrayInner::I64(a) => a.size(),
            ArrayInner::F64(a) => a.size(),
            ArrayInner::C64(a) => a.size(),
        }
    }

    /// Copy strides into a `Vec` for the Python `strides` getter.
    ///
    /// Core stores strides as a slice; Python expects an owned tuple. Values
    /// are in bytes and follow C-contiguous layout unless the array is a view.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Owned stride vector (one entry per dimension).
    ///
    /// # Errors
    ///
    /// Never fails.
    pub fn strides(&self) -> Vec<isize> {
        match self {
            ArrayInner::Bool(a) => a.strides().to_vec(),
            ArrayInner::I64(a) => a.strides().to_vec(),
            ArrayInner::F64(a) => a.strides().to_vec(),
            ArrayInner::C64(a) => a.strides().to_vec(),
        }
    }

    /// Read the sole element of a 0-D array as a [`PyScalar`].
    ///
    /// Bridges internal scalar storage to Python unwrap. Callers must ensure
    /// `ndim == 0`; the core `item()` API errors on multi-dimensional arrays.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Tagged scalar matching the array's dtype.
    ///
    /// # Errors
    ///
    /// * `ValueError` — array is not 0-D (via core `item()`).
    pub fn item_scalar(&self) -> PyResult<PyScalar> {
        match self {
            ArrayInner::Bool(a) => Ok(PyScalar::Bool(map_sdnp(a.item())?)),
            ArrayInner::I64(a) => Ok(PyScalar::I64(map_sdnp(a.item())?)),
            ArrayInner::F64(a) => Ok(PyScalar::F64(map_sdnp(a.item())?)),
            ArrayInner::C64(a) => Ok(PyScalar::C64(map_sdnp(a.item())?)),
        }
    }

    /// Downcast to `Array<bool>` or raise a dtype error.
    ///
    /// Used when a Python API requires boolean storage (logical ufuncs,
    /// masking). Other variants produce a clear `ValueError`.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Reference to the inner `Array<bool>`.
    ///
    /// # Errors
    ///
    /// * `ValueError` — active variant is not `Bool`.
    pub fn as_bool(&self) -> PyResult<&Array<bool>> {
        match self {
            ArrayInner::Bool(a) => Ok(a),
            _ => Err(value_error("expected bool array")),
        }
    }

    /// Downcast to `Array<i64>` or raise a dtype error.
    ///
    /// Used by integer-specific kernels and validation that rejects floats.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Reference to the inner `Array<i64>`.
    ///
    /// # Errors
    ///
    /// * `ValueError` — active variant is not `I64`.
    pub fn as_i64(&self) -> PyResult<&Array<i64>> {
        match self {
            ArrayInner::I64(a) => Ok(a),
            _ => Err(value_error("expected int array")),
        }
    }

    /// Downcast to `Array<f64>` or raise a dtype error.
    ///
    /// Used by float-only ufuncs (`isnan`, `isinf`, …) and real-valued paths.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Reference to the inner `Array<f64>`.
    ///
    /// # Errors
    ///
    /// * `ValueError` — active variant is not `F64`.
    pub fn as_f64(&self) -> PyResult<&Array<f64>> {
        match self {
            ArrayInner::F64(a) => Ok(a),
            _ => Err(value_error("expected float array")),
        }
    }

    /// Downcast to `Array<Complex64>` or raise a dtype error.
    ///
    /// Used by complex unary ufuncs (`conj`, `real`, `imag`) and complex
    /// binary dispatch after promotion.
    ///
    /// # Arguments
    ///
    /// None (`self`).
    ///
    /// # Returns
    ///
    /// Reference to the inner `Array<Complex64>`.
    ///
    /// # Errors
    ///
    /// * `ValueError` — active variant is not `C64`.
    pub fn as_c64(&self) -> PyResult<&Array<Complex64>> {
        match self {
            ArrayInner::C64(a) => Ok(a),
            _ => Err(value_error("expected complex array")),
        }
    }
}

/// Wrap a [`PyScalar`] in a typed 0-D [`ArrayInner`] for ufunc dispatch.
///
/// Creates minimal internal storage so unary/binary kernels can treat scalars
/// like 0-D arrays without exposing 0-D `Array` objects to Python. Infallible
/// because 0-D `from_vec` with one element cannot fail.
///
/// # Arguments
///
/// * `scalar` - Coerced Python scalar (bool/int/float/complex).
///
/// # Returns
///
/// Matching [`ArrayInner`] variant holding a single element.
///
/// # Errors
///
/// Never fails (uses `expect` on infallible 0-D construction).
pub(crate) fn scalar_to_inner(scalar: &PyScalar) -> ArrayInner {
    match scalar {
        PyScalar::Bool(v) => {
            ArrayInner::Bool(Array::from_vec(vec![*v], &[]).expect("0-D bool"))
        }
        PyScalar::I64(v) => {
            ArrayInner::I64(Array::from_vec(vec![*v], &[]).expect("0-D i64"))
        }
        PyScalar::F64(v) => {
            ArrayInner::F64(Array::from_vec(vec![*v], &[]).expect("0-D f64"))
        }
        PyScalar::C64(v) => {
            ArrayInner::C64(Array::from_vec(vec![*v], &[]).expect("0-D c64"))
        }
    }
}

/// Apply the 0-D unwrap policy: scalars out, ndim ≥ 1 arrays as `PyArray`.
///
/// NumPy-compatible UX: ufunc/reduction results with `ndim == 0` become bare
/// Python bool/int/float/complex via [`scalar_from_item`]. Higher-dimensional
/// results are wrapped as `sdnp.Array`.
///
/// # Arguments
///
/// * `py` - GIL token for constructing Python objects.
/// * `inner` - Result storage after a kernel or coercion step.
///
/// # Returns
///
/// Either a native Python scalar or a new `Array` PyObject.
///
/// # Errors
///
/// * `ValueError` — failed to read 0-D element (via [`item_scalar`]).
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// x = np.array([1, 2, 3])
/// assert np.sum(x) == 6          # 0-D unwrap → int, not Array
/// ```
pub(crate) fn finish_array(
    py: Python<'_>,
    inner: ArrayInner,
) -> PyResult<PyObject> {
    if inner.ndim() == 0 {
        // NumPy UX: reduction/ufunc on 0-D → native Python scalar.
        scalar_from_item(py, inner.item_scalar()?)
    } else {
        crate::array::into_pyobject_raw(py, crate::array::PyArray { inner })
    }
}
