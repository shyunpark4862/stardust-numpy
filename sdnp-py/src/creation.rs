//! Array-creation free functions (`array`, `zeros`, ranges, grids, …).
//!
//! Each entry point parses Python arguments, applies Python-only surface
//! policy, selects a typed `sdnp` factory kernel, and returns a value through
//! the 0-D unwrap policy. The core owns shared shape and allocation semantics.
//! Bool restrictions on `eye`, `tri`, `tril`, `triu`, `diag`, and `meshgrid`
//! are deliberately Python-only dtype policy, not generic core semantics.
//! Default dtype is float64 where NumPy would agree.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sdnp::MeshgridIndexing;

use crate::array::{array_from_inner, wrap_result, PyArray};
use crate::coerce::{
    coerce_array_like, coerce_scalar, parse_shape, require_pyarray,
};
use crate::dtype::PyDType;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::check_meshgrid_indexing;

/// Construct an array from nested sequences, or broadcast a scalar to `shape`.
///
/// Bare Python scalars cannot become 0-D arrays; use `shape=` with a scalar
/// fill value instead. Default dtype follows nested-sequence inference or
/// float64 for factory-style fills.
///
/// # Arguments
///
/// * `obj` - Nested sequence, scalar (with `shape=`), or existing `Array`.
/// * `dtype` - Optional target dtype (`bool`, `int`, `float`, `complex`).
/// * `shape` - When set, broadcast `obj` as a scalar fill value.
///
/// # Returns
///
/// An `Array` with ndim ≥ 1, or a bare Python scalar when ndim would be 0.
///
/// # Errors
///
/// * `ValueError` — 0-D scalar without `shape=`, invalid shape, or core
///   allocation failure.
/// * `TypeError` — unsupported nested structure or dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])
/// assert a.shape == (2, 2)
/// b = np.array(5, shape=(2, 2))
/// assert b.to_list() == [[5, 5], [5, 5]]
/// ```
#[pyfunction]
#[pyo3(signature = (obj, *, dtype=None, shape=None))]
pub fn array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let dt = dtype.map(PyDType::from_python_type).transpose()?;
    if let Some(shape) = shape {
        let shape = parse_shape(shape)?;
        let scalar = coerce_scalar(obj)?;
        let mut arr = scalar_fill_array(scalar, &shape)?;
        if let Some(dt) = dt {
            arr.inner = crate::dispatch::cast_inner(arr.inner, dt)?;
        }
        return crate::array::into_pyobject(py, arr);
    }
    if crate::coerce::is_python_scalar(obj) {
        // Bare scalars cannot become 0-D arrays from Python.
        return Err(value_error(
            "0-dimensional arrays cannot be created from Python",
        ));
    }
    let arr = coerce_array_like(obj, dt)?;
    crate::array::into_pyobject(py, arr)
}

/// Fill `shape` with one scalar value (typed by the scalar's dtype).
///
/// Internal helper for [`full`] and [`array`] with `shape=`.
///
/// # Arguments
///
/// * `scalar` - Coerced Python scalar with resolved storage type.
/// * `shape` - Parsed output dimensions.
///
/// # Returns
///
/// An `Array` filled with `scalar` at every element.
///
/// # Errors
///
/// * `ValueError` — invalid shape or core allocation failure.
fn scalar_fill_array(
    scalar: crate::unwrap::PyScalar,
    shape: &[usize],
) -> PyResult<PyArray> {
    use crate::unwrap::PyScalar;
    let inner = match scalar {
        PyScalar::Bool(v) => ArrayInner::Bool(map_sdnp(sdnp::full(shape, v))?),
        PyScalar::I64(v) => ArrayInner::I64(map_sdnp(sdnp::full(shape, v))?),
        PyScalar::F64(v) => ArrayInner::F64(map_sdnp(sdnp::full(shape, v))?),
        PyScalar::C64(v) => ArrayInner::C64(map_sdnp(sdnp::full(shape, v))?),
    };
    Ok(array_from_inner(inner))
}

/// Return a new array of zeros with the given `shape`.
///
/// Default dtype is float64 when `dtype` is omitted.
///
/// # Arguments
///
/// * `shape` - Tuple or int sequence defining output dimensions.
/// * `dtype` - Optional element type (`bool`, `int`, `float`, `complex`).
///
/// # Returns
///
/// A zero-filled `Array`.
///
/// # Errors
///
/// * `ValueError` — invalid shape or allocation failure.
/// * `TypeError` — unsupported dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.zeros((2, 3))
/// assert a.shape == (2, 3)
/// assert a[0, 0] == 0.0
/// ```
#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None))]
pub fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let shape = parse_shape(shape)?;
    let dt = dtype
        .map(PyDType::from_python_type)
        .transpose()?
        .unwrap_or(PyDType::F64);
    let inner = match dt {
        PyDType::Bool => ArrayInner::Bool(map_sdnp(sdnp::full(&shape, false))?),
        PyDType::I64 => ArrayInner::I64(map_sdnp(sdnp::zeros(&shape))?),
        PyDType::F64 => ArrayInner::F64(map_sdnp(sdnp::zeros(&shape))?),
        PyDType::C64 => ArrayInner::C64(map_sdnp(sdnp::zeros(&shape))?),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return a new array of ones with the given `shape`.
///
/// Default dtype is float64 when `dtype` is omitted.
///
/// # Arguments
///
/// * `shape` - Tuple or int sequence defining output dimensions.
/// * `dtype` - Optional element type (`bool`, `int`, `float`, `complex`).
///
/// # Returns
///
/// A one-filled `Array`.
///
/// # Errors
///
/// * `ValueError` — invalid shape or allocation failure.
/// * `TypeError` — unsupported dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.ones(3, dtype=int)
/// assert a.to_list() == [1, 1, 1]
/// ```
#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None))]
pub fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let shape = parse_shape(shape)?;
    let dt = dtype
        .map(PyDType::from_python_type)
        .transpose()?
        .unwrap_or(PyDType::F64);
    let inner = match dt {
        PyDType::Bool => ArrayInner::Bool(map_sdnp(sdnp::full(&shape, true))?),
        PyDType::I64 => ArrayInner::I64(map_sdnp(sdnp::ones(&shape))?),
        PyDType::F64 => ArrayInner::F64(map_sdnp(sdnp::ones(&shape))?),
        PyDType::C64 => ArrayInner::C64(map_sdnp(sdnp::ones(&shape))?),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return a new array filled with `fill_value`.
///
/// Dtype is inferred from `fill_value`; no explicit `dtype` keyword.
///
/// # Arguments
///
/// * `shape` - Tuple or int sequence defining output dimensions.
/// * `fill_value` - Scalar broadcast to every element.
///
/// # Returns
///
/// An `Array` where every element equals `fill_value`.
///
/// # Errors
///
/// * `ValueError` — invalid shape or allocation failure.
/// * `TypeError` — unsupported scalar type.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.full((2, 2), 7)
/// assert a.to_list() == [[7, 7], [7, 7]]
/// ```
#[pyfunction]
#[pyo3(signature = (shape, fill_value))]
pub fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let shape = parse_shape(shape)?;
    let scalar = coerce_scalar(fill_value)?;
    crate::array::into_pyobject(py, scalar_fill_array(scalar, &shape)?)
}

/// Return evenly spaced integer values in `[start, stop)` or `[0, start)`.
///
/// When `stop` is omitted, `start` acts as the exclusive upper bound and the
/// implicit start is 0. Output dtype is always int64.
///
/// # Arguments
///
/// * `start` - Inclusive start, or exclusive stop when `stop` is `None`.
/// * `stop` - Exclusive upper bound; when omitted, range is `[0, start)`.
/// * `step` - Stride between consecutive values (must be non-zero).
///
/// # Returns
///
/// A 1-D int64 `Array`.
///
/// # Errors
///
/// * `ValueError` — zero step or core range failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert np.arange(5).to_list() == [0, 1, 2, 3, 4]
/// assert np.arange(2, 10, 2).to_list() == [2, 4, 6, 8]
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop=None, step=1))]
pub fn arange(
    py: Python<'_>,
    start: i64,
    stop: Option<i64>,
    step: i64,
) -> PyResult<PyObject> {
    let arr = match stop {
        None => map_sdnp(sdnp::arange_stop(start))?,
        Some(stop) => map_sdnp(sdnp::arange(start, stop, step))?,
    };
    crate::array::into_pyobject(py, array_from_inner(ArrayInner::I64(arr)))
}

/// Return evenly spaced float64 values between `start` and `stop`.
///
/// When `endpoint` is true (default), `stop` is included; otherwise the step
/// is adjusted so samples span the interval without the final point.
///
/// # Arguments
///
/// * `start` - First sample value (must be finite).
/// * `stop` - Last sample when `endpoint=true` (must be finite).
/// * `num` - Number of samples to generate.
/// * `endpoint` - Include `stop` in the output when true.
///
/// # Returns
///
/// A 1-D float64 `Array`, or a bare Python float when `num` is 1 and 0-D
/// unwrap applies.
///
/// # Errors
///
/// * `ValueError` — non-finite bounds or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.linspace(0.0, 1.0, 5)
/// assert len(a) == 5
/// assert a[0] == 0.0
/// assert a[-1] == 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, num, *, endpoint=true))]
pub fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> PyResult<PyObject> {
    wrap_result(
        py,
        ArrayInner::F64(map_sdnp(sdnp::linspace(start, stop, num, endpoint))?),
    )
}

/// Return evenly spaced samples on a log scale.
///
/// Values are `base ** x` where `x` comes from [`linspace`] over `[start,
/// stop]`. Default `base` is 10.
///
/// # Arguments
///
/// * `start` - Exponent at the first sample (must be finite).
/// * `stop` - Exponent at the last sample when `endpoint=true`.
/// * `num` - Number of samples to generate.
/// * `endpoint` - Include the `stop` exponent when true.
/// * `base` - Logarithm base (must be positive and not 1).
///
/// # Returns
///
/// A 1-D float64 `Array` of powers of `base`.
///
/// # Errors
///
/// * `ValueError` — non-finite bounds, invalid base, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.logspace(0.0, 2.0, 3)
/// assert a.to_list() == [1.0, 10.0, 100.0]
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, num, *, endpoint=true, base=10.0))]
pub fn logspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    base: f64,
) -> PyResult<PyObject> {
    wrap_result(
        py,
        ArrayInner::F64(map_sdnp(sdnp::logspace(
            start, stop, num, endpoint, base,
        ))?),
    )
}

/// Return evenly spaced samples on a geometric progression.
///
/// Samples lie on a multiplicative scale from `start` to `stop`. Both
/// endpoints must be finite and non-zero with the same sign.
///
/// # Arguments
///
/// * `start` - First sample value.
/// * `stop` - Last sample when `endpoint=true`.
/// * `num` - Number of samples to generate.
/// * `endpoint` - Include `stop` in the output when true.
///
/// # Returns
///
/// A 1-D float64 `Array`.
///
/// # Errors
///
/// * `ValueError` — invalid bounds or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.geomspace(1.0, 8.0, 4)
/// assert a.to_list() == [1.0, 2.0, 4.0, 8.0]
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, num, *, endpoint=true))]
pub fn geomspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> PyResult<PyObject> {
    wrap_result(
        py,
        ArrayInner::F64(map_sdnp(sdnp::geomspace(start, stop, num, endpoint))?),
    )
}

/// Return an `n × n` identity matrix.
///
/// Default dtype is float64. Bool dtype is not supported.
///
/// # Arguments
///
/// * `n` - Side length of the square output.
/// * `dtype` - Optional element type (`int`, `float`, or `complex`).
///
/// # Returns
///
/// A 2-D identity `Array`.
///
/// # Errors
///
/// * `ValueError` — bool dtype requested or core failure.
/// * `TypeError` — unsupported dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.eye(3)
/// assert a.to_list() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
/// ```
#[pyfunction]
#[pyo3(signature = (n, *, dtype=None))]
pub fn eye(
    py: Python<'_>,
    n: usize,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let dt = dtype
        .map(PyDType::from_python_type)
        .transpose()?
        .unwrap_or(PyDType::F64);
    let inner = match dt {
        PyDType::I64 => ArrayInner::I64(map_sdnp(sdnp::eye(n))?),
        PyDType::F64 => ArrayInner::F64(map_sdnp(sdnp::eye(n))?),
        PyDType::C64 => ArrayInner::C64(map_sdnp(sdnp::eye(n))?),
        PyDType::Bool => {
            return Err(value_error("eye does not support bool dtype"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return an `n × m` identity-like matrix with diagonal offset `k`.
///
/// Default dtype is float64. Bool dtype is not supported.
///
/// # Arguments
///
/// * `n` - Number of rows.
/// * `m` - Number of columns.
/// * `k` - Diagonal offset (`0` is main diagonal, positive is above).
/// * `dtype` - Optional element type (`int`, `float`, or `complex`).
///
/// # Returns
///
/// A 2-D `Array` with ones on the selected diagonal.
///
/// # Errors
///
/// * `ValueError` — bool dtype requested or core failure.
/// * `TypeError` — unsupported dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.eye_with(3, 4, k=1)
/// assert a.shape == (3, 4)
/// ```
#[pyfunction]
#[pyo3(signature = (n, m, *, k=0, dtype=None))]
pub fn eye_with(
    py: Python<'_>,
    n: usize,
    m: usize,
    k: isize,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let dt = dtype
        .map(PyDType::from_python_type)
        .transpose()?
        .unwrap_or(PyDType::F64);
    let inner = match dt {
        PyDType::I64 => ArrayInner::I64(map_sdnp(sdnp::eye_with(n, m, k))?),
        PyDType::F64 => ArrayInner::F64(map_sdnp(sdnp::eye_with(n, m, k))?),
        PyDType::C64 => ArrayInner::C64(map_sdnp(sdnp::eye_with(n, m, k))?),
        PyDType::Bool => {
            return Err(value_error("eye_with does not support bool dtype"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return an `n × n` lower-triangular matrix of ones.
///
/// Default dtype is float64. Bool dtype is not supported.
///
/// # Arguments
///
/// * `n` - Side length of the square output.
/// * `dtype` - Optional element type (`int`, `float`, or `complex`).
///
/// # Returns
///
/// A 2-D lower-triangular `Array`.
///
/// # Errors
///
/// * `ValueError` — bool dtype requested or core failure.
/// * `TypeError` — unsupported dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.tri(3)
/// assert a[2, 0] == 1.0
/// assert a[0, 2] == 0.0
/// ```
#[pyfunction]
#[pyo3(signature = (n, *, dtype=None))]
pub fn tri(
    py: Python<'_>,
    n: usize,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let dt = dtype
        .map(PyDType::from_python_type)
        .transpose()?
        .unwrap_or(PyDType::F64);
    let inner = match dt {
        PyDType::I64 => ArrayInner::I64(map_sdnp(sdnp::tri(n))?),
        PyDType::F64 => ArrayInner::F64(map_sdnp(sdnp::tri(n))?),
        PyDType::C64 => ArrayInner::C64(map_sdnp(sdnp::tri(n))?),
        PyDType::Bool => {
            return Err(value_error("tri does not support bool dtype"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return an `n × m` lower-triangular matrix with diagonal offset `k`.
///
/// Default dtype is float64. Bool dtype is not supported.
///
/// # Arguments
///
/// * `n` - Number of rows.
/// * `m` - Number of columns.
/// * `k` - Diagonal offset controlling which triangle is filled.
/// * `dtype` - Optional element type (`int`, `float`, or `complex`).
///
/// # Returns
///
/// A 2-D lower-triangular `Array`.
///
/// # Errors
///
/// * `ValueError` — bool dtype requested or core failure.
/// * `TypeError` — unsupported dtype object.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.tri_with(3, 4, k=-1)
/// assert a.shape == (3, 4)
/// ```
#[pyfunction]
#[pyo3(signature = (n, m, k=0, *, dtype=None))]
pub fn tri_with(
    py: Python<'_>,
    n: usize,
    m: usize,
    k: isize,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let dt = dtype
        .map(PyDType::from_python_type)
        .transpose()?
        .unwrap_or(PyDType::F64);
    let inner = match dt {
        PyDType::I64 => ArrayInner::I64(map_sdnp(sdnp::tri_with(n, m, k))?),
        PyDType::F64 => ArrayInner::F64(map_sdnp(sdnp::tri_with(n, m, k))?),
        PyDType::C64 => ArrayInner::C64(map_sdnp(sdnp::tri_with(n, m, k))?),
        PyDType::Bool => {
            return Err(value_error("tri does not support bool dtype"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return a copy with elements above the `k`-th diagonal zeroed.
///
/// Bool dtype is not supported. Input must be at least 2-D.
///
/// # Arguments
///
/// * `array` - Input 2-D (or higher) array.
/// * `k` - Diagonal offset (`0` keeps main diagonal and below).
///
/// # Returns
///
/// Lower-triangular view copy with the same shape and dtype.
///
/// # Errors
///
/// * `TypeError` — 0-D input or bool dtype.
/// * `ValueError` — non-2-D input or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])
/// assert np.tril(a).to_list() == [[1, 0], [3, 4]]
/// ```
#[pyfunction]
#[pyo3(signature = (array, k=0))]
pub fn tril(
    py: Python<'_>,
    array: PyRef<PyArray>,
    k: isize,
) -> PyResult<PyObject> {
    array.reject_zero_dim_input("tril")?;
    let inner = match &array.inner {
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(sdnp::tril(a, k))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(sdnp::tril(a, k))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(sdnp::tril(a, k))?),
        ArrayInner::Bool(_) => {
            return Err(value_error("tril does not support bool dtype"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return a copy with elements below the `k`-th diagonal zeroed.
///
/// Bool dtype is not supported. Input must be at least 2-D.
///
/// # Arguments
///
/// * `array` - Input 2-D (or higher) array.
/// * `k` - Diagonal offset (`0` keeps main diagonal and above).
///
/// # Returns
///
/// Upper-triangular view copy with the same shape and dtype.
///
/// # Errors
///
/// * `TypeError` — 0-D input or bool dtype.
/// * `ValueError` — non-2-D input or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])
/// assert np.triu(a).to_list() == [[1, 2], [0, 4]]
/// ```
#[pyfunction]
#[pyo3(signature = (array, k=0))]
pub fn triu(
    py: Python<'_>,
    array: PyRef<PyArray>,
    k: isize,
) -> PyResult<PyObject> {
    array.reject_zero_dim_input("triu")?;
    let inner = match &array.inner {
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(sdnp::triu(a, k))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(sdnp::triu(a, k))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(sdnp::triu(a, k))?),
        ArrayInner::Bool(_) => {
            return Err(value_error("triu does not support bool dtype"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Extract a diagonal or construct a diagonal matrix from a vector.
///
/// For 2-D input, returns the `k`-th diagonal as a 1-D array. For 1-D input,
/// returns a square matrix with the vector on the main diagonal. This creation
/// API intentionally rejects bool input at the Python boundary; the separate
/// `linalg.diagonal` API has a broader dtype policy and accepts bool arrays.
///
/// # Arguments
///
/// * `array` - 1-D vector or 2-D matrix.
/// * `k` - Diagonal offset (`0` is main diagonal).
///
/// # Returns
///
/// A 1-D diagonal vector or 2-D diagonal matrix.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — boolean input, invalid shape, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])
/// assert np.diag(a).to_list() == [1, 4]
/// np.diag(np.array([[True, False]]))  # ValueError: Python dtype policy
/// ```
#[pyfunction]
#[pyo3(signature = (array, k=0))]
pub fn diag(
    py: Python<'_>,
    array: PyRef<PyArray>,
    k: isize,
) -> PyResult<PyObject> {
    array.reject_zero_dim_input("diag")?;
    let inner = match &array.inner {
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(sdnp::diag(a, k))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(sdnp::diag(a, k))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(sdnp::diag(a, k))?),
        ArrayInner::Bool(_) => {
            return Err(value_error("diag does not support boolean arrays"))
        }
    };
    wrap_result(py, inner)
}

/// Return coordinate matrices from coordinate vectors.
///
/// All inputs must share the same dtype (int, float, or complex). Bool is
/// not supported. Empty input returns an empty tuple.
///
/// # Arguments
///
/// * `arrays` - Tuple of 1-D coordinate arrays.
/// * `indexing` - `"xy"` (Cartesian default) or `"ij"` (matrix indexing).
///
/// # Returns
///
/// A tuple of broadcast coordinate grids, one per input vector.
///
/// # Errors
///
/// * `TypeError` — non-array input.
/// * `ValueError` — dtype mismatch, bool dtype, invalid `indexing`, shape
///   rules, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// x, y = np.meshgrid(np.array([0, 1]), np.array([10, 20]))
/// assert x.shape == (2, 2)
/// assert y.shape == (2, 2)
/// ```
#[pyfunction]
#[pyo3(signature = (*arrays, indexing="xy"))]
pub fn meshgrid(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    indexing: &str,
) -> PyResult<PyObject> {
    let tuple = arrays.downcast::<pyo3::types::PyTuple>()?;
    check_meshgrid_indexing(indexing)?;
    if tuple.is_empty() {
        return Ok(PyTuple::empty(py).into());
    }
    let validated = tuple
        .iter()
        .map(|item| Ok(require_pyarray(&item, "meshgrid")?.inner.clone()))
        .collect::<PyResult<Vec<_>>>()?;
    let idx = match indexing {
        "xy" => MeshgridIndexing::Xy,
        "ij" => MeshgridIndexing::Ij,
        _ => unreachable!("validated above"),
    };
    let inner = match &validated[0] {
        ArrayInner::I64(_) => {
            let owned: Vec<_> = validated
                .iter()
                .map(|arr| match arr {
                    ArrayInner::I64(a) => Ok(a.clone()),
                    _ => Err(value_error("meshgrid dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            let refs: Vec<_> = owned.iter().collect();
            let out = map_sdnp(sdnp::meshgrid(&refs, idx))?;
            out.into_iter().map(ArrayInner::I64).collect::<Vec<_>>()
        }
        ArrayInner::F64(_) => {
            let owned: Vec<_> = validated
                .iter()
                .map(|arr| match arr {
                    ArrayInner::F64(a) => Ok(a.clone()),
                    _ => Err(value_error("meshgrid dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            let refs: Vec<_> = owned.iter().collect();
            let out = map_sdnp(sdnp::meshgrid(&refs, idx))?;
            out.into_iter().map(ArrayInner::F64).collect::<Vec<_>>()
        }
        ArrayInner::C64(_) => {
            let owned: Vec<_> = validated
                .iter()
                .map(|arr| match arr {
                    ArrayInner::C64(a) => Ok(a.clone()),
                    _ => Err(value_error("meshgrid dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            let refs: Vec<_> = owned.iter().collect();
            let out = map_sdnp(sdnp::meshgrid(&refs, idx))?;
            out.into_iter().map(ArrayInner::C64).collect::<Vec<_>>()
        }
        ArrayInner::Bool(_) => {
            return Err(value_error("meshgrid does not support bool dtype"))
        }
    };
    let tuple = PyTuple::new(
        py,
        inner
            .into_iter()
            .map(|a| crate::array::into_pyobject(py, array_from_inner(a)))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(tuple.into())
}

/// Register array-creation callables on the extension module.
///
/// Adds `array`, factory functions, range generators, triangular matrices,
/// and `meshgrid` to the `sdnp` module object.
///
/// # Arguments
///
/// * `m` - Bound reference to the `sdnp` extension module.
///
/// # Returns
///
/// `Ok(())` when every callable is registered successfully.
///
/// # Errors
///
/// Returns `PyErr` if PyO3 function wrapping or registration fails.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert callable(np.zeros)
/// assert callable(np.meshgrid)
/// ```
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(logspace, m)?)?;
    m.add_function(wrap_pyfunction!(geomspace, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(eye_with, m)?)?;
    m.add_function(wrap_pyfunction!(tri, m)?)?;
    m.add_function(wrap_pyfunction!(tri_with, m)?)?;
    m.add_function(wrap_pyfunction!(tril, m)?)?;
    m.add_function(wrap_pyfunction!(triu, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(meshgrid, m)?)?;
    Ok(())
}
