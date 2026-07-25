//! Linear algebra free functions and `__matmul__` support.
//!
//! Coerces operands, promotes dtypes, validates contraction/batch shapes at
//! the Python boundary, then calls typed `sdnp` linalg kernels. Results use
//! the 0-D unwrap policy when a contraction returns a scalar.

use pyo3::prelude::*;

use crate::array::{array_from_inner, wrap_result, PyArray};
use crate::coerce::coerce_array_like;
use crate::dispatch::cast_inner;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::{
    check_diagonal_axes, check_dot, check_matmul, check_vdot,
};

/// Shared matmul path for `sdnp.matmul` and `Array.__matmul__`.
///
/// Thin wrapper around [`matmul_impl`] so method and free-function entry
/// points share validation and dispatch.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `left` - Left operand (array or scalar-like).
/// * `right` - Right operand (array or scalar-like).
///
/// # Returns
///
/// Matrix product as an `Array` or bare scalar when 0-D unwrap applies.
///
/// # Errors
///
/// * `TypeError` — incompatible operands.
/// * `ValueError` — shape/dtype mismatch or core failure.
pub fn py_matmul(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    matmul_impl(py, left, right)
}

/// Coerce, promote, validate, and dispatch matrix multiplication.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `left` - Left operand after coercion.
/// * `right` - Right operand after coercion.
///
/// # Returns
///
/// Matrix product storage wrapped for Python export.
///
/// # Errors
///
/// * `TypeError` — incompatible operands.
/// * `ValueError` — contraction shape mismatch or core failure.
fn matmul_impl(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let l = coerce_array_like(left, None)?;
    let r = coerce_array_like(right, None)?;
    check_matmul(&l.inner, &r.inner)?;
    let dt = l.inner.dtype().promote(r.inner.dtype());
    let l = cast_inner(l.inner, dt)?;
    let r = cast_inner(r.inner, dt)?;
    let inner = match (l, r) {
        (ArrayInner::I64(l), ArrayInner::I64(r)) => {
            ArrayInner::I64(map_sdnp(sdnp::matmul(&l, &r))?)
        }
        (ArrayInner::F64(l), ArrayInner::F64(r)) => {
            ArrayInner::F64(map_sdnp(sdnp::matmul(&l, &r))?)
        }
        (ArrayInner::C64(l), ArrayInner::C64(r)) => {
            ArrayInner::C64(map_sdnp(sdnp::matmul(&l, &r))?)
        }
        _ => return Err(value_error("matmul dtype mismatch")),
    };
    wrap_result(py, inner)
}

/// Matrix product (`@` operator and free function).
///
/// Supports vector-matrix, matrix-vector, and batched contractions after
/// dtype promotion.
///
/// # Arguments
///
/// * `left` - Left operand (array or coercible sequence).
/// * `right` - Right operand (array or coercible sequence).
///
/// # Returns
///
/// Product array, or a bare Python scalar when the result is 0-D.
///
/// # Errors
///
/// * `TypeError` — incompatible operands.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])
/// b = np.array([[5, 6], [7, 8]])
/// assert np.matmul(a, b).shape == (2, 2)
/// ```
#[pyfunction]
pub fn matmul(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    matmul_impl(py, left, right)
}

/// Inner product for 1-D/2-D operands (NumPy `dot` semantics).
///
/// Differs from [`matmul`] for higher-rank tensor contractions.
///
/// # Arguments
///
/// * `left` - Left operand (array or coercible sequence).
/// * `right` - Right operand (array or coercible sequence).
///
/// # Returns
///
/// Dot-product array or bare scalar when 0-D unwrap applies.
///
/// # Errors
///
/// * `TypeError` — incompatible operands.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3])
/// b = np.array([4, 5, 6])
/// assert np.dot(a, b) == 32
/// ```
#[pyfunction]
pub fn dot(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let l = coerce_array_like(left, None)?;
    let r = coerce_array_like(right, None)?;
    check_dot(&l.inner, &r.inner)?;
    let dt = l.inner.dtype().promote(r.inner.dtype());
    let l = cast_inner(l.inner, dt)?;
    let r = cast_inner(r.inner, dt)?;
    let inner = match (l, r) {
        (ArrayInner::I64(l), ArrayInner::I64(r)) => {
            ArrayInner::I64(map_sdnp(sdnp::dot(&l, &r))?)
        }
        (ArrayInner::F64(l), ArrayInner::F64(r)) => {
            ArrayInner::F64(map_sdnp(sdnp::dot(&l, &r))?)
        }
        (ArrayInner::C64(l), ArrayInner::C64(r)) => {
            ArrayInner::C64(map_sdnp(sdnp::dot(&l, &r))?)
        }
        _ => return Err(value_error("dot dtype mismatch")),
    };
    wrap_result(py, inner)
}

/// Flattened inner product with complex conjugation on the left operand.
///
/// Both operands are flattened before the contraction.
///
/// # Arguments
///
/// * `left` - Left operand (conjugated when complex).
/// * `right` - Right operand.
///
/// # Returns
///
/// Scalar or 0-D array result after conjugated dot product.
///
/// # Errors
///
/// * `TypeError` — incompatible operands.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1 + 1j, 2 - 1j])
/// b = np.array([1, 1])
/// assert np.vdot(a, b) == (3 + 0j)
/// ```
#[pyfunction]
pub fn vdot(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let l = coerce_array_like(left, None)?;
    let r = coerce_array_like(right, None)?;
    check_vdot(&l.inner, &r.inner)?;
    let dt = l.inner.dtype().promote(r.inner.dtype());
    let l = cast_inner(l.inner, dt)?;
    let r = cast_inner(r.inner, dt)?;
    let inner = match (l, r) {
        (ArrayInner::I64(l), ArrayInner::I64(r)) => {
            ArrayInner::I64(map_sdnp(sdnp::vdot(&l, &r))?)
        }
        (ArrayInner::F64(l), ArrayInner::F64(r)) => {
            ArrayInner::F64(map_sdnp(sdnp::vdot(&l, &r))?)
        }
        (ArrayInner::C64(l), ArrayInner::C64(r)) => {
            ArrayInner::C64(map_sdnp(sdnp::vdot(&l, &r))?)
        }
        _ => return Err(value_error("vdot dtype mismatch")),
    };
    wrap_result(py, inner)
}

/// Outer product of two 1-D vectors.
///
/// Produces a 2-D matrix `left[:, None] * right[None, :]`.
///
/// # Arguments
///
/// * `left` - 1-D left vector.
/// * `right` - 1-D right vector.
///
/// # Returns
///
/// A 2-D `Array` with shape `(len(left), len(right))`.
///
/// # Errors
///
/// * `TypeError` — non-1-D operands after coercion.
/// * `ValueError` — dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2])
/// b = np.array([3, 4, 5])
/// assert np.outer(a, b).shape == (2, 3)
/// ```
#[pyfunction]
pub fn outer(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let l = coerce_array_like(left, None)?;
    let r = coerce_array_like(right, None)?;
    let dt = l.inner.dtype().promote(r.inner.dtype());
    let l = cast_inner(l.inner, dt)?;
    let r = cast_inner(r.inner, dt)?;
    let inner = match (l, r) {
        (ArrayInner::I64(l), ArrayInner::I64(r)) => {
            ArrayInner::I64(map_sdnp(sdnp::outer(&l, &r))?)
        }
        (ArrayInner::F64(l), ArrayInner::F64(r)) => {
            ArrayInner::F64(map_sdnp(sdnp::outer(&l, &r))?)
        }
        (ArrayInner::C64(l), ArrayInner::C64(r)) => {
            ArrayInner::C64(map_sdnp(sdnp::outer(&l, &r))?)
        }
        _ => return Err(value_error("outer dtype mismatch")),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Extract a diagonal along offset `k` between `axis1` and `axis2`.
///
/// Returns a 1-D view copy of the selected diagonal elements.
///
/// # Arguments
///
/// * `a` - Input array with at least two dimensions.
/// * `offset` - Diagonal offset from the main diagonal.
/// * `axis1` - First axis defining the 2-D plane.
/// * `axis2` - Second axis defining the 2-D plane.
///
/// # Returns
///
/// A 1-D `Array` containing the diagonal elements.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axes or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2, 3], [4, 5, 6]])
/// assert np.diagonal(a).to_list() == [1, 5]
/// ```
#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
pub fn diagonal(
    py: Python<'_>,
    a: PyRef<PyArray>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("diagonal")?;
    let ndim = a.inner.ndim();
    check_diagonal_axes(ndim, axis1, axis2)?;
    let inner = match &a.inner {
        ArrayInner::I64(arr) => ArrayInner::I64(map_sdnp(sdnp::diagonal(
            arr, offset, axis1, axis2,
        ))?),
        ArrayInner::F64(arr) => ArrayInner::F64(map_sdnp(sdnp::diagonal(
            arr, offset, axis1, axis2,
        ))?),
        ArrayInner::C64(arr) => ArrayInner::C64(map_sdnp(sdnp::diagonal(
            arr, offset, axis1, axis2,
        ))?),
        ArrayInner::Bool(arr) => ArrayInner::Bool(map_sdnp(sdnp::diagonal(
            arr, offset, axis1, axis2,
        ))?),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Sum of diagonal elements; bool input promotes to int64 result.
///
/// Equivalent to `sum(diagonal(a, offset, axis1, axis2))`.
///
/// # Arguments
///
/// * `a` - Input array with at least two dimensions.
/// * `offset` - Diagonal offset from the main diagonal.
/// * `axis1` - First axis defining the 2-D plane.
/// * `axis2` - Second axis defining the 2-D plane.
///
/// # Returns
///
/// Trace scalar or 0-D array (int64 for bool input).
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axes or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])
/// assert np.trace(a) == 5
/// ```
#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
pub fn trace(
    py: Python<'_>,
    a: PyRef<PyArray>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("trace")?;
    let ndim = a.inner.ndim();
    check_diagonal_axes(ndim, axis1, axis2)?;
    let inner = match &a.inner {
        ArrayInner::I64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::trace(arr, offset, axis1, axis2))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::F64(map_sdnp(sdnp::trace(arr, offset, axis1, axis2))?)
        }
        ArrayInner::C64(arr) => {
            ArrayInner::C64(map_sdnp(sdnp::trace(arr, offset, axis1, axis2))?)
        }
        ArrayInner::Bool(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::trace(arr, offset, axis1, axis2))?)
        }
    };
    wrap_result(py, inner)
}

/// Register linalg callables on the extension module.
///
/// Adds `dot`, `matmul`, `vdot`, `outer`, `diagonal`, and `trace` to the
/// `sdnp` module object.
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
/// assert callable(np.matmul)
/// assert callable(np.trace)
/// ```
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    Ok(())
}
