//! Sorting and uniqueness free functions.
//!
//! Validates axis arguments and dtype support at the boundary, then dispatches
//! to typed `sdnp::sort`, `sdnp::argsort`, or `sdnp::unique` kernels.
//! Complex arrays are rejected here because ordering is undefined.

use pyo3::prelude::*;

use crate::array::{array_from_inner, PyArray};
use crate::coerce::coerce_optional_axis;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::check_optional_axis;

/// Return a sorted copy along `axis` (default: last axis).
///
/// Complex arrays are not supported because ordering is undefined.
///
/// # Arguments
///
/// * `a` - Input array (bool, int, or float).
/// * `axis` - Axis along which to sort, or `None` for the last axis.
///
/// # Returns
///
/// A sorted copy with the same shape and dtype as the input.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — complex dtype, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([3, 1, 2])
/// assert np.sort(a).to_list() == [1, 2, 3]
/// ```
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn sort(
    py: Python<'_>,
    a: PyRef<PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("sort")?;
    let ax = coerce_optional_axis(axis)?;
    check_optional_axis(ax, a.inner.ndim())?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::sort(arr, ax))?)
        }
        ArrayInner::I64(arr) => ArrayInner::I64(map_sdnp(sdnp::sort(arr, ax))?),
        ArrayInner::F64(arr) => ArrayInner::F64(map_sdnp(sdnp::sort(arr, ax))?),
        ArrayInner::C64(_) => {
            return Err(value_error("sort is not supported for complex arrays"))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return indices that would sort `a` along `axis`.
///
/// Output dtype is always int64. Complex arrays are not supported.
///
/// # Arguments
///
/// * `a` - Input array (bool, int, or float).
/// * `axis` - Axis along which to sort, or `None` for the last axis.
///
/// # Returns
///
/// An int64 index array with the same shape as the input.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — complex dtype, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([3, 1, 2])
/// assert np.argsort(a).to_list() == [1, 2, 0]
/// ```
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn argsort(
    py: Python<'_>,
    a: PyRef<PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("argsort")?;
    let ax = coerce_optional_axis(axis)?;
    check_optional_axis(ax, a.inner.ndim())?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argsort(arr, ax))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argsort(arr, ax))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argsort(arr, ax))?)
        }
        ArrayInner::C64(_) => {
            return Err(value_error(
                "argsort is not supported for complex arrays",
            ))
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return sorted unique elements (1-D result).
///
/// Flattens the input, removes duplicates, and returns a sorted 1-D array.
///
/// # Arguments
///
/// * `a` - Input array of any supported dtype.
///
/// # Returns
///
/// A 1-D `Array` of unique values in sorted order.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([2, 1, 2, 3, 1])
/// assert np.unique(a).to_list() == [1, 2, 3]
/// ```
#[pyfunction]
pub fn unique(py: Python<'_>, a: PyRef<PyArray>) -> PyResult<PyObject> {
    a.reject_zero_dim_input("unique")?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => ArrayInner::Bool(map_sdnp(sdnp::unique(arr))?),
        ArrayInner::I64(arr) => ArrayInner::I64(map_sdnp(sdnp::unique(arr))?),
        ArrayInner::F64(arr) => ArrayInner::F64(map_sdnp(sdnp::unique(arr))?),
        ArrayInner::C64(arr) => ArrayInner::C64(map_sdnp(sdnp::unique(arr))?),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Register sorting callables on the extension module.
///
/// Adds `sort`, `argsort`, and `unique` to the `sdnp` module object.
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
/// assert callable(np.sort)
/// assert callable(np.unique)
/// ```
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    Ok(())
}
