//! Sorting free functions.

use pyo3::prelude::*;

use crate::array::{array_from_inner, PyArray};
use crate::coerce::coerce_optional_axis;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::check_optional_axis;

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn sort(
    py: Python<'_>,
    a: PyRef<PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
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

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn argsort(
    py: Python<'_>,
    a: PyRef<PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
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

#[pyfunction]
pub fn unique(py: Python<'_>, a: PyRef<PyArray>) -> PyResult<PyObject> {
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => ArrayInner::Bool(map_sdnp(sdnp::unique(arr))?),
        ArrayInner::I64(arr) => ArrayInner::I64(map_sdnp(sdnp::unique(arr))?),
        ArrayInner::F64(arr) => ArrayInner::F64(map_sdnp(sdnp::unique(arr))?),
        ArrayInner::C64(arr) => ArrayInner::C64(map_sdnp(sdnp::unique(arr))?),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    Ok(())
}
