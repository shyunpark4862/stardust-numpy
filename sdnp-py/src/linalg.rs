//! Linear algebra free functions.

use pyo3::prelude::*;

use crate::array::{array_from_inner, wrap_result, PyArray};
use crate::coerce::coerce_array_like;
use crate::dispatch::cast_inner;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::{
    check_diagonal_axes, check_dot, check_matmul, check_vdot,
};

pub fn py_matmul(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    matmul_impl(py, left, right)
}

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

#[pyfunction]
pub fn matmul(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    matmul_impl(py, left, right)
}

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

#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
pub fn diagonal(
    py: Python<'_>,
    a: PyRef<PyArray>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> PyResult<PyObject> {
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

#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
pub fn trace(
    py: Python<'_>,
    a: PyRef<PyArray>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> PyResult<PyObject> {
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    Ok(())
}
