//! Manipulation free functions.

use pyo3::prelude::*;

use crate::array::array_from_inner;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::{
    check_concatenate, check_hstack, check_same_dtype, check_stack,
    check_vstack, collect_pyarrays,
};

fn concatenate_inner(
    arrays: &[ArrayInner],
    axis: isize,
) -> PyResult<ArrayInner> {
    let axis = check_concatenate(arrays, axis)?;
    check_same_dtype(arrays, "concatenate")?;
    match arrays[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = arrays
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArrayInner::Bool(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
        crate::dtype::PyDType::I64 => {
            let refs = arrays
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArrayInner::I64(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = arrays
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch in concatenate")),
                })
                .collect::<PyResult<_>>()?;
            Ok(ArrayInner::F64(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = arrays
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch in concatenate")),
                })
                .collect::<PyResult<_>>()?;
            Ok(ArrayInner::C64(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn concatenate(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "concatenate")?;
    crate::array::into_pyobject(
        py,
        array_from_inner(concatenate_inner(&inners, axis)?),
    )
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "stack")?;
    check_stack(&inners, axis)?;
    let inner = match inners[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::Bool(map_sdnp(sdnp::stack(&refs, axis))?)
        }
        crate::dtype::PyDType::I64 => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::I64(map_sdnp(sdnp::stack(&refs, axis))?)
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::F64(map_sdnp(sdnp::stack(&refs, axis))?)
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::C64(map_sdnp(sdnp::stack(&refs, axis))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

#[pyfunction]
pub fn vstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "vstack")?;
    check_same_dtype(&inners, "vstack")?;
    check_vstack(&inners)?;
    let inner = match inners[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::Bool(map_sdnp(sdnp::vstack(&refs))?)
        }
        crate::dtype::PyDType::I64 => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::I64(map_sdnp(sdnp::vstack(&refs))?)
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::F64(map_sdnp(sdnp::vstack(&refs))?)
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::C64(map_sdnp(sdnp::vstack(&refs))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

#[pyfunction]
pub fn hstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "hstack")?;
    check_same_dtype(&inners, "hstack")?;
    check_hstack(&inners)?;
    let inner = match inners[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::Bool(map_sdnp(sdnp::hstack(&refs))?)
        }
        crate::dtype::PyDType::I64 => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::I64(map_sdnp(sdnp::hstack(&refs))?)
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::F64(map_sdnp(sdnp::hstack(&refs))?)
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::C64(map_sdnp(sdnp::hstack(&refs))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    Ok(())
}
