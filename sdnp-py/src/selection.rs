//! Selection free functions.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::array::{array_from_inner, PyArray};
use crate::coerce::coerce_array_like;
use crate::dispatch::cast_inner;
use crate::error::{map_sdnp, type_error, value_error};
use crate::inner::ArrayInner;
use crate::unwrap::PyScalar;
use crate::validate::{check_broadcastable, require_bool_array};

#[pyfunction]
#[pyo3(name = "where")]
pub fn where_(
    py: Python<'_>,
    condition: PyRef<PyArray>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    require_bool_array(&condition.inner, "where condition")?;
    let cond = condition.inner.as_bool()?;
    let x_arr = coerce_array_like(x, None)?;
    let y_arr = coerce_array_like(y, None)?;
    check_broadcastable(
        "where",
        &[
            condition.inner.shape(),
            x_arr.inner.shape(),
            y_arr.inner.shape(),
        ],
    )?;
    let dt = x_arr.inner.dtype().promote(y_arr.inner.dtype());
    let x_inner = cast_inner(x_arr.inner, dt)?;
    let y_inner = cast_inner(y_arr.inner, dt)?;
    let inner = match (x_inner, y_inner) {
        (ArrayInner::Bool(x), ArrayInner::Bool(y)) => {
            ArrayInner::Bool(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        (ArrayInner::I64(x), ArrayInner::I64(y)) => {
            ArrayInner::I64(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        (ArrayInner::F64(x), ArrayInner::F64(y)) => {
            ArrayInner::F64(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        (ArrayInner::C64(x), ArrayInner::C64(y)) => {
            ArrayInner::C64(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        _ => return Err(value_error("dtype mismatch in where")),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

#[pyfunction]
pub fn nonzero(py: Python<'_>, a: PyRef<PyArray>) -> PyResult<PyObject> {
    let coords = match &a.inner {
        ArrayInner::Bool(arr) => map_sdnp(sdnp::nonzero(arr))?,
        ArrayInner::I64(arr) => map_sdnp(sdnp::nonzero(arr))?,
        ArrayInner::F64(arr) => map_sdnp(sdnp::nonzero(arr))?,
        ArrayInner::C64(arr) => map_sdnp(sdnp::nonzero(arr))?,
    };
    let tuple = PyTuple::new(
        py,
        coords
            .into_iter()
            .map(|c| {
                crate::array::into_pyobject(
                    py,
                    array_from_inner(ArrayInner::I64(c)),
                )
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(tuple.into())
}

fn clip_bound(obj: &Bound<'_, PyAny>) -> PyResult<Option<PyScalar>> {
    if obj.is_none() {
        return Ok(None);
    }
    let inner = coerce_array_like(obj, None)?.inner;
    if inner.ndim() == 0 {
        let scalar = inner.item_scalar()?;
        if matches!(scalar, PyScalar::C64(_)) {
            return Err(type_error("clip bounds must be real scalar values"));
        }
        Ok(Some(scalar))
    } else {
        Err(value_error("clip bounds must be scalar values"))
    }
}

#[pyfunction]
pub fn clip(
    py: Python<'_>,
    a: PyRef<PyArray>,
    min: &Bound<'_, PyAny>,
    max: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let dt = a.inner.dtype();
    if dt == crate::dtype::PyDType::C64 {
        return Err(value_error("clip is not supported for complex arrays"));
    }
    let min_scalar = clip_bound(min)?;
    let max_scalar = clip_bound(max)?;
    let inner = match (&a.inner, min_scalar, max_scalar) {
        (ArrayInner::I64(a), min_s, max_s) => {
            let min_v = min_s.map(|s| match s {
                PyScalar::I64(v) => v,
                PyScalar::F64(v) => v as i64,
                PyScalar::Bool(v) => i64::from(v),
                _ => unreachable!(),
            });
            let max_v = max_s.map(|s| match s {
                PyScalar::I64(v) => v,
                PyScalar::F64(v) => v as i64,
                PyScalar::Bool(v) => i64::from(v),
                _ => unreachable!(),
            });
            ArrayInner::I64(map_sdnp(sdnp::clip(a, min_v, max_v))?)
        }
        (ArrayInner::F64(a), min_s, max_s) => {
            let min_v = min_s.map(|s| match s {
                PyScalar::F64(v) => v,
                PyScalar::I64(v) => v as f64,
                PyScalar::Bool(v) => {
                    if v {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            });
            let max_v = max_s.map(|s| match s {
                PyScalar::F64(v) => v,
                PyScalar::I64(v) => v as f64,
                PyScalar::Bool(v) => {
                    if v {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            });
            ArrayInner::F64(map_sdnp(sdnp::clip(a, min_v, max_v))?)
        }
        (ArrayInner::Bool(a), min_s, max_s) => {
            let min_v = min_s.map(|s| match s {
                PyScalar::Bool(v) => v,
                PyScalar::I64(v) => v != 0,
                PyScalar::F64(v) => v != 0.0,
                _ => unreachable!(),
            });
            let max_v = max_s.map(|s| match s {
                PyScalar::Bool(v) => v,
                PyScalar::I64(v) => v != 0,
                PyScalar::F64(v) => v != 0.0,
                _ => unreachable!(),
            });
            ArrayInner::Bool(map_sdnp(sdnp::clip(a, min_v, max_v))?)
        }
        _ => return Err(value_error("clip dtype mismatch")),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    Ok(())
}
