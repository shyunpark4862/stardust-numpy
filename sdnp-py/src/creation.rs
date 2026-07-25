//! Array creation free functions.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sdnp::MeshgridIndexing;

use crate::array::{array_from_inner, wrap_result, PyArray};
use crate::coerce::{coerce_array_like, coerce_scalar, parse_shape};
use crate::dtype::PyDType;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::{
    check_arange_step, check_diag_input, check_finite_bounds,
    check_geomspace_bounds, check_logspace_base, check_meshgrid_arrays,
    check_meshgrid_indexing, check_triangle_input, require_pyarray,
};

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
        return Err(value_error(
            "0-dimensional arrays cannot be created from Python",
        ));
    }
    let arr = coerce_array_like(obj, dt)?;
    crate::array::into_pyobject(py, arr)
}

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

#[pyfunction]
#[pyo3(signature = (start, stop=None, step=1))]
pub fn arange(
    py: Python<'_>,
    start: i64,
    stop: Option<i64>,
    step: i64,
) -> PyResult<PyObject> {
    if stop.is_some() {
        check_arange_step(step)?;
    }
    let arr = match stop {
        None => map_sdnp(sdnp::arange_stop(start))?,
        Some(stop) => map_sdnp(sdnp::arange(start, stop, step))?,
    };
    crate::array::into_pyobject(py, array_from_inner(ArrayInner::I64(arr)))
}

#[pyfunction]
#[pyo3(signature = (start, stop, num, *, endpoint=true))]
pub fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> PyResult<PyObject> {
    check_finite_bounds("linspace", start, stop)?;
    wrap_result(
        py,
        ArrayInner::F64(map_sdnp(sdnp::linspace(start, stop, num, endpoint))?),
    )
}

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
    check_finite_bounds("logspace", start, stop)?;
    check_logspace_base(base)?;
    wrap_result(
        py,
        ArrayInner::F64(map_sdnp(sdnp::logspace(
            start, stop, num, endpoint, base,
        ))?),
    )
}

#[pyfunction]
#[pyo3(signature = (start, stop, num, *, endpoint=true))]
pub fn geomspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> PyResult<PyObject> {
    check_geomspace_bounds(start, stop)?;
    wrap_result(
        py,
        ArrayInner::F64(map_sdnp(sdnp::geomspace(start, stop, num, endpoint))?),
    )
}

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

#[pyfunction]
#[pyo3(signature = (array, k=0))]
pub fn tril(
    py: Python<'_>,
    array: PyRef<PyArray>,
    k: isize,
) -> PyResult<PyObject> {
    check_triangle_input("tril", &array.inner)?;
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

#[pyfunction]
#[pyo3(signature = (array, k=0))]
pub fn triu(
    py: Python<'_>,
    array: PyRef<PyArray>,
    k: isize,
) -> PyResult<PyObject> {
    check_triangle_input("triu", &array.inner)?;
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

#[pyfunction]
#[pyo3(signature = (array, k=0))]
pub fn diag(
    py: Python<'_>,
    array: PyRef<PyArray>,
    k: isize,
) -> PyResult<PyObject> {
    check_diag_input(&array.inner)?;
    let inner = match &array.inner {
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(sdnp::diag(a, k))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(sdnp::diag(a, k))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(sdnp::diag(a, k))?),
        ArrayInner::Bool(a) => {
            let i64_a = map_sdnp(a.astype())?;
            ArrayInner::I64(map_sdnp(sdnp::diag(&i64_a, k))?)
        }
    };
    wrap_result(py, inner)
}

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
    check_meshgrid_arrays(&validated)?;
    let first = require_pyarray(&tuple.get_item(0)?, "meshgrid")?;
    let idx = match indexing {
        "xy" => MeshgridIndexing::Xy,
        "ij" => MeshgridIndexing::Ij,
        _ => unreachable!("validated above"),
    };
    let inner = match &first.inner {
        ArrayInner::I64(_) => {
            let owned: Vec<_> = tuple
                .iter()
                .map(|item| {
                    let arr = require_pyarray(&item, "meshgrid")?;
                    match &arr.inner {
                        ArrayInner::I64(a) => Ok(a.clone()),
                        _ => Err(value_error("meshgrid dtype mismatch")),
                    }
                })
                .collect::<PyResult<_>>()?;
            let refs: Vec<_> = owned.iter().collect();
            let out = map_sdnp(sdnp::meshgrid(&refs, idx))?;
            out.into_iter().map(ArrayInner::I64).collect::<Vec<_>>()
        }
        ArrayInner::F64(_) => {
            let owned: Vec<_> = tuple
                .iter()
                .map(|item| {
                    let arr = require_pyarray(&item, "meshgrid")?;
                    match &arr.inner {
                        ArrayInner::F64(a) => Ok(a.clone()),
                        _ => Err(value_error("meshgrid dtype mismatch")),
                    }
                })
                .collect::<PyResult<_>>()?;
            let refs: Vec<_> = owned.iter().collect();
            let out = map_sdnp(sdnp::meshgrid(&refs, idx))?;
            out.into_iter().map(ArrayInner::F64).collect::<Vec<_>>()
        }
        ArrayInner::C64(_) => {
            let owned: Vec<_> = tuple
                .iter()
                .map(|item| {
                    let arr = require_pyarray(&item, "meshgrid")?;
                    match &arr.inner {
                        ArrayInner::C64(a) => Ok(a.clone()),
                        _ => Err(value_error("meshgrid dtype mismatch")),
                    }
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
