//! Parse Python index objects into `IndexSpec` lists.

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use sdnp::{gather, scatter, scatter_array, IndexSpec};

use crate::array::PyArray;
use crate::coerce::coerce_scalar;
use crate::dispatch::cast_inner;
use crate::error::{index_error, map_sdnp, type_error, value_error};
use crate::inner::ArrayInner;
use crate::unwrap::{finish, PyScalar};
use crate::validate::check_slice_step;

pub fn get_item(
    py: Python<'_>,
    array: &PyArray,
    index: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let spec = parse_index(index)?;
    validate_index(array.inner.shape(), &spec)?;
    let inner = match &array.inner {
        ArrayInner::Bool(a) => ArrayInner::Bool(map_sdnp(gather(a, &spec))?),
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(gather(a, &spec))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(gather(a, &spec))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(gather(a, &spec))?),
    };
    finish(py, inner)
}

pub fn set_item(
    array: &mut PyArray,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let spec = parse_index(index)?;
    validate_index(array.inner.shape(), &spec)?;
    if let Ok(arr) = value.extract::<PyRef<PyArray>>() {
        return set_item_array(array, &spec, &arr.inner);
    }
    let scalar = coerce_scalar(value)?;
    set_item_scalar(array, &spec, scalar)
}

fn set_item_scalar(
    array: &mut PyArray,
    spec: &[IndexSpec],
    scalar: PyScalar,
) -> PyResult<()> {
    match (&mut array.inner, scalar) {
        (ArrayInner::Bool(a), PyScalar::Bool(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::I64(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::F64(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::C64(a), PyScalar::C64(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::Bool(a), PyScalar::I64(v)) => {
            map_sdnp(scatter(a, spec, v != 0)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::Bool(v)) => {
            map_sdnp(scatter(a, spec, i64::from(v))).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::I64(v)) => {
            map_sdnp(scatter(a, spec, v as f64)).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::Bool(v)) => {
            map_sdnp(scatter(a, spec, if v { 1.0 } else { 0.0 })).map(|_| ())
        }
        (ArrayInner::C64(a), other) => {
            map_sdnp(scatter(a, spec, scalar_to_c64(other)?)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::F64(v)) => {
            map_sdnp(scatter(a, spec, v as i64)).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::C64(v)) => {
            map_sdnp(scatter(a, spec, v.re)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::C64(v)) => {
            map_sdnp(scatter(a, spec, v.re as i64)).map(|_| ())
        }
        (ArrayInner::Bool(_), _) => {
            Err(value_error("cannot assign value to bool array"))
        }
    }
}

fn scalar_to_c64(s: PyScalar) -> PyResult<sdnp::Complex64> {
    use sdnp::Complex64;
    Ok(match s {
        PyScalar::Bool(v) => Complex64::new(if v { 1.0 } else { 0.0 }, 0.0),
        PyScalar::I64(v) => Complex64::new(v as f64, 0.0),
        PyScalar::F64(v) => Complex64::new(v, 0.0),
        PyScalar::C64(v) => v,
    })
}

fn set_item_array(
    array: &mut PyArray,
    spec: &[IndexSpec],
    values: &ArrayInner,
) -> PyResult<()> {
    match (&mut array.inner, values) {
        (ArrayInner::Bool(a), ArrayInner::Bool(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (ArrayInner::I64(a), ArrayInner::I64(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (ArrayInner::F64(a), ArrayInner::F64(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (ArrayInner::C64(a), ArrayInner::C64(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (dst, src) => {
            let promoted = cast_inner(src.clone(), dst.dtype())?;
            set_item_array(array, spec, &promoted)
        }
    }
}

fn parse_index(obj: &Bound<'_, PyAny>) -> PyResult<Vec<IndexSpec>> {
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut specs = Vec::new();
        for item in tuple.iter() {
            specs.extend(parse_index_item(&item)?);
        }
        return Ok(specs);
    }
    parse_index_item(obj)
}

fn parse_index_item(obj: &Bound<'_, PyAny>) -> PyResult<Vec<IndexSpec>> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(vec![IndexSpec::NewAxis]);
    }
    if obj.is(&py.Ellipsis()) {
        return Ok(vec![IndexSpec::Ellipsis]);
    }
    if let Ok(slice) = obj.downcast::<PySlice>() {
        return Ok(vec![parse_slice(slice)?]);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(vec![IndexSpec::Index(i)]);
    }
    if let Ok(arr) = obj.extract::<PyRef<PyArray>>() {
        return fancy_spec(&arr);
    }
    Err(type_error(format!(
        "index must be int, slice, ellipsis, None, or array; got {obj}"
    )))
}

fn parse_slice(slice: &Bound<'_, PySlice>) -> PyResult<IndexSpec> {
    let start = optional_index(&slice.getattr("start")?)?;
    let stop = optional_index(&slice.getattr("stop")?)?;
    let step = optional_index(&slice.getattr("step")?)?;
    check_slice_step(step)?;
    Ok(IndexSpec::Slice { start, stop, step })
}

fn optional_index(obj: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if obj.is_none() {
        return Ok(None);
    }
    Ok(Some(obj.extract::<i64>()?))
}

fn fancy_spec(arr: &PyRef<PyArray>) -> PyResult<Vec<IndexSpec>> {
    match &arr.inner {
        ArrayInner::Bool(a) => Ok(vec![IndexSpec::BoolArray(a.clone())]),
        ArrayInner::I64(a) => Ok(vec![IndexSpec::IntegerArray(a.clone())]),
        ArrayInner::F64(_) | ArrayInner::C64(_) => Err(type_error(
            "fancy index must be an integer or boolean array",
        )),
    }
}

fn axes_consumed(spec: &IndexSpec) -> usize {
    match spec {
        IndexSpec::NewAxis | IndexSpec::Ellipsis => 0,
        IndexSpec::BoolArray(mask) => mask.ndim(),
        IndexSpec::Index(_)
        | IndexSpec::Slice { .. }
        | IndexSpec::IntegerArray(_) => 1,
    }
}

fn validate_index(shape: &[usize], specs: &[IndexSpec]) -> PyResult<()> {
    let ellipsis_count = specs
        .iter()
        .filter(|spec| matches!(spec, IndexSpec::Ellipsis))
        .count();
    if ellipsis_count > 1 {
        return Err(index_error("an index can only have a single ellipsis"));
    }

    let used: usize = specs.iter().map(axes_consumed).sum();
    if used > shape.len() {
        return Err(index_error(format!(
            "too many indices for array: array is {}-dimensional, but {used} were indexed",
            shape.len()
        )));
    }
    let missing = shape.len() - used;
    let mut source_axis = 0usize;
    for spec in specs {
        match spec {
            IndexSpec::NewAxis => {}
            IndexSpec::Ellipsis => source_axis += missing,
            IndexSpec::BoolArray(mask) => {
                let end = source_axis + mask.ndim();
                if end > shape.len() || mask.shape() != &shape[source_axis..end]
                {
                    return Err(index_error(format!(
                        "boolean index shape {:?} does not match indexed dimensions {:?}",
                        mask.shape(),
                        shape.get(source_axis..end).unwrap_or(&[])
                    )));
                }
                source_axis = end;
            }
            IndexSpec::Index(_)
            | IndexSpec::Slice { .. }
            | IndexSpec::IntegerArray(_) => {
                source_axis += 1;
            }
        }
    }
    Ok(())
}
