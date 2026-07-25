//! Coerce Python objects into internal arrays and scalars.

use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyComplex, PyFloat, PyInt, PyList, PySequence, PyTuple,
};
use sdnp::Array;
use sdnp::Complex64;

use crate::array::PyArray;
use crate::dtype::{
    scalar_to_bool, scalar_to_c64, scalar_to_f64, scalar_to_i64, PyDType,
};
use crate::error::{map_sdnp, type_error, value_error};
use crate::inner::ArrayInner;
use crate::unwrap::PyScalar;

pub fn parse_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(seq) = obj.downcast::<PySequence>() {
        let mut shape = Vec::with_capacity(seq.len()?);
        for item in seq.try_iter()? {
            let item = item?;
            let dim: isize = item.extract()?;
            if dim < 0 {
                return Err(value_error(
                    "shape dimensions must be non-negative",
                ));
            }
            shape.push(dim as usize);
        }
        if shape.is_empty() {
            return Err(value_error(
                "0-dimensional arrays cannot be created from Python",
            ));
        }
        validate_shape_size(&shape)?;
        return Ok(shape);
    }
    let dim: isize = obj.extract()?;
    if dim < 0 {
        return Err(value_error("shape dimensions must be non-negative"));
    }
    let shape = vec![dim as usize];
    validate_shape_size(&shape)?;
    Ok(shape)
}

fn validate_shape_size(shape: &[usize]) -> PyResult<()> {
    shape
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
        .ok_or_else(|| value_error("shape size overflows usize"))?;
    Ok(())
}

pub fn coerce_scalar(obj: &Bound<'_, PyAny>) -> PyResult<PyScalar> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(PyScalar::Bool(scalar_to_bool(obj)?));
    }
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        return Ok(PyScalar::I64(scalar_to_i64(obj)?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(PyScalar::F64(scalar_to_f64(obj)?));
    }
    if obj.is_instance_of::<PyComplex>() {
        return Ok(PyScalar::C64(scalar_to_c64(obj)?));
    }
    Err(type_error(format!(
        "expected a scalar (bool, int, float, complex), got {obj}"
    )))
}

pub fn coerce_array_like(
    obj: &Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyArray> {
    if let Ok(array) = obj.extract::<PyRef<PyArray>>() {
        let inner = array.inner.clone();
        return Ok(PyArray {
            inner: match dtype {
                Some(dtype) => crate::dispatch::cast_inner(inner, dtype)?,
                None => inner,
            },
        });
    }
    if is_python_scalar(obj) {
        let scalar = coerce_scalar(obj)?;
        let want = dtype.unwrap_or(scalar.dtype());
        return scalar_to_array(scalar, want);
    }
    let (shape, values, inferred) = flatten_nested(obj)?;
    if shape.is_empty() {
        return Err(value_error(
            "0-dimensional arrays cannot be created from Python",
        ));
    }
    let dt = dtype.unwrap_or(inferred);
    build_array_from_flat(&shape, &values, dt)
}

pub fn is_python_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyBool>()
        || (obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>())
        || obj.is_instance_of::<PyFloat>()
        || obj.is_instance_of::<PyComplex>()
}

fn scalar_to_array(scalar: PyScalar, dtype: PyDType) -> PyResult<PyArray> {
    let inner = match (scalar, dtype) {
        (PyScalar::Bool(v), PyDType::Bool) => {
            ArrayInner::Bool(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
        (PyScalar::Bool(v), PyDType::I64) => {
            ArrayInner::I64(map_sdnp(Array::from_vec(vec![i64::from(v)], &[]))?)
        }
        (PyScalar::Bool(v), PyDType::F64) => ArrayInner::F64(map_sdnp(
            Array::from_vec(vec![if v { 1.0 } else { 0.0 }], &[]),
        )?),
        (PyScalar::Bool(v), PyDType::C64) => {
            ArrayInner::C64(map_sdnp(Array::from_vec(
                vec![Complex64::new(if v { 1.0 } else { 0.0 }, 0.0)],
                &[],
            ))?)
        }
        (PyScalar::I64(_v), PyDType::Bool) => {
            return Err(value_error("cannot cast int scalar to bool array"))
        }
        (PyScalar::I64(v), PyDType::I64) => {
            ArrayInner::I64(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
        (PyScalar::I64(v), PyDType::F64) => {
            ArrayInner::F64(map_sdnp(Array::from_vec(vec![v as f64], &[]))?)
        }
        (PyScalar::I64(v), PyDType::C64) => ArrayInner::C64(map_sdnp(
            Array::from_vec(vec![Complex64::new(v as f64, 0.0)], &[]),
        )?),
        (PyScalar::F64(_v), PyDType::Bool) => {
            return Err(value_error("cannot cast float scalar to bool array"))
        }
        (PyScalar::F64(v), PyDType::I64) => {
            ArrayInner::I64(map_sdnp(Array::from_vec(vec![v as i64], &[]))?)
        }
        (PyScalar::F64(v), PyDType::F64) => {
            ArrayInner::F64(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
        (PyScalar::F64(v), PyDType::C64) => ArrayInner::C64(map_sdnp(
            Array::from_vec(vec![Complex64::new(v, 0.0)], &[]),
        )?),
        (PyScalar::C64(_v), PyDType::Bool)
        | (PyScalar::C64(_v), PyDType::I64) => {
            return Err(value_error("cannot cast complex scalar to real array"))
        }
        (PyScalar::C64(v), PyDType::F64) => {
            ArrayInner::F64(map_sdnp(Array::from_vec(vec![v.re], &[]))?)
        }
        (PyScalar::C64(v), PyDType::C64) => {
            ArrayInner::C64(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
    };
    Ok(PyArray { inner })
}

enum FlatValue {
    Bool(bool),
    I64(i64),
    F64(f64),
    C64(Complex64),
}

fn flatten_nested(
    obj: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, Vec<FlatValue>, PyDType)> {
    let mut values = Vec::new();
    let shape = infer_shape(obj, 0)?;
    flatten_into(obj, &shape, 0, &mut values)?;
    let dtype = infer_dtype_from_values(&values)?;
    Ok((shape, values, dtype))
}

fn infer_shape(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<Vec<usize>> {
    if is_python_scalar(obj) {
        if depth == 0 {
            return Err(value_error(
                "0-dimensional arrays cannot be created from Python",
            ));
        }
        return Ok(vec![]);
    }
    let seq = if let Ok(list) = obj.downcast::<PyList>() {
        list.as_sequence()
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        tuple.as_sequence()
    } else if let Ok(seq) = obj.downcast::<PySequence>() {
        seq
    } else {
        return Err(type_error(format!(
            "expected nested list/tuple, got {obj}"
        )));
    };
    let len = seq.len()?;
    if len == 0 {
        let mut shape = vec![0usize];
        shape.extend(std::iter::repeat_n(0, depth));
        return Ok(shape);
    }
    let sub = infer_shape(&seq.get_item(0)?, depth + 1)?;
    if !sub.is_empty() {
        let first_len = sub[0];
        for i in 1..len {
            let other = infer_shape(&seq.get_item(i)?, depth + 1)?;
            if other != sub {
                return Err(value_error("inhomogeneous nested sequence"));
            }
            if other[0] != first_len && !other.is_empty() {
                return Err(value_error("inhomogeneous nested sequence"));
            }
        }
    }
    let mut shape = vec![len];
    shape.extend(sub);
    Ok(shape)
}

fn flatten_into(
    obj: &Bound<'_, PyAny>,
    shape: &[usize],
    depth: usize,
    out: &mut Vec<FlatValue>,
) -> PyResult<()> {
    if depth == shape.len() {
        out.push(extract_flat_value(obj)?);
        return Ok(());
    }
    let seq = if let Ok(list) = obj.downcast::<PyList>() {
        list.as_sequence()
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        tuple.as_sequence()
    } else {
        obj.downcast::<PySequence>()?
    };
    for i in 0..seq.len()? {
        flatten_into(&seq.get_item(i)?, shape, depth + 1, out)?;
    }
    Ok(())
}

fn extract_flat_value(obj: &Bound<'_, PyAny>) -> PyResult<FlatValue> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(FlatValue::Bool(scalar_to_bool(obj)?));
    }
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        return Ok(FlatValue::I64(scalar_to_i64(obj)?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(FlatValue::F64(scalar_to_f64(obj)?));
    }
    if obj.is_instance_of::<PyComplex>() {
        return Ok(FlatValue::C64(scalar_to_c64(obj)?));
    }
    Err(type_error(format!("unsupported element type: {obj}")))
}

fn infer_dtype_from_values(values: &[FlatValue]) -> PyResult<PyDType> {
    let mut dt = PyDType::Bool;
    for v in values {
        dt = match v {
            FlatValue::Bool(_) => dt.max(PyDType::Bool),
            FlatValue::I64(_) => dt.max(PyDType::I64),
            FlatValue::F64(_) => dt.max(PyDType::F64),
            FlatValue::C64(_) => PyDType::C64,
        };
    }
    Ok(dt)
}

fn build_array_from_flat(
    shape: &[usize],
    values: &[FlatValue],
    dtype: PyDType,
) -> PyResult<PyArray> {
    match dtype {
        PyDType::Bool => {
            let data: Vec<bool> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => Ok(*b),
                    _ => {
                        Err(value_error("cannot convert mixed values to bool"))
                    }
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::Bool(map_sdnp(Array::from_vec(
                    data, shape,
                ))?),
            })
        }
        PyDType::I64 => {
            let data: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => Ok(i64::from(*b)),
                    FlatValue::I64(i) => Ok(*i),
                    _ => Err(value_error(
                        "cannot convert float/complex to int without dtype",
                    )),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::I64(map_sdnp(Array::from_vec(data, shape))?),
            })
        }
        PyDType::F64 => {
            let data: Vec<f64> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
                    FlatValue::I64(i) => Ok(*i as f64),
                    FlatValue::F64(f) => Ok(*f),
                    _ => Err(value_error(
                        "cannot convert complex to float without dtype",
                    )),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::F64(map_sdnp(Array::from_vec(data, shape))?),
            })
        }
        PyDType::C64 => {
            let data: Vec<Complex64> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => {
                        Ok(Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0))
                    }
                    FlatValue::I64(i) => Ok(Complex64::new(*i as f64, 0.0)),
                    FlatValue::F64(f) => Ok(Complex64::new(*f, 0.0)),
                    FlatValue::C64(c) => Ok(*c),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::C64(map_sdnp(Array::from_vec(data, shape))?),
            })
        }
    }
}

pub fn coerce_axis(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    obj.extract::<isize>()
        .map_err(|_| type_error("axis must be an integer"))
}

pub fn coerce_axes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(axis) = obj.extract::<isize>() {
        return Ok(vec![axis]);
    }
    let seq = obj.downcast::<PySequence>()?;
    let mut axes = Vec::with_capacity(seq.len()?);
    for item in seq.try_iter()? {
        let item = item?;
        axes.push(coerce_axis(&item)?);
    }
    Ok(axes)
}

pub fn coerce_optional_axes(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<isize>>> {
    match obj {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(None),
        Some(obj) => Ok(Some(coerce_axes(obj)?)),
    }
}

pub fn coerce_optional_axis(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<isize>> {
    match obj {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(None),
        Some(obj) => Ok(Some(coerce_axis(obj)?)),
    }
}
