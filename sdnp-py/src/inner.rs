//! Internal tagged array storage and dtype dispatch helpers.

use pyo3::prelude::*;
use sdnp::Array;
use sdnp::Complex64;

use crate::dtype::PyDType;
use crate::error::{map_sdnp, value_error};
use crate::unwrap::{scalar_from_item, PyScalar};

#[derive(Clone)]
pub enum ArrayInner {
    Bool(Array<bool>),
    I64(Array<i64>),
    F64(Array<f64>),
    C64(Array<Complex64>),
}

impl ArrayInner {
    pub fn dtype(&self) -> PyDType {
        match self {
            ArrayInner::Bool(_) => PyDType::Bool,
            ArrayInner::I64(_) => PyDType::I64,
            ArrayInner::F64(_) => PyDType::F64,
            ArrayInner::C64(_) => PyDType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayInner::Bool(a) => a.shape(),
            ArrayInner::I64(a) => a.shape(),
            ArrayInner::F64(a) => a.shape(),
            ArrayInner::C64(a) => a.shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn size(&self) -> usize {
        match self {
            ArrayInner::Bool(a) => a.size(),
            ArrayInner::I64(a) => a.size(),
            ArrayInner::F64(a) => a.size(),
            ArrayInner::C64(a) => a.size(),
        }
    }

    pub fn strides(&self) -> Vec<isize> {
        match self {
            ArrayInner::Bool(a) => a.strides().to_vec(),
            ArrayInner::I64(a) => a.strides().to_vec(),
            ArrayInner::F64(a) => a.strides().to_vec(),
            ArrayInner::C64(a) => a.strides().to_vec(),
        }
    }

    pub fn item_scalar(&self) -> PyResult<PyScalar> {
        match self {
            ArrayInner::Bool(a) => Ok(PyScalar::Bool(map_sdnp(a.item())?)),
            ArrayInner::I64(a) => Ok(PyScalar::I64(map_sdnp(a.item())?)),
            ArrayInner::F64(a) => Ok(PyScalar::F64(map_sdnp(a.item())?)),
            ArrayInner::C64(a) => Ok(PyScalar::C64(map_sdnp(a.item())?)),
        }
    }

    pub fn as_bool(&self) -> PyResult<&Array<bool>> {
        match self {
            ArrayInner::Bool(a) => Ok(a),
            _ => Err(value_error("expected bool array")),
        }
    }

    pub fn as_i64(&self) -> PyResult<&Array<i64>> {
        match self {
            ArrayInner::I64(a) => Ok(a),
            _ => Err(value_error("expected int array")),
        }
    }

    pub fn as_f64(&self) -> PyResult<&Array<f64>> {
        match self {
            ArrayInner::F64(a) => Ok(a),
            _ => Err(value_error("expected float array")),
        }
    }

    pub fn as_c64(&self) -> PyResult<&Array<Complex64>> {
        match self {
            ArrayInner::C64(a) => Ok(a),
            _ => Err(value_error("expected complex array")),
        }
    }
}

pub(crate) fn scalar_to_inner(scalar: &PyScalar) -> ArrayInner {
    match scalar {
        PyScalar::Bool(v) => {
            ArrayInner::Bool(Array::from_vec(vec![*v], &[]).expect("0-D bool"))
        }
        PyScalar::I64(v) => {
            ArrayInner::I64(Array::from_vec(vec![*v], &[]).expect("0-D i64"))
        }
        PyScalar::F64(v) => {
            ArrayInner::F64(Array::from_vec(vec![*v], &[]).expect("0-D f64"))
        }
        PyScalar::C64(v) => {
            ArrayInner::C64(Array::from_vec(vec![*v], &[]).expect("0-D c64"))
        }
    }
}

pub(crate) fn finish_array(
    py: Python<'_>,
    inner: ArrayInner,
) -> PyResult<PyObject> {
    if inner.ndim() == 0 {
        scalar_from_item(py, inner.item_scalar()?)
    } else {
        Ok(crate::array::into_pyobject(
            py,
            crate::array::PyArray { inner },
        )?)
    }
}
