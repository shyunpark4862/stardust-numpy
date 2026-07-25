//! 0-D unwrap and scalar conversion at the Python boundary.

use pyo3::prelude::*;
use pyo3::types::PyComplex;
use sdnp::Complex64;

use crate::inner::{finish_array, ArrayInner};

#[derive(Clone, Debug)]
pub enum PyScalar {
    Bool(bool),
    I64(i64),
    F64(f64),
    C64(Complex64),
}

impl PyScalar {
    pub fn dtype(&self) -> crate::dtype::PyDType {
        use crate::dtype::PyDType;
        match self {
            PyScalar::Bool(_) => PyDType::Bool,
            PyScalar::I64(_) => PyDType::I64,
            PyScalar::F64(_) => PyDType::F64,
            PyScalar::C64(_) => PyDType::C64,
        }
    }
}

pub(crate) fn scalar_from_item(
    py: Python<'_>,
    scalar: PyScalar,
) -> PyResult<PyObject> {
    match scalar {
        PyScalar::Bool(v) => {
            Ok(pyo3::types::PyBool::new(py, v).to_owned().unbind().into())
        }
        PyScalar::I64(v) => {
            let bound = v.into_pyobject(py)?;
            Ok(bound.into_any().unbind())
        }
        PyScalar::F64(v) => {
            let bound = v.into_pyobject(py)?;
            Ok(bound.into_any().unbind())
        }
        PyScalar::C64(v) => Ok(PyComplex::from_doubles(py, v.re, v.im).into()),
    }
}

/// Return a scalar or `PyArray` — never expose 0-D arrays to users.
pub fn finish(py: Python<'_>, inner: ArrayInner) -> PyResult<PyObject> {
    finish_array(py, inner)
}
