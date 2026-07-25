//! Runtime dtype tag for Python-facing dispatch.

use pyo3::prelude::*;
use pyo3::types::PyType;
use sdnp::Complex64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PyDType {
    Bool,
    I64,
    F64,
    C64,
}

impl PyDType {
    pub fn promote(self, other: Self) -> Self {
        use PyDType::*;
        match (self, other) {
            (Bool, Bool) => Bool,
            (Bool, I64) | (I64, Bool) => I64,
            (Bool, F64) | (F64, Bool) | (I64, F64) | (F64, I64) => F64,
            (I64, I64) => I64,
            (F64, F64) => F64,
            (C64, _) | (_, C64) => C64,
        }
    }

    pub fn python_type<'py>(
        self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyType>> {
        match self {
            PyDType::Bool => Ok(py.get_type::<pyo3::types::PyBool>()),
            PyDType::I64 => Ok(py.get_type::<pyo3::types::PyInt>()),
            PyDType::F64 => Ok(py.get_type::<pyo3::types::PyFloat>()),
            PyDType::C64 => Ok(py.get_type::<pyo3::types::PyComplex>()),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PyDType::Bool => "bool",
            PyDType::I64 => "int64",
            PyDType::F64 => "float64",
            PyDType::C64 => "complex128",
        }
    }

    pub fn from_python_type(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<pyo3::types::PyBool>() {
            Ok(PyDType::Bool)
        } else if obj.is_instance_of::<pyo3::types::PyInt>() {
            Ok(PyDType::I64)
        } else if obj.is_instance_of::<pyo3::types::PyFloat>() {
            Ok(PyDType::F64)
        } else if obj.is_instance_of::<pyo3::types::PyComplex>() {
            Ok(PyDType::C64)
        } else if let Ok(name) = obj.getattr("__name__") {
            let name: String = name.extract()?;
            match name.as_str() {
                "bool" => Ok(PyDType::Bool),
                "int" => Ok(PyDType::I64),
                "float" => Ok(PyDType::F64),
                "complex" => Ok(PyDType::C64),
                other => Err(crate::error::type_error(format!(
                    "unsupported dtype: {other}"
                ))),
            }
        } else {
            Err(crate::error::type_error(format!(
                "unsupported dtype object: {obj}"
            )))
        }
    }
}

pub(crate) fn scalar_to_bool(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.extract::<bool>()
}

pub(crate) fn scalar_to_i64(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    obj.extract::<i64>()
}

pub(crate) fn scalar_to_f64(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    obj.extract::<f64>()
}

pub(crate) fn scalar_to_c64(obj: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    let complex = obj.downcast::<pyo3::types::PyComplex>()?;
    Ok(Complex64::new(complex.real(), complex.imag()))
}
