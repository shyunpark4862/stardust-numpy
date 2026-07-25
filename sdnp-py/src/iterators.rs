//! Iteration free functions.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sdnp::NdIndex;

use crate::error::map_sdnp;
use crate::validate::{
    check_broadcastable, check_nditer_operands, check_nditer_same_dtype,
    require_pyarray,
};

use crate::array::PyArray;
use crate::coerce::parse_shape;
use crate::inner::ArrayInner;
use crate::unwrap::{scalar_from_item, PyScalar};

#[pyclass(name = "ndindex", module = "sdnp")]
pub struct PyNdIndex {
    inner: NdIndex,
}

#[pymethods]
impl PyNdIndex {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.inner.next() {
            Some(idx) => Ok(Some(PyTuple::new(py, idx)?.into())),
            None => Ok(None),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyfunction]
#[pyo3(name = "ndindex")]
pub fn py_ndindex(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
) -> PyResult<Py<PyNdIndex>> {
    let shape = parse_shape(shape)?;
    let inner = map_sdnp(sdnp::ndindex(&shape))?;
    Py::new(py, PyNdIndex { inner })
}

#[pyclass(name = "ndenumerate", module = "sdnp")]
pub struct PyNdEnumerate {
    items: Vec<(Vec<usize>, PyScalar)>,
    index: usize,
}

#[pymethods]
impl PyNdEnumerate {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.index >= self.items.len() {
            return Ok(None);
        }
        let (idx, scalar) = self.items[self.index].clone();
        self.index += 1;
        let tuple = PyTuple::new(
            py,
            [PyTuple::new(py, idx)?.into(), scalar_from_item(py, scalar)?],
        )?;
        Ok(Some(tuple.into()))
    }
}

#[pyfunction]
pub fn ndenumerate(
    _py: Python<'_>,
    a: PyRef<PyArray>,
) -> PyResult<Py<PyNdEnumerate>> {
    let shape = a.inner.shape().to_vec();
    let mut indices = map_sdnp(sdnp::ndindex(&shape))?;
    let scalars: Vec<PyScalar> = match &a.inner {
        ArrayInner::Bool(arr) => {
            arr.to_vec().into_iter().map(PyScalar::Bool).collect()
        }
        ArrayInner::I64(arr) => {
            arr.to_vec().into_iter().map(PyScalar::I64).collect()
        }
        ArrayInner::F64(arr) => {
            arr.to_vec().into_iter().map(PyScalar::F64).collect()
        }
        ArrayInner::C64(arr) => {
            arr.to_vec().into_iter().map(PyScalar::C64).collect()
        }
    };
    let mut items = Vec::with_capacity(scalars.len());
    for scalar in scalars {
        if let Some(idx) = indices.next() {
            items.push((idx, scalar));
        }
    }
    Py::new(a.py(), PyNdEnumerate { items, index: 0 })
}

#[pyclass(name = "nditer", module = "sdnp")]
pub struct PyNdIter {
    items: Vec<Vec<PyScalar>>,
    index: usize,
    n_operands: usize,
}

#[pymethods]
impl PyNdIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.index >= self.items.len() {
            return Ok(None);
        }
        let vals = self.items[self.index].clone();
        self.index += 1;
        if self.n_operands == 1 {
            return Ok(Some(scalar_from_item(py, vals[0].clone())?));
        }
        let tuple = PyTuple::new(
            py,
            vals.into_iter()
                .map(|s| scalar_from_item(py, s))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(Some(tuple.into()))
    }
}

#[pyfunction]
#[pyo3(name = "nditer")]
pub fn py_nditer(
    py: Python<'_>,
    operands: &Bound<'_, PyAny>,
) -> PyResult<Py<PyNdIter>> {
    let tuple = operands.downcast::<PyTuple>()?;
    let n = tuple.len();
    check_nditer_operands(n)?;
    let arrays: Vec<PyRef<PyArray>> = tuple
        .iter()
        .map(|item| require_pyarray(&item, "nditer"))
        .collect::<PyResult<_>>()?;
    check_nditer_same_dtype(&arrays)?;
    let shapes = arrays
        .iter()
        .map(|array| array.inner.shape())
        .collect::<Vec<_>>();
    check_broadcastable("nditer", &shapes)?;
    let dt = arrays[0].inner.dtype();

    let steps: Vec<Vec<PyScalar>> = match (dt, n) {
        (crate::dtype::PyDType::Bool, 1) => {
            map_sdnp(sdnp::nditer(&[arrays[0].inner.as_bool()?]))?
                .map(|v| vec![PyScalar::Bool(v[0])])
                .collect()
        }
        (crate::dtype::PyDType::Bool, 2) => map_sdnp(sdnp::nditer(&[
            arrays[0].inner.as_bool()?,
            arrays[1].inner.as_bool()?,
        ]))?
        .map(|v| vec![PyScalar::Bool(v[0]), PyScalar::Bool(v[1])])
        .collect(),
        (crate::dtype::PyDType::I64, 1) => {
            map_sdnp(sdnp::nditer(&[arrays[0].inner.as_i64()?]))?
                .map(|v| vec![PyScalar::I64(v[0])])
                .collect()
        }
        (crate::dtype::PyDType::I64, 2) => map_sdnp(sdnp::nditer(&[
            arrays[0].inner.as_i64()?,
            arrays[1].inner.as_i64()?,
        ]))?
        .map(|v| vec![PyScalar::I64(v[0]), PyScalar::I64(v[1])])
        .collect(),
        (crate::dtype::PyDType::F64, 1) => {
            map_sdnp(sdnp::nditer(&[arrays[0].inner.as_f64()?]))?
                .map(|v| vec![PyScalar::F64(v[0])])
                .collect()
        }
        (crate::dtype::PyDType::F64, 2) => map_sdnp(sdnp::nditer(&[
            arrays[0].inner.as_f64()?,
            arrays[1].inner.as_f64()?,
        ]))?
        .map(|v| vec![PyScalar::F64(v[0]), PyScalar::F64(v[1])])
        .collect(),
        (crate::dtype::PyDType::C64, 1) => {
            map_sdnp(sdnp::nditer(&[arrays[0].inner.as_c64()?]))?
                .map(|v| vec![PyScalar::C64(v[0])])
                .collect()
        }
        (crate::dtype::PyDType::C64, 2) => map_sdnp(sdnp::nditer(&[
            arrays[0].inner.as_c64()?,
            arrays[1].inner.as_c64()?,
        ]))?
        .map(|v| vec![PyScalar::C64(v[0]), PyScalar::C64(v[1])])
        .collect(),
        _ => {
            return Err(crate::error::value_error(
                "nditer supports 1-2 operands with the same dtype",
            ))
        }
    };

    Py::new(
        py,
        PyNdIter {
            items: steps,
            index: 0,
            n_operands: n,
        },
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_ndindex, m)?)?;
    m.add_function(wrap_pyfunction!(ndenumerate, m)?)?;
    m.add_function(wrap_pyfunction!(py_nditer, m)?)?;
    Ok(())
}
