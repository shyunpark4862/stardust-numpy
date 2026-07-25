//! `PyArray` Python class and array methods.

use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyList};
use sdnp::Array;

use crate::dispatch::{py_binary, py_unary, BinaryOp, UnaryOp};
use crate::dtype::PyDType;
use crate::error::{map_sdnp, type_error};
use crate::index_parse::{get_item, set_item};
use crate::inner::ArrayInner;
use crate::repr::array_repr;
use crate::unwrap::{finish, scalar_from_item, PyScalar};
use crate::validate::{
    check_permute_axes, check_squeeze_axes, parse_reshape_shape,
};

#[pyclass(name = "Array", module = "sdnp")]
#[derive(Clone)]
pub struct PyArray {
    pub(crate) inner: ArrayInner,
}

fn slf_any<'py>(slf: PyRef<'py, PyArray>) -> Bound<'py, PyAny> {
    Bound::new(
        slf.py(),
        PyArray {
            inner: slf.inner.clone(),
        },
    )
    .expect("PyArray")
    .into_any()
}

#[pymethods]
impl PyArray {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn strides(&self) -> Vec<isize> {
        self.inner.strides()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.inner.dtype().python_type(py)?.into())
    }

    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        Self {
            inner: match &self.inner {
                ArrayInner::Bool(a) => ArrayInner::Bool(a.transpose()),
                ArrayInner::I64(a) => ArrayInner::I64(a.transpose()),
                ArrayInner::F64(a) => ArrayInner::F64(a.transpose()),
                ArrayInner::C64(a) => ArrayInner::C64(a.transpose()),
            },
        }
    }

    fn copy(&self) -> Self {
        Self {
            inner: match &self.inner {
                ArrayInner::Bool(a) => ArrayInner::Bool(a.copy()),
                ArrayInner::I64(a) => ArrayInner::I64(a.copy()),
                ArrayInner::F64(a) => ArrayInner::F64(a.copy()),
                ArrayInner::C64(a) => ArrayInner::C64(a.copy()),
            },
        }
    }

    #[pyo3(signature = (dtype))]
    fn astype(&self, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dt = PyDType::from_python_type(dtype)?;
        let inner = crate::dispatch::cast_inner(self.inner.clone(), dt)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (shape))]
    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = parse_reshape_shape(shape, self.inner.size())?;
        let inner = match &self.inner {
            ArrayInner::Bool(a) => {
                ArrayInner::Bool(map_sdnp(a.reshape(&shape))?)
            }
            ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(a.reshape(&shape))?),
            ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(a.reshape(&shape))?),
            ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(a.reshape(&shape))?),
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (axis=None))]
    fn squeeze(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let axes = match axis {
            None => None,
            Some(obj) if obj.is_none() => None,
            Some(obj) => Some(crate::coerce::coerce_axes(obj)?),
        };
        check_squeeze_axes(&self.inner, axes.as_deref())?;
        let inner = match &self.inner {
            ArrayInner::Bool(a) => {
                ArrayInner::Bool(map_sdnp(a.squeeze(axes.as_deref()))?)
            }
            ArrayInner::I64(a) => {
                ArrayInner::I64(map_sdnp(a.squeeze(axes.as_deref()))?)
            }
            ArrayInner::F64(a) => {
                ArrayInner::F64(map_sdnp(a.squeeze(axes.as_deref()))?)
            }
            ArrayInner::C64(a) => {
                ArrayInner::C64(map_sdnp(a.squeeze(axes.as_deref()))?)
            }
        };
        Ok(Self { inner })
    }

    fn transpose(&self) -> Self {
        self.T()
    }

    #[pyo3(signature = (axes))]
    fn permute_axes(&self, axes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let axes = crate::coerce::coerce_axes(axes)?;
        check_permute_axes(&axes, self.inner.ndim())?;
        let inner = match &self.inner {
            ArrayInner::Bool(a) => {
                ArrayInner::Bool(map_sdnp(a.permute_axes(&axes))?)
            }
            ArrayInner::I64(a) => {
                ArrayInner::I64(map_sdnp(a.permute_axes(&axes))?)
            }
            ArrayInner::F64(a) => {
                ArrayInner::F64(map_sdnp(a.permute_axes(&axes))?)
            }
            ArrayInner::C64(a) => {
                ArrayInner::C64(map_sdnp(a.permute_axes(&axes))?)
            }
        };
        Ok(Self { inner })
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        nested_list(py, &self.inner)
    }

    fn __repr__(&self) -> PyResult<String> {
        array_repr(&self.inner)
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    fn __getitem__(
        slf: PyRef<'_, Self>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        get_item(slf.py(), &slf, index)
    }

    fn __setitem__(
        &mut self,
        index: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        set_item(self, index, value)
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.inner.ndim() == 0 {
            return Err(type_error("len() of unsized object"));
        }
        Ok(self.inner.shape()[0])
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<Axis0Iter>> {
        if slf.inner.ndim() == 0 {
            return Err(type_error("iteration over a 0-D array"));
        }
        Axis0Iter::new(slf)
    }

    fn __add__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Add)
    }

    fn __sub__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Sub)
    }

    fn __mul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Mul)
    }

    fn __truediv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Div)
    }

    fn __floordiv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::FloorDiv)
    }

    fn __mod__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Mod)
    }

    fn __pow__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        if modulus.is_some_and(|value| !value.is_none()) {
            return Err(type_error("modular array power is not supported"));
        }
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Pow)
    }

    fn __neg__(slf: PyRef<'_, Self>) -> PyResult<PyObject> {
        py_unary(slf.py(), &slf_any(slf), UnaryOp::Neg)
    }

    fn __abs__(slf: PyRef<'_, Self>) -> PyResult<PyObject> {
        py_unary(slf.py(), &slf_any(slf), UnaryOp::Abs)
    }

    fn __matmul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        crate::linalg::py_matmul(slf.py(), &slf_any(slf), other)
    }

    fn __eq__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Eq)
    }

    fn __ne__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Ne)
    }

    fn __lt__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Lt)
    }

    fn __le__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Le)
    }

    fn __gt__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Gt)
    }

    fn __ge__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Ge)
    }

    fn __radd__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Add)
    }

    fn __rsub__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Sub)
    }

    fn __rmul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Mul)
    }

    fn __rtruediv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Div)
    }

    fn __rfloordiv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::FloorDiv)
    }

    fn __rmod__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Mod)
    }

    fn __rpow__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        if modulus.is_some_and(|value| !value.is_none()) {
            return Err(type_error("modular array power is not supported"));
        }
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Pow)
    }

    fn __rmatmul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        crate::linalg::py_matmul(slf.py(), other, &slf_any(slf))
    }

    #[getter]
    fn flat(slf: PyRef<'_, Self>) -> PyResult<Py<FlatIter>> {
        FlatIter::new(slf)
    }
}

fn nested_list<'py>(py: Python<'py>, inner: &ArrayInner) -> PyResult<PyObject> {
    let shape = inner.shape();
    if shape.is_empty() {
        return scalar_from_item(py, inner.item_scalar()?);
    }
    if shape.len() == 1 {
        let list = PyList::empty(py);
        match inner {
            ArrayInner::Bool(a) => {
                for v in a.to_vec() {
                    list.append(v)?;
                }
            }
            ArrayInner::I64(a) => {
                for v in a.to_vec() {
                    list.append(v)?;
                }
            }
            ArrayInner::F64(a) => {
                for v in a.to_vec() {
                    list.append(v)?;
                }
            }
            ArrayInner::C64(a) => {
                for v in a.to_vec() {
                    list.append(PyComplex::from_doubles(py, v.re, v.im))?;
                }
            }
        }
        return Ok(list.into());
    }
    let list = PyList::empty(py);
    for i in 0..shape[0] {
        let sub = slice_axis(inner, i)?;
        list.append(nested_list(py, &sub)?)?;
    }
    Ok(list.into())
}

fn slice_axis(inner: &ArrayInner, i: usize) -> PyResult<ArrayInner> {
    use sdnp::{gather, IndexSpec};
    let spec = vec![IndexSpec::Index(i as i64)];
    Ok(match inner {
        ArrayInner::Bool(a) => ArrayInner::Bool(map_sdnp(gather(a, &spec))?),
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(gather(a, &spec))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(gather(a, &spec))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(gather(a, &spec))?),
    })
}

enum FlatState {
    Bool(std::vec::IntoIter<bool>),
    I64(std::vec::IntoIter<i64>),
    F64(std::vec::IntoIter<f64>),
    C64(std::vec::IntoIter<sdnp::Complex64>),
}

#[pyclass(name = "flatiter", module = "sdnp")]
pub struct FlatIter {
    state: FlatState,
}

#[pymethods]
impl FlatIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let scalar = match &mut self.state {
            FlatState::Bool(it) => it.next().map(PyScalar::Bool),
            FlatState::I64(it) => it.next().map(PyScalar::I64),
            FlatState::F64(it) => it.next().map(PyScalar::F64),
            FlatState::C64(it) => it.next().map(PyScalar::C64),
        };
        match scalar {
            Some(s) => Ok(Some(scalar_from_item(py, s)?)),
            None => Ok(None),
        }
    }
}

impl FlatIter {
    fn new(array: PyRef<'_, PyArray>) -> PyResult<Py<Self>> {
        let state = match &array.inner {
            ArrayInner::Bool(a) => FlatState::Bool(a.to_vec().into_iter()),
            ArrayInner::I64(a) => FlatState::I64(a.to_vec().into_iter()),
            ArrayInner::F64(a) => FlatState::F64(a.to_vec().into_iter()),
            ArrayInner::C64(a) => FlatState::C64(a.to_vec().into_iter()),
        };
        Py::new(array.py(), Self { state })
    }
}

enum Axis0State {
    Bool(std::vec::IntoIter<Array<bool>>),
    I64(std::vec::IntoIter<Array<i64>>),
    F64(std::vec::IntoIter<Array<f64>>),
    C64(std::vec::IntoIter<Array<sdnp::Complex64>>),
}

#[pyclass(name = "axis0iter", module = "sdnp")]
pub struct Axis0Iter {
    state: Axis0State,
}

#[pymethods]
impl Axis0Iter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let item = match &mut self.state {
            Axis0State::Bool(it) => it
                .next()
                .map(|a| finish(py, ArrayInner::Bool(a)))
                .transpose()?,
            Axis0State::I64(it) => it
                .next()
                .map(|a| finish(py, ArrayInner::I64(a)))
                .transpose()?,
            Axis0State::F64(it) => it
                .next()
                .map(|a| finish(py, ArrayInner::F64(a)))
                .transpose()?,
            Axis0State::C64(it) => it
                .next()
                .map(|a| finish(py, ArrayInner::C64(a)))
                .transpose()?,
        };
        Ok(item)
    }
}

impl Axis0Iter {
    fn new(array: PyRef<'_, PyArray>) -> PyResult<Py<Self>> {
        let state = match &array.inner {
            ArrayInner::Bool(a) => {
                Axis0State::Bool(a.iter_axis0().collect::<Vec<_>>().into_iter())
            }
            ArrayInner::I64(a) => {
                Axis0State::I64(a.iter_axis0().collect::<Vec<_>>().into_iter())
            }
            ArrayInner::F64(a) => {
                Axis0State::F64(a.iter_axis0().collect::<Vec<_>>().into_iter())
            }
            ArrayInner::C64(a) => {
                Axis0State::C64(a.iter_axis0().collect::<Vec<_>>().into_iter())
            }
        };
        Py::new(array.py(), Self { state })
    }
}

pub fn array_from_inner(inner: ArrayInner) -> PyArray {
    PyArray { inner }
}

pub fn into_pyobject(py: Python<'_>, arr: PyArray) -> PyResult<PyObject> {
    Ok(Bound::new(py, arr)?.into_any().unbind())
}

pub fn wrap_result(py: Python<'_>, inner: ArrayInner) -> PyResult<PyObject> {
    finish(py, inner)
}
