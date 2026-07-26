//! The `Array` Python class, dunder methods, and iteration helpers.
//!
//! `PyArray` wraps [`ArrayInner`] and enforces a strict 0-D policy: internal
//! 0-D buffers exist for scalar ufunc paths, but Python never sees them as
//! arrays. Properties, indexing, operators, and `flat`/`__iter__` all reject
//! or unwrap 0-D inputs. Dtype-specific work is dispatched via enum matches.

use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyList};
use sdnp::Array;

use crate::coerce::coerce_reshape_shape;
use crate::dispatch::{py_binary, py_unary, BinaryOp, UnaryOp};
use crate::dtype::PyDType;
use crate::error::{map_sdnp, type_error};
use crate::index_parse::{get_item, set_item};
use crate::inner::ArrayInner;
use crate::repr::{array_repr, array_str};
use crate::unwrap::{finish, scalar_from_item, PyScalar};
use crate::validate::{
    check_permute_axes, check_reshape_shape, check_squeeze_axes,
};

/// NumPy-like n-dimensional array exposed to Python as `sdnp.Array`.
#[pyclass(name = "Array", module = "sdnp")]
#[derive(Clone)]
pub struct PyArray {
    pub(crate) inner: ArrayInner,
}

/// Re-wrap `slf` as `PyAny` for binary/unary operator dispatch.
///
/// Operator helpers need an owned `Bound<PyAny>` handle; cloning the inner
/// storage is cheaper than round-tripping through Python.
///
/// # Arguments
///
/// * `slf` - Borrowed `PyArray` reference from a dunder method.
///
/// # Returns
///
/// A `Bound<PyAny>` pointing at a fresh `PyArray` clone.
///
/// # Errors
///
/// Never fails; panics only if `PyArray` allocation fails internally.
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

/// Swap the last two axes (2-D matrix transpose).
///
/// Dispatches to the typed core `Array::transpose` for each storage variant.
/// Used by both `.T` and `.transpose()` on 2-D (and higher) inputs.
///
/// # Arguments
///
/// * `inner` - Source typed storage.
///
/// # Returns
///
/// A new `ArrayInner` with the last two axes exchanged.
///
/// # Errors
///
/// None; core transpose is infallible for valid ndim ≥ 2 inputs.
fn transpose_inner(inner: &ArrayInner) -> ArrayInner {
    match inner {
        ArrayInner::Bool(a) => ArrayInner::Bool(a.transpose()),
        ArrayInner::I64(a) => ArrayInner::I64(a.transpose()),
        ArrayInner::F64(a) => ArrayInner::F64(a.transpose()),
        ArrayInner::C64(a) => ArrayInner::C64(a.transpose()),
    }
}

impl PyArray {
    /// Reject 0-D arrays for APIs that require at least one dimension.
    ///
    /// Internal 0-D buffers exist for scalar ufunc paths, but Python callers
    /// must never observe them as sized or indexable arrays.
    ///
    /// # Arguments
    ///
    /// * `context` - Operation name embedded in the `TypeError` message.
    ///
    /// # Returns
    ///
    /// `Ok(())` when `ndim >= 1`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — `ndim == 0`.
    pub(crate) fn reject_zero_dim_input(&self, context: &str) -> PyResult<()> {
        if self.inner.ndim() == 0 {
            return Err(type_error(format!(
                "{context} does not accept a 0-dimensional array"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyArray {
    /// Tuple of dimension lengths along each axis.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A Python `tuple` of `int` lengths.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array (`shape` does not accept a 0-dimensional
    ///   array).
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        self.reject_zero_dim_input("shape")?;
        Ok(self.inner.shape().to_vec())
    }

    /// Number of array dimensions.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// An `int` ≥ 1.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        self.reject_zero_dim_input("ndim")?;
        Ok(self.inner.ndim())
    }

    /// Total number of elements (`product(shape)`).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// An `int` element count.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    #[getter]
    fn size(&self) -> PyResult<usize> {
        self.reject_zero_dim_input("size")?;
        Ok(self.inner.size())
    }

    /// Element type object (`bool`, `int`, `float`, or `complex`).
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// The Python scalar type matching storage.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.reject_zero_dim_input("dtype")?;
        Ok(self.inner.dtype().python_type(py)?.into())
    }

    /// View with the last two axes transposed (`.T` property).
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// A new `Array` with swapped trailing axes.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([[1, 2], [3, 4]])
    /// assert a.T.shape == (2, 2)
    /// ```
    #[getter]
    #[allow(non_snake_case)]
    fn T(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.reject_zero_dim_input("transpose")?;
        finish(py, transpose_inner(&self.inner))
    }

    /// Return a deep copy of the array.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// A new `Array` owning independent storage.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1, 2, 3])
    /// b = a.copy()
    /// b[0] = 99
    /// assert a[0] == 1
    /// ```
    fn copy(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.reject_zero_dim_input("copy")?;
        let inner = match &self.inner {
            ArrayInner::Bool(a) => ArrayInner::Bool(a.copy()),
            ArrayInner::I64(a) => ArrayInner::I64(a.copy()),
            ArrayInner::F64(a) => ArrayInner::F64(a.copy()),
            ArrayInner::C64(a) => ArrayInner::C64(a.copy()),
        };
        finish(py, inner)
    }

    /// Cast elements to `dtype`, returning a new array.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    /// * `dtype` - Target type (`bool`, `int`, `float`, or `complex`).
    ///
    /// # Returns
    ///
    /// A new `Array` with the requested storage dtype.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array or invalid dtype object.
    /// * `ValueError` — unsupported or lossy cast from the core.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1.5, 2.5])
    /// b = a.astype(int)
    /// assert b.to_list() == [1, 2]
    /// ```
    #[pyo3(signature = (dtype))]
    fn astype(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.reject_zero_dim_input("astype")?;
        let dt = PyDType::from_python_type(dtype)?;
        let inner = crate::dispatch::cast_inner(self.inner.clone(), dt)?;
        finish(py, inner)
    }

    /// Return a view with a new shape (total size must match).
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    /// * `shape` - Int or tuple of ints defining the new dimensions.
    ///
    /// # Returns
    ///
    /// A reshaped `Array`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array or non-integer shape component.
    /// * `ValueError` — incompatible total size or invalid shape.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1, 2, 3, 4])
    /// assert a.reshape((2, 2)).shape == (2, 2)
    /// ```
    #[pyo3(signature = (shape))]
    fn reshape(
        &self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.reject_zero_dim_input("reshape")?;
        let shape = coerce_reshape_shape(shape)?;
        check_reshape_shape(&shape, self.inner.size())?;
        let inner = match &self.inner {
            ArrayInner::Bool(a) => {
                ArrayInner::Bool(map_sdnp(a.reshape(&shape))?)
            }
            ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(a.reshape(&shape))?),
            ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(a.reshape(&shape))?),
            ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(a.reshape(&shape))?),
        };
        finish(py, inner)
    }

    /// Remove length-1 axes from the shape.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    /// * `axis` - Optional int or tuple of ints to squeeze selectively.
    ///
    /// # Returns
    ///
    /// A squeezed `Array`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array or invalid axis object.
    /// * `ValueError` — axis not of length 1 or out of range.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.zeros((1, 3, 1))
    /// assert a.squeeze().shape == (3,)
    /// ```
    #[pyo3(signature = (axis=None))]
    fn squeeze(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        self.reject_zero_dim_input("squeeze")?;
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
        finish(py, inner)
    }

    /// Swap the last two axes (same as the `.T` property).
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// A transposed `Array`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    fn transpose(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.reject_zero_dim_input("transpose")?;
        finish(py, transpose_inner(&self.inner))
    }

    /// Permute axes by the given reordering.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    /// * `axes` - Permutation of `range(ndim)` as int or tuple.
    ///
    /// # Returns
    ///
    /// A new `Array` with axes rearranged.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array or non-integer axis list.
    /// * `ValueError` — duplicate, missing, or out-of-range axis index.
    #[pyo3(signature = (axes))]
    fn permute_axes(
        &self,
        py: Python<'_>,
        axes: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.reject_zero_dim_input("permute_axes")?;
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
        finish(py, inner)
    }

    /// Convert the array to nested Python lists.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// A nested `list` mirroring array shape and scalar values.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.reject_zero_dim_input("to_list")?;
        nested_list(py, &self.inner)
    }

    /// NumPy-style string representation (`array([...], dtype=...)`).
    ///
    /// Long 1-D arrays are truncated with `...` for readability.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A display string suitable for interactive sessions.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1, 2, 3])
    /// assert repr(a).startswith("array([")
    /// ```
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        slf.reject_zero_dim_input("repr")?;
        let address = slf.as_ptr() as usize;
        array_repr(&slf.inner, address)
    }

    /// Zero-based R-style, width-bounded array display.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// R-style vector, matrix, or leading-axis page text. Every line is at
    /// most 80 columns; long rows, columns, and page sequences are abbreviated.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    fn __str__(&self) -> PyResult<String> {
        self.reject_zero_dim_input("str")?;
        array_str(&self.inner)
    }

    /// Return elements selected by `index`.
    ///
    /// Supports integer, slice, ellipsis, `None` (newaxis), tuple, and fancy
    /// integer or boolean index arrays.
    ///
    /// # Arguments
    ///
    /// * `index` - NumPy-style index object.
    ///
    /// # Returns
    ///
    /// A scalar (0-D unwrap) or a new `Array` view/copy.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array, invalid index type, or float/complex fancy
    ///   index.
    /// * `IndexError` — out-of-bounds index, too many indices, bad ellipsis,
    ///   or boolean mask shape mismatch.
    /// * `ValueError` — invalid slice step or core gather failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([[1, 2], [3, 4]])
    /// assert a[0, 1] == 2
    /// assert a[:, 0].to_list() == [1, 3]
    /// ```
    fn __getitem__(
        slf: PyRef<'_, Self>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        slf.reject_zero_dim_input("indexing")?;
        get_item(slf.py(), &slf, index)
    }

    /// Assign `value` to elements selected by `index`.
    ///
    /// Accepts Python scalars or same-shaped `Array` values. Cross-dtype
    /// scalar assignment follows NumPy-like coercion rules.
    ///
    /// # Arguments
    ///
    /// * `index` - NumPy-style index object.
    /// * `value` - Scalar or array to write.
    ///
    /// # Returns
    ///
    /// `None`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array, invalid index type, or float/complex fancy
    ///   index.
    /// * `IndexError` — out-of-bounds index or boolean mask shape mismatch.
    /// * `ValueError` — incompatible assignment dtype or core scatter failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.zeros((2, 2))
    /// a[0, :] = 1.0
    /// assert a[0, 0] == 1.0
    /// ```
    fn __setitem__(
        &mut self,
        index: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.reject_zero_dim_input("indexed assignment")?;
        set_item(self, index, value)
    }

    /// Length along axis 0 (`len(array)`).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// `shape[0]` when `ndim >= 1`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array (`len() of unsized object`).
    fn __len__(&self) -> PyResult<usize> {
        // NumPy: 0-D arrays are not sized for len().
        if self.inner.ndim() == 0 {
            return Err(type_error("len() of unsized object"));
        }
        Ok(self.inner.shape()[0])
    }

    /// Iterate over axis-0 sub-arrays (`for row in array`).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// An `axis0iter` yielding views along the leading dimension.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array (`iteration over a 0-D array`).
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<Axis0Iter>> {
        if slf.inner.ndim() == 0 {
            return Err(type_error("iteration over a 0-D array"));
        }
        Axis0Iter::new(slf)
    }

    /// Element-wise addition (`+`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1, 2, 3])
    /// assert (a + 1).to_list() == [2, 3, 4]
    /// ```
    fn __add__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Add)
    }

    /// Element-wise subtraction (`-`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([5, 4, 3])
    /// assert (a - 1).to_list() == [4, 3, 2]
    /// ```
    fn __sub__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Sub)
    }

    /// Element-wise multiplication (`*`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([2, 3, 4])
    /// assert (a * 2).to_list() == [4, 6, 8]
    /// ```
    fn __mul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Mul)
    }

    /// True division (`/`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast, division by zero, or ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([4.0, 6.0])
    /// assert (a / 2).to_list() == [2.0, 3.0]
    /// ```
    fn __truediv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Div)
    }

    /// Floor division (`//`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __floordiv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::FloorDiv)
    }

    /// Modulo (`%`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __mod__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Mod)
    }

    /// Element-wise power (`**`).
    ///
    /// Three-argument modular power (`pow(a, b, mod)`) is not supported.
    ///
    /// # Arguments
    ///
    /// * `other` - Exponent as array or Python scalar.
    /// * `modulus` - Must be absent or `None` (unsupported otherwise).
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — modular array power or incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __pow__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
        modulus: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // Python 3 pow(a, b, mod) is not supported for arrays.
        if modulus.is_some_and(|value| !value.is_none()) {
            return Err(type_error("modular array power is not supported"));
        }
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Pow)
    }

    /// Unary negation (`-a`).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported dtype for negation.
    /// * `ValueError` — ufunc failure from the core.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1, -2, 3])
    /// assert (-a).to_list() == [-1, 2, -3]
    /// ```
    fn __neg__(slf: PyRef<'_, Self>) -> PyResult<PyObject> {
        py_unary(slf.py(), &slf_any(slf), UnaryOp::Neg)
    }

    /// Element-wise absolute value (`abs(a)`).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported dtype for `abs`.
    /// * `ValueError` — ufunc failure from the core.
    fn __abs__(slf: PyRef<'_, Self>) -> PyResult<PyObject> {
        py_unary(slf.py(), &slf_any(slf), UnaryOp::Abs)
    }

    /// Matrix multiplication (`@`), NumPy `matmul` semantics.
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand `Array` operand.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — non-array operand, 0-D input, or scalar `@`.
    /// * `ValueError` — incompatible inner/batch dimensions.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([[1, 2], [3, 4]])
    /// b = np.array([[5, 6], [7, 8]])
    /// c = a @ b
    /// assert c.shape == (2, 2)
    /// ```
    fn __matmul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        crate::linalg::py_matmul(slf.py(), &slf_any(slf), other)
    }

    /// Element-wise equality (`==`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A `bool` `Array` or Python `bool` when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __eq__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Eq)
    }

    /// Element-wise inequality (`!=`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A `bool` `Array` or Python `bool` when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __ne__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Ne)
    }

    /// Element-wise less-than (`<`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A `bool` `Array` or Python `bool` when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __lt__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Lt)
    }

    /// Element-wise less-than-or-equal (`<=`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A `bool` `Array` or Python `bool` when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __le__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Le)
    }

    /// Element-wise greater-than (`>`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A `bool` `Array` or Python `bool` when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __gt__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Gt)
    }

    /// Element-wise greater-than-or-equal (`>=`).
    ///
    /// # Arguments
    ///
    /// * `other` - Array or Python scalar broadcast against `self`.
    ///
    /// # Returns
    ///
    /// A `bool` `Array` or Python `bool` when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __ge__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), &slf_any(slf), other, BinaryOp::Ge)
    }

    /// Reflected addition (`other + self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand array or Python scalar.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __radd__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Add)
    }

    /// Reflected subtraction (`other - self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand array or Python scalar.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __rsub__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Sub)
    }

    /// Reflected multiplication (`other * self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand array or Python scalar.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __rmul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Mul)
    }

    /// Reflected true division (`other / self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand array or Python scalar.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast, division by zero, or ufunc failure.
    fn __rtruediv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Div)
    }

    /// Reflected floor division (`other // self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand array or Python scalar.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __rfloordiv__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::FloorDiv)
    }

    /// Reflected modulo (`other % self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand array or Python scalar.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operand type or unsupported dtype mix.
    /// * `ValueError` — broadcast or ufunc failure from the core.
    fn __rmod__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        py_binary(slf.py(), other, &slf_any(slf), BinaryOp::Mod)
    }

    /// Reflected power (`other ** self`).
    ///
    /// Three-argument modular power is not supported.
    ///
    /// # Arguments
    ///
    /// * `other` - Base as array or Python scalar.
    /// * `modulus` - Must be absent or `None`.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — modular array power or incompatible operand type.
    /// * `ValueError` — broadcast or ufunc failure from the core.
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

    /// Reflected matrix multiplication (`other @ self`).
    ///
    /// # Arguments
    ///
    /// * `other` - Left-hand `Array` operand.
    ///
    /// # Returns
    ///
    /// A promoted `Array` or scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — non-array operand, 0-D input, or scalar `@`.
    /// * `ValueError` — incompatible inner/batch dimensions.
    fn __rmatmul__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        crate::linalg::py_matmul(slf.py(), other, &slf_any(slf))
    }

    /// C-contiguous flat iterator over every element.
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// A `flatiter` yielding Python scalars in row-major order.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([[1, 2], [3, 4]])
    /// assert list(a.flat) == [1, 2, 3, 4]
    /// ```
    #[getter]
    fn flat(slf: PyRef<'_, Self>) -> PyResult<Py<FlatIter>> {
        FlatIter::new(slf)
    }
}

/// Build nested Python lists mirroring array shape (for `to_list`).
///
/// Recursively walks axis 0, materializing scalars at the leaves. Complex
/// values become `complex` objects; other dtypes map to native Python types.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `inner` - Typed storage to convert.
///
/// # Returns
///
/// A nested `list` or bare scalar for internal 0-D paths.
///
/// # Errors
///
/// * Propagates gather/scalar conversion failures from nested slices.
fn nested_list<'py>(py: Python<'py>, inner: &ArrayInner) -> PyResult<PyObject> {
    let shape = inner.shape();
    if shape.is_empty() {
        // 0-D → bare scalar in list conversion path.
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

/// Take one index along axis 0 via core gather.
///
/// # Arguments
///
/// * `inner` - Source typed storage.
/// * `i` - Zero-based axis-0 index.
///
/// # Returns
///
/// A sub-array with the leading dimension removed.
///
/// # Errors
///
/// * `IndexError` / `ValueError` — out-of-bounds or gather failure.
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

/// Owning flat iterator state per element type.
enum FlatState {
    Bool(std::vec::IntoIter<bool>),
    I64(std::vec::IntoIter<i64>),
    F64(std::vec::IntoIter<f64>),
    C64(std::vec::IntoIter<sdnp::Complex64>),
}

/// C-contiguous flat iterator (`array.flat`).
#[pyclass(name = "flatiter", module = "sdnp")]
pub struct FlatIter {
    state: FlatState,
}

#[pymethods]
impl FlatIter {
    /// Return `self` (flat iterators are their own iterator).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// Borrowed reference to this iterator.
    ///
    /// # Errors
    ///
    /// None.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Yield the next scalar element, or `None` when exhausted.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// `Some(scalar)` or `None` at end of iteration.
    ///
    /// # Errors
    ///
    /// * Propagates scalar boxing failures for complex values.
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
    /// Construct a flat iterator over all elements of `array`.
    ///
    /// Materializes C-contiguous values into an owning Rust iterator.
    ///
    /// # Arguments
    ///
    /// * `array` - Source `PyArray` (must have `ndim >= 1`).
    ///
    /// # Returns
    ///
    /// A Python handle to the new `flatiter`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    fn new(array: PyRef<'_, PyArray>) -> PyResult<Py<Self>> {
        array.reject_zero_dim_input("flat iteration")?;
        let state = match &array.inner {
            ArrayInner::Bool(a) => FlatState::Bool(a.to_vec().into_iter()),
            ArrayInner::I64(a) => FlatState::I64(a.to_vec().into_iter()),
            ArrayInner::F64(a) => FlatState::F64(a.to_vec().into_iter()),
            ArrayInner::C64(a) => FlatState::C64(a.to_vec().into_iter()),
        };
        Py::new(array.py(), Self { state })
    }
}

/// Axis-0 slice iterator state per element type.
enum Axis0State {
    Bool(std::vec::IntoIter<Array<bool>>),
    I64(std::vec::IntoIter<Array<i64>>),
    F64(std::vec::IntoIter<Array<f64>>),
    C64(std::vec::IntoIter<Array<sdnp::Complex64>>),
}

/// Iterator yielding views along axis 0 (`for row in array`).
#[pyclass(name = "axis0iter", module = "sdnp")]
pub struct Axis0Iter {
    state: Axis0State,
}

#[pymethods]
impl Axis0Iter {
    /// Return `self` (axis-0 iterators are their own iterator).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// Borrowed reference to this iterator.
    ///
    /// # Errors
    ///
    /// None.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Yield the next axis-0 sub-array, or `None` when exhausted.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// `Some(Array)` view or `None` at end of iteration.
    ///
    /// # Errors
    ///
    /// * Propagates array boxing failures from `finish`.
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
    /// Construct an axis-0 iterator over `array`.
    ///
    /// Pre-collects axis-0 views so iteration can proceed without holding
    /// a borrow on the parent array.
    ///
    /// # Arguments
    ///
    /// * `array` - Source `PyArray` (must have `ndim >= 1`).
    ///
    /// # Returns
    ///
    /// A Python handle to the new `axis0iter`.
    ///
    /// # Errors
    ///
    /// * `TypeError` — 0-D array.
    fn new(array: PyRef<'_, PyArray>) -> PyResult<Py<Self>> {
        array.reject_zero_dim_input("iteration")?;
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

/// Wrap typed storage in a `PyArray` without Python allocation.
///
/// # Arguments
///
/// * `inner` - Fully constructed typed storage.
///
/// # Returns
///
/// An unregistered `PyArray` suitable for further Rust-side use.
///
/// # Errors
///
/// None.
pub fn array_from_inner(inner: ArrayInner) -> PyArray {
    PyArray { inner }
}

/// Convert a `PyArray` to Python, applying 0-D unwrap when needed.
///
/// 0-D results become bare Python scalars; ndim ≥ 1 results become `Array`.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `arr` - Wrapper around typed storage.
///
/// # Returns
///
/// A Python scalar or `Array` object.
///
/// # Errors
///
/// * Propagates scalar boxing or allocation failures.
pub fn into_pyobject(py: Python<'_>, arr: PyArray) -> PyResult<PyObject> {
    finish(py, arr.inner)
}

/// Expose ndim ≥ 1 arrays only; 0-D must go through [`finish`].
///
/// Internal guard for APIs that must never surface 0-D arrays to Python.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `arr` - Wrapper around typed storage.
///
/// # Returns
///
/// A Python `Array` object.
///
/// # Errors
///
/// * `TypeError` — internal error when `ndim == 0`.
pub(crate) fn into_pyobject_raw(
    py: Python<'_>,
    arr: PyArray,
) -> PyResult<PyObject> {
    if arr.inner.ndim() == 0 {
        return Err(type_error(
            "internal error: attempted to expose a 0-dimensional array",
        ));
    }
    Ok(Bound::new(py, arr)?.into_any().unbind())
}

/// Apply 0-D unwrap policy to raw [`ArrayInner`] results.
///
/// Convenience alias used by creation and ufunc entry points.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `inner` - Typed storage from a core operation.
///
/// # Returns
///
/// A Python scalar or `Array` object.
///
/// # Errors
///
/// * Propagates scalar boxing or allocation failures.
pub fn wrap_result(py: Python<'_>, inner: ArrayInner) -> PyResult<PyObject> {
    finish(py, inner)
}
