//! N-dimensional iteration helpers (`ndindex`, `ndenumerate`, `nditer`).
//!
//! Wraps core iteration utilities and materializes steps as Python iterators.
//! Scalar elements are unwrapped at yield time so callers never see 0-D arrays.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sdnp::NdIndex;

use crate::coerce::{parse_shape, require_pyarray};
use crate::error::map_sdnp;
use crate::validate::{
    check_broadcastable, check_nditer_operands, check_nditer_same_dtype,
};

use crate::array::PyArray;
use crate::inner::ArrayInner;
use crate::unwrap::{scalar_from_item, PyScalar};

/// Iterator over all index tuples for a shape.
///
/// Returned by [`py_ndindex`]. Implements the Python iterator protocol via
/// [`PyNdIndex::__iter__`] and [`PyNdIndex::__next__`].
///
/// # Arguments
///
/// Constructed internally from a validated shape vector; users call
/// `sdnp.ndindex(...)` instead of instantiating this type directly.
///
/// # Returns
///
/// Yields C-order multi-index tuples until exhausted.
///
/// # Errors
///
/// Shape parsing and allocation errors are raised by [`py_ndindex`].
#[pyclass(name = "ndindex", module = "sdnp")]
pub struct PyNdIndex {
    inner: NdIndex,
}

#[pymethods]
impl PyNdIndex {
    /// Return `self` (`ndindex` objects are their own iterator).
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

    /// Yield the next index tuple, or `None` when exhausted.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// `Some(tuple)` of axis indices or `None`.
    ///
    /// # Errors
    ///
    /// * Propagates tuple allocation failures.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.inner.next() {
            Some(idx) => Ok(Some(PyTuple::new(py, idx)?.into())),
            None => Ok(None),
        }
    }

    /// Number of index tuples remaining (including the current one).
    ///
    /// # Arguments
    ///
    /// None.
    ///
    /// # Returns
    ///
    /// Remaining iteration length as `int`.
    ///
    /// # Errors
    ///
    /// None.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Return an iterator of index tuples covering `shape`.
///
/// Equivalent to NumPy `numpy.ndindex`: yields every multi-index for the
/// given shape in C-order.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `shape` - Int or tuple of ints defining iteration bounds.
///
/// # Returns
///
/// An `ndindex` iterator object.
///
/// # Errors
///
/// * `ValueError` — invalid shape or core allocation failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert list(np.ndindex(2, 2)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
/// ```
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

/// Iterator yielding `(index_tuple, value)` pairs for every element.
///
/// Returned by [`ndenumerate`]. Each step unwraps the indexed element to a
/// Python scalar (never a 0-D [`PyArray`]).
///
/// # Arguments
///
/// Built from a rank ≥ 1 [`PyArray`]; construct via `sdnp.ndenumerate(a)`.
///
/// # Returns
///
/// Yields `(tuple, scalar)` pairs in C-order until exhausted.
///
/// # Errors
///
/// * `TypeError` — 0-D input (rejected at construction).
/// * `ValueError` — core iteration setup failure.
#[pyclass(name = "ndenumerate", module = "sdnp")]
pub struct PyNdEnumerate {
    items: Vec<(Vec<usize>, PyScalar)>,
    index: usize,
}

#[pymethods]
impl PyNdEnumerate {
    /// Return `self` (`ndenumerate` objects are their own iterator).
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

    /// Yield the next `(index, value)` pair, or `None` when exhausted.
    ///
    /// Values are bare Python scalars (0-D unwrap at yield time).
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// `Some((index_tuple, scalar))` or `None`.
    ///
    /// # Errors
    ///
    /// * Propagates tuple or scalar boxing failures.
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

/// Enumerate `(multi_index, scalar)` over all elements of `a`.
///
/// Equivalent to NumPy `numpy.ndenumerate`: pairs each C-order multi-index
/// with the corresponding scalar element.
///
/// # Arguments
///
/// * `_py` - Python interpreter token (unused; pairs built at construction).
/// * `a` - Source `Array` (`ndim >= 1`).
///
/// # Returns
///
/// An `ndenumerate` iterator object.
///
/// # Errors
///
/// * `TypeError` — 0-D array.
/// * `ValueError` — core `ndindex` failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[10, 20], [30, 40]])
/// assert list(np.ndenumerate(a)) == [((0, 0), 10), ((0, 1), 20),
///                                    ((1, 0), 30), ((1, 1), 40)]
/// ```
#[pyfunction]
pub fn ndenumerate(
    _py: Python<'_>,
    a: PyRef<PyArray>,
) -> PyResult<Py<PyNdEnumerate>> {
    a.reject_zero_dim_input("ndenumerate")?;
    let shape = a.inner.shape().to_vec();
    let mut indices = map_sdnp(sdnp::ndindex(&shape))?;
    // Materialize flat values once; pairs are zipped with ndindex steps.
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

/// Multi-operand iterator over broadcast-compatible arrays (1–2 operands).
///
/// Returned by [`py_nditer`]. Operands must share dtype and broadcast to a
/// common shape; each step yields scalars or a tuple of scalars.
///
/// # Arguments
///
/// Built from one or two [`PyArray`] inputs via `sdnp.nditer((a, b))`.
///
/// # Returns
///
/// Yields aligned elements in broadcast C-order until exhausted.
///
/// # Errors
///
/// * `TypeError` — non-array operand or 0-D input.
/// * `ValueError` — dtype mismatch or non-broadcastable shapes.
#[pyclass(name = "nditer", module = "sdnp")]
pub struct PyNdIter {
    items: Vec<Vec<PyScalar>>,
    index: usize,
    n_operands: usize,
}

#[pymethods]
impl PyNdIter {
    /// Return `self` (`nditer` objects are their own iterator).
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

    /// Yield the next aligned operand value(s), or `None` when exhausted.
    ///
    /// Single-operand iteration yields bare scalars; two-operand iteration
    /// yields `(scalar, scalar)` tuples.
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter token.
    ///
    /// # Returns
    ///
    /// `Some(scalar)`, `Some(tuple)`, or `None`.
    ///
    /// # Errors
    ///
    /// * Propagates scalar or tuple boxing failures.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.index >= self.items.len() {
            return Ok(None);
        }
        let vals = self.items[self.index].clone();
        self.index += 1;
        // Single-operand nditer yields bare scalars, not 1-tuples.
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

/// Iterate over aligned elements of 1–2 same-dtype broadcastable arrays.
///
/// Equivalent to a restricted NumPy `numpy.nditer`: operands must share
/// dtype and be mutually broadcastable; iteration follows broadcast shape.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `operands` - Tuple of one or two `Array` objects.
///
/// # Returns
///
/// An `nditer` iterator object.
///
/// # Errors
///
/// * `TypeError` — non-array operand or 0-D input.
/// * `ValueError` — wrong operand count, dtype mismatch, or non-broadcastable
///   shapes.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3])
/// b = np.array([10, 20, 30])
/// assert list(np.nditer((a, b))) == [(1, 10), (2, 20), (3, 30)]
/// ```
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

    // Match (dtype, operand count) — core nditer is monomorphized per type.
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

/// Register iteration callables and classes on the extension module.
///
/// Adds `ndindex`, `ndenumerate`, and `nditer` to the `sdnp` module table.
///
/// # Arguments
///
/// * `m` - Bound Python module object.
///
/// # Returns
///
/// `Ok(())` when all symbols are registered.
///
/// # Errors
///
/// * Propagates PyO3 registration failures.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_ndindex, m)?)?;
    m.add_function(wrap_pyfunction!(ndenumerate, m)?)?;
    m.add_function(wrap_pyfunction!(py_nditer, m)?)?;
    Ok(())
}
