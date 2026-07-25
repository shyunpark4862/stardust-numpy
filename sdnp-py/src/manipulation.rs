//! Shape manipulation: concatenate, stack, vstack, hstack.
//!
//! Collects Python sequences into `Vec<ArrayInner>`, validates join rules at
//! the boundary, then dispatches to typed `sdnp` manipulation kernels. Each
//! arm matches on [`PyDType`] because Rust generics require monomorphization.

use pyo3::prelude::*;

use crate::array::array_from_inner;
use crate::coerce::collect_pyarrays;
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::{
    check_concatenate, check_hstack, check_same_dtype, check_stack,
    check_vstack,
};

/// Shared concatenate path after Python arrays are collected.
///
/// Validates join rules and dispatches to the typed `sdnp::concatenate`
/// kernel for the common element dtype.
///
/// # Arguments
///
/// * `arrays` - Homogeneous typed storage for each input array.
/// * `axis` - Existing axis along which to join.
///
/// # Returns
///
/// Concatenated storage wrapped in [`ArrayInner`].
///
/// # Errors
///
/// * `ValueError` — shape/dtype mismatch or core failure.
fn concatenate_inner(
    arrays: &[ArrayInner],
    axis: isize,
) -> PyResult<ArrayInner> {
    check_concatenate(arrays, axis)?;
    check_same_dtype(arrays, "concatenate")?;
    match arrays[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = arrays
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArrayInner::Bool(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
        crate::dtype::PyDType::I64 => {
            let refs = arrays
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArrayInner::I64(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = arrays
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch in concatenate")),
                })
                .collect::<PyResult<_>>()?;
            Ok(ArrayInner::F64(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = arrays
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch in concatenate")),
                })
                .collect::<PyResult<_>>()?;
            Ok(ArrayInner::C64(map_sdnp(sdnp::concatenate(&refs, axis))?))
        }
    }
}

/// Join arrays along an existing axis.
///
/// All inputs must share the same dtype and be joinable along `axis`.
///
/// # Arguments
///
/// * `arrays` - Sequence of `Array` objects to join.
/// * `axis` - Axis along which to concatenate (default 0).
///
/// # Returns
///
/// A new `Array` with expanded shape along `axis`.
///
/// # Errors
///
/// * `TypeError` — sequence contains non-array elements.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2])
/// b = np.array([3, 4])
/// assert np.concatenate([a, b]).to_list() == [1, 2, 3, 4]
/// ```
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn concatenate(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "concatenate")?;
    crate::array::into_pyobject(
        py,
        array_from_inner(concatenate_inner(&inners, axis)?),
    )
}

/// Stack arrays along a new axis.
///
/// All inputs must have identical shape and dtype. Inserts a new dimension
/// of length `len(arrays)` at `axis`.
///
/// # Arguments
///
/// * `arrays` - Sequence of `Array` objects with matching shape.
/// * `axis` - Position of the new axis (default 0).
///
/// # Returns
///
/// A new `Array` with rank increased by one.
///
/// # Errors
///
/// * `TypeError` — sequence contains non-array elements.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2])
/// b = np.array([3, 4])
/// out = np.stack([a, b])
/// assert out.shape == (2, 2)
/// ```
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "stack")?;
    check_stack(&inners, axis)?;
    let inner = match inners[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::Bool(map_sdnp(sdnp::stack(&refs, axis))?)
        }
        crate::dtype::PyDType::I64 => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::I64(map_sdnp(sdnp::stack(&refs, axis))?)
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::F64(map_sdnp(sdnp::stack(&refs, axis))?)
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::C64(map_sdnp(sdnp::stack(&refs, axis))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Vertically stack 1-D/2-D arrays (row-wise join).
///
/// Equivalent to `concatenate` along axis 0 after shape normalization.
///
/// # Arguments
///
/// * `arrays` - Sequence of 1-D or 2-D arrays with matching width.
///
/// # Returns
///
/// A 2-D `Array` with rows from each input.
///
/// # Errors
///
/// * `TypeError` — sequence contains non-array elements.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2])
/// b = np.array([3, 4])
/// assert np.vstack([a, b]).to_list() == [[1, 2], [3, 4]]
/// ```
#[pyfunction]
pub fn vstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "vstack")?;
    check_same_dtype(&inners, "vstack")?;
    check_vstack(&inners)?;
    let inner = match inners[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::Bool(map_sdnp(sdnp::vstack(&refs))?)
        }
        crate::dtype::PyDType::I64 => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::I64(map_sdnp(sdnp::vstack(&refs))?)
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::F64(map_sdnp(sdnp::vstack(&refs))?)
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::C64(map_sdnp(sdnp::vstack(&refs))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Horizontally stack 1-D/2-D arrays (column-wise join).
///
/// Equivalent to `concatenate` along the last axis after shape
/// normalization.
///
/// # Arguments
///
/// * `arrays` - Sequence of 1-D or 2-D arrays with matching height.
///
/// # Returns
///
/// A 2-D `Array` with columns from each input.
///
/// # Errors
///
/// * `TypeError` — sequence contains non-array elements.
/// * `ValueError` — shape/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2])
/// b = np.array([3, 4])
/// assert np.hstack([a, b]).to_list() == [1, 2, 3, 4]
/// ```
#[pyfunction]
pub fn hstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let inners = collect_pyarrays(arrays, "hstack")?;
    check_same_dtype(&inners, "hstack")?;
    check_hstack(&inners)?;
    let inner = match inners[0].dtype() {
        crate::dtype::PyDType::Bool => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_bool)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::Bool(map_sdnp(sdnp::hstack(&refs))?)
        }
        crate::dtype::PyDType::I64 => {
            let refs = inners
                .iter()
                .map(ArrayInner::as_i64)
                .collect::<PyResult<Vec<_>>>()?;
            ArrayInner::I64(map_sdnp(sdnp::hstack(&refs))?)
        }
        crate::dtype::PyDType::F64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::F64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::F64(map_sdnp(sdnp::hstack(&refs))?)
        }
        crate::dtype::PyDType::C64 => {
            let refs: Vec<_> = inners
                .iter()
                .map(|a| match a {
                    ArrayInner::C64(x) => Ok(x),
                    _ => Err(value_error("dtype mismatch")),
                })
                .collect::<PyResult<_>>()?;
            ArrayInner::C64(map_sdnp(sdnp::hstack(&refs))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Register manipulation callables on the extension module.
///
/// Adds `concatenate`, `stack`, `vstack`, and `hstack` to the `sdnp` module
/// object.
///
/// # Arguments
///
/// * `m` - Bound reference to the `sdnp` extension module.
///
/// # Returns
///
/// `Ok(())` when every callable is registered successfully.
///
/// # Errors
///
/// Returns `PyErr` if PyO3 function wrapping or registration fails.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert callable(np.concatenate)
/// assert callable(np.vstack)
/// ```
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    Ok(())
}
