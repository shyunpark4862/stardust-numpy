//! Boolean indexing, selection, and clipping free functions.
//!
//! Coerces operands to [`ArrayInner`], checks broadcast rules and dtype
//! constraints at the Python boundary, then calls typed `sdnp` selection
//! kernels. Results pass through the 0-D unwrap policy when applicable.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::array::{array_from_inner, PyArray};
use crate::coerce::coerce_array_like;
use crate::dispatch::cast_inner;
use crate::error::{map_sdnp, type_error, value_error};
use crate::inner::ArrayInner;
use crate::unwrap::PyScalar;
use crate::validate::require_bool_array;

/// Select from `x` or `y` element-wise where `condition` is true.
///
/// Branch dtypes promote to a common type before the typed kernel runs.
/// All three operands must broadcast to the same shape.
///
/// # Arguments
///
/// * `condition` - Boolean mask array.
/// * `x` - Values chosen where `condition` is true.
/// * `y` - Values chosen where `condition` is false.
///
/// # Returns
///
/// An `Array` with the promoted branch dtype.
///
/// # Errors
///
/// * `TypeError` — 0-D condition or non-boolean mask.
/// * `ValueError` — broadcast/dtype mismatch or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// cond = np.array([True, False, True])
/// assert np.where(cond, 1, 0).to_list() == [1, 0, 1]
/// ```
#[pyfunction]
#[pyo3(name = "where")]
pub fn where_(
    py: Python<'_>,
    condition: PyRef<PyArray>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    condition.reject_zero_dim_input("where")?;
    require_bool_array(&condition.inner, "where condition")?;
    let cond = condition.inner.as_bool()?;
    let x_arr = coerce_array_like(x, None)?;
    let y_arr = coerce_array_like(y, None)?;
    // Promote branch dtypes before calling the typed where_ kernel.
    let dt = x_arr.inner.dtype().promote(y_arr.inner.dtype());
    let x_inner = cast_inner(x_arr.inner, dt)?;
    let y_inner = cast_inner(y_arr.inner, dt)?;
    let inner = match (x_inner, y_inner) {
        (ArrayInner::Bool(x), ArrayInner::Bool(y)) => {
            ArrayInner::Bool(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        (ArrayInner::I64(x), ArrayInner::I64(y)) => {
            ArrayInner::I64(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        (ArrayInner::F64(x), ArrayInner::F64(y)) => {
            ArrayInner::F64(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        (ArrayInner::C64(x), ArrayInner::C64(y)) => {
            ArrayInner::C64(map_sdnp(sdnp::where_(cond, &x, &y))?)
        }
        _ => return Err(value_error("dtype mismatch in where")),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Return coordinate arrays for every non-zero element.
///
/// Output is a tuple of 1-D int64 index arrays, one per dimension.
///
/// # Arguments
///
/// * `a` - Input array of any supported dtype.
///
/// # Returns
///
/// A tuple of int64 coordinate arrays `(row_indices, col_indices, ...)`.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[0, 1], [2, 0]])
/// rows, cols = np.nonzero(a)
/// assert rows.to_list() == [0, 1]
/// assert cols.to_list() == [1, 0]
/// ```
#[pyfunction]
pub fn nonzero(py: Python<'_>, a: PyRef<PyArray>) -> PyResult<PyObject> {
    a.reject_zero_dim_input("nonzero")?;
    let coords = match &a.inner {
        ArrayInner::Bool(arr) => map_sdnp(sdnp::nonzero(arr))?,
        ArrayInner::I64(arr) => map_sdnp(sdnp::nonzero(arr))?,
        ArrayInner::F64(arr) => map_sdnp(sdnp::nonzero(arr))?,
        ArrayInner::C64(arr) => map_sdnp(sdnp::nonzero(arr))?,
    };
    let tuple = PyTuple::new(
        py,
        coords
            .into_iter()
            .map(|c| {
                crate::array::into_pyobject(
                    py,
                    array_from_inner(ArrayInner::I64(c)),
                )
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(tuple.into())
}

/// Parse a clip bound: `None`, or a 0-D real scalar wrapped as array.
///
/// # Arguments
///
/// * `obj` - Python bound (`None`, scalar, or 0-D array).
///
/// # Returns
///
/// `None` when unbounded on that side, otherwise a coerced real scalar.
///
/// # Errors
///
/// * `TypeError` — complex scalar bound.
/// * `ValueError` — non-scalar bound array.
fn clip_bound(obj: &Bound<'_, PyAny>) -> PyResult<Option<PyScalar>> {
    if obj.is_none() {
        return Ok(None);
    }
    let inner = coerce_array_like(obj, None)?.inner;
    if inner.ndim() == 0 {
        let scalar = inner.item_scalar()?;
        if matches!(scalar, PyScalar::C64(_)) {
            return Err(type_error("clip bounds must be real scalar values"));
        }
        Ok(Some(scalar))
    } else {
        Err(value_error("clip bounds must be scalar values"))
    }
}

/// Clip values to `[min, max]`; `None` means no bound on that side.
///
/// Complex arrays are not supported. Scalar bounds coerce to the array dtype.
///
/// # Arguments
///
/// * `a` - Input array (bool, int, or float).
/// * `min` - Lower bound scalar or `None`.
/// * `max` - Upper bound scalar or `None`.
///
/// # Returns
///
/// A clipped copy with the input dtype.
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex scalar bound.
/// * `ValueError` — complex array, non-scalar bound, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([-1, 0, 5, 10])
/// assert np.clip(a, 0, 8).to_list() == [0, 0, 5, 8]
/// ```
#[pyfunction]
pub fn clip(
    py: Python<'_>,
    a: PyRef<PyArray>,
    min: &Bound<'_, PyAny>,
    max: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("clip")?;
    let dt = a.inner.dtype();
    if dt == crate::dtype::PyDType::C64 {
        return Err(value_error("clip is not supported for complex arrays"));
    }
    let min_scalar = clip_bound(min)?;
    let max_scalar = clip_bound(max)?;
    // Coerce bound scalars to the array's element type per arm.
    let inner = match (&a.inner, min_scalar, max_scalar) {
        (ArrayInner::I64(a), min_s, max_s) => {
            let min_v = min_s.map(|s| match s {
                PyScalar::I64(v) => v,
                PyScalar::F64(v) => v as i64,
                PyScalar::Bool(v) => i64::from(v),
                _ => unreachable!(),
            });
            let max_v = max_s.map(|s| match s {
                PyScalar::I64(v) => v,
                PyScalar::F64(v) => v as i64,
                PyScalar::Bool(v) => i64::from(v),
                _ => unreachable!(),
            });
            ArrayInner::I64(map_sdnp(sdnp::clip(a, min_v, max_v))?)
        }
        (ArrayInner::F64(a), min_s, max_s) => {
            let min_v = min_s.map(|s| match s {
                PyScalar::F64(v) => v,
                PyScalar::I64(v) => v as f64,
                PyScalar::Bool(v) => {
                    if v {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            });
            let max_v = max_s.map(|s| match s {
                PyScalar::F64(v) => v,
                PyScalar::I64(v) => v as f64,
                PyScalar::Bool(v) => {
                    if v {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            });
            ArrayInner::F64(map_sdnp(sdnp::clip(a, min_v, max_v))?)
        }
        (ArrayInner::Bool(a), min_s, max_s) => {
            let min_v = min_s.map(|s| match s {
                PyScalar::Bool(v) => v,
                PyScalar::I64(v) => v != 0,
                PyScalar::F64(v) => v != 0.0,
                _ => unreachable!(),
            });
            let max_v = max_s.map(|s| match s {
                PyScalar::Bool(v) => v,
                PyScalar::I64(v) => v != 0,
                PyScalar::F64(v) => v != 0.0,
                _ => unreachable!(),
            });
            ArrayInner::Bool(map_sdnp(sdnp::clip(a, min_v, max_v))?)
        }
        _ => return Err(value_error("clip dtype mismatch")),
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Register selection callables on the extension module.
///
/// Adds `where`, `nonzero`, and `clip` to the `sdnp` module object.
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
/// assert callable(np.where)
/// assert callable(np.clip)
/// ```
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    Ok(())
}
