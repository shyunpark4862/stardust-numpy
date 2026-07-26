//! Reduction and cumulative free functions.
//!
//! Parses `axis`/`axes`, `keepdims`, and `nan_policy` at the Python boundary,
//! applies Python-only argument policy, and maps typed `sdnp` reduction
//! results. Dtype-specific result types
//! (e.g. bool `sum` → int64) are chosen in the dispatch table below.

use pyo3::prelude::*;
use sdnp::NanPolicy;

use crate::array::{array_from_inner, wrap_result};
use crate::coerce::{coerce_optional_axes, coerce_optional_axis};
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::check_axis_xor_axes;

/// Parse `nan_policy` string into the core enum.
///
/// Accepts only `"propagate"` and `"ignore"`.
///
/// # Arguments
///
/// * `s` - Python keyword value for `nan_policy`.
///
/// # Returns
///
/// The corresponding [`NanPolicy`] variant.
///
/// # Errors
///
/// * `ValueError` — string is not a recognized policy name.
fn parse_nan_policy(s: &str) -> PyResult<NanPolicy> {
    match s {
        "propagate" => Ok(NanPolicy::Propagate),
        "ignore" => Ok(NanPolicy::Ignore),
        other => Err(value_error(format!(
            "nan_policy must be 'propagate' or 'ignore', got '{other}'"
        ))),
    }
}

/// Preserve the Python API rule that an explicit axes sequence is non-empty.
fn reject_empty_axes(axes: Option<&[isize]>) -> PyResult<()> {
    if axes.is_some_and(<[isize]>::is_empty) {
        return Err(value_error(
            "axes must be a non-empty sequence of integers",
        ));
    }
    Ok(())
}

/// Typed reduction dispatch keyed by storage variant and op name.
///
/// Selects the monomorphized `sdnp` kernel and result dtype (e.g. bool
/// `sum` → int64, integer `mean` → float64).
///
/// # Arguments
///
/// * `inner` - Typed input storage.
/// * `name` - Reduction op tag (`"sum"`, `"mean"`, etc.).
/// * `axes` - Optional axis list after Python coercion.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `policy` - NaN policy for floating reductions.
///
/// # Returns
///
/// Result storage wrapped in [`ArrayInner`].
///
/// # Errors
///
/// * `ValueError` — unsupported op/dtype pair or core failure.
fn reduce_inner(
    inner: &ArrayInner,
    name: &str,
    axes: Option<&[isize]>,
    keepdims: bool,
    policy: NanPolicy,
) -> PyResult<ArrayInner> {
    Ok(match inner {
        ArrayInner::Bool(arr) => match name {
            "sum" => ArrayInner::I64(map_sdnp(sdnp::sum(
                arr, axes, keepdims, policy,
            ))?),
            "prod" => ArrayInner::I64(map_sdnp(sdnp::prod(
                arr, axes, keepdims, policy,
            ))?),
            "min" => ArrayInner::Bool(map_sdnp(sdnp::min(
                arr, axes, keepdims, policy,
            ))?),
            "max" => ArrayInner::Bool(map_sdnp(sdnp::max(
                arr, axes, keepdims, policy,
            ))?),
            "mean" => ArrayInner::F64(map_sdnp(sdnp::mean(
                arr, axes, keepdims, policy,
            ))?),
            "var" => ArrayInner::F64(map_sdnp(sdnp::var(
                arr, axes, keepdims, policy,
            ))?),
            "std" => ArrayInner::F64(map_sdnp(sdnp::std(
                arr, axes, keepdims, policy,
            ))?),
            _ => {
                return Err(value_error(format!(
                    "unsupported reduction {name} for bool"
                )))
            }
        },
        ArrayInner::I64(arr) => match name {
            "sum" => ArrayInner::I64(map_sdnp(sdnp::sum(
                arr, axes, keepdims, policy,
            ))?),
            "prod" => ArrayInner::I64(map_sdnp(sdnp::prod(
                arr, axes, keepdims, policy,
            ))?),
            "min" => ArrayInner::I64(map_sdnp(sdnp::min(
                arr, axes, keepdims, policy,
            ))?),
            "max" => ArrayInner::I64(map_sdnp(sdnp::max(
                arr, axes, keepdims, policy,
            ))?),
            "mean" => ArrayInner::F64(map_sdnp(sdnp::mean(
                arr, axes, keepdims, policy,
            ))?),
            "var" => ArrayInner::F64(map_sdnp(sdnp::var(
                arr, axes, keepdims, policy,
            ))?),
            "std" => ArrayInner::F64(map_sdnp(sdnp::std(
                arr, axes, keepdims, policy,
            ))?),
            _ => {
                return Err(value_error(format!(
                    "unsupported reduction {name} for int"
                )))
            }
        },
        ArrayInner::F64(arr) => match name {
            "sum" => ArrayInner::F64(map_sdnp(sdnp::sum(
                arr, axes, keepdims, policy,
            ))?),
            "prod" => ArrayInner::F64(map_sdnp(sdnp::prod(
                arr, axes, keepdims, policy,
            ))?),
            "min" => ArrayInner::F64(map_sdnp(sdnp::min(
                arr, axes, keepdims, policy,
            ))?),
            "max" => ArrayInner::F64(map_sdnp(sdnp::max(
                arr, axes, keepdims, policy,
            ))?),
            "mean" => ArrayInner::F64(map_sdnp(sdnp::mean(
                arr, axes, keepdims, policy,
            ))?),
            "var" => ArrayInner::F64(map_sdnp(sdnp::var(
                arr, axes, keepdims, policy,
            ))?),
            "std" => ArrayInner::F64(map_sdnp(sdnp::std(
                arr, axes, keepdims, policy,
            ))?),
            _ => {
                return Err(value_error(format!(
                    "unsupported reduction {name} for float"
                )))
            }
        },
        ArrayInner::C64(arr) => match name {
            "sum" => ArrayInner::C64(map_sdnp(sdnp::sum(
                arr, axes, keepdims, policy,
            ))?),
            "prod" => ArrayInner::C64(map_sdnp(sdnp::prod(
                arr, axes, keepdims, policy,
            ))?),
            "mean" => ArrayInner::C64(map_sdnp(sdnp::mean(
                arr, axes, keepdims, policy,
            ))?),
            "var" | "std" => {
                return Err(value_error(
                    "var/std are not supported for complex arrays",
                ))
            }
            "min" | "max" => {
                return Err(value_error(
                    "min/max are not supported for complex arrays",
                ))
            }
            _ => {
                return Err(value_error(format!(
                    "unsupported reduction {name} for complex"
                )))
            }
        },
    })
}

/// Shared axis-reduction body for `sum`, `prod`, `min`, `max`, `mean`, `var`.
///
/// Parses axes, enforces Python-only keyword policy, dispatches through
/// [`reduce_inner`], and applies 0-D unwrap on the result.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `a` - Input array reference.
/// * `name` - Reduction op tag passed to [`reduce_inner`].
/// * `axis_args` - The mutually exclusive `axis` and `axes` keywords.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - `"propagate"` or `"ignore"`.
/// # Returns
///
/// Python object for the reduced array or bare scalar.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — axis conflict, bad policy, empty slice, or core failure.
fn reduce_axis_fn(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    name: &str,
    axis_args: (Option<&Bound<'_, PyAny>>, Option<&Bound<'_, PyAny>>),
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    let (axis, axes) = axis_args;
    a.reject_zero_dim_input(name)?;
    let policy = parse_nan_policy(nan_policy)?;
    check_axis_xor_axes(axis.is_some(), axes.is_some())?;
    let ax = if axes.is_some() {
        coerce_optional_axes(axes)?
    } else {
        coerce_optional_axes(axis)?
    };
    reject_empty_axes(ax.as_deref())?;
    wrap_result(
        py,
        reduce_inner(&a.inner, name, ax.as_deref(), keepdims, policy)?,
    )
}

/// Sum of array elements over the given axes.
///
/// Boolean input accumulates into int64. Floating reductions honor
/// `nan_policy`: "propagate" poisons on NaN; "ignore" skips NaN.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple (mutually exclusive with `axes`).
/// * `axes` - Explicit axis list (mutually exclusive with `axis`).
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - "propagate" or "ignore" for floating dtypes.
///
/// # Returns
///
/// Sum array; bool input yields int64.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axis/axes, bad `nan_policy`, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3])
/// assert np.sum(a) == 6
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn sum(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "sum", (axis, axes), keepdims, nan_policy)
}

/// Product of array elements over the given axes.
///
/// Boolean input accumulates into int64. Floating reductions honor
/// `nan_policy`.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes.
/// * `axes` - Explicit axis list.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - "propagate" or "ignore".
///
/// # Returns
///
/// Product array; bool input yields int64.
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axis/axes or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([2, 3, 4])
/// assert np.prod(a) == 24
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn prod(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "prod", (axis, axes), keepdims, nan_policy)
}

/// Minimum value over the given axes.
///
/// For float64, `nan_policy` selects propagate vs ignore behavior.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes.
/// * `axes` - Explicit axis list.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - "propagate" or "ignore".
///
/// # Returns
///
/// Minimum array with the input dtype.
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex dtype.
/// * `ValueError` — empty slice, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([3, 1, 2])
/// assert np.min(a) == 1
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn min(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "min", (axis, axes), keepdims, nan_policy)
}

/// Maximum value over the given axes.
///
/// For float64, `nan_policy` selects propagate vs ignore behavior.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes.
/// * `axes` - Explicit axis list.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - "propagate" or "ignore".
///
/// # Returns
///
/// Maximum array with the input dtype.
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex dtype.
/// * `ValueError` — empty slice, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([3, 1, 2])
/// assert np.max(a) == 3
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn max(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "max", (axis, axes), keepdims, nan_policy)
}

/// Arithmetic mean over the given axes.
///
/// Floating reductions honor `nan_policy`. Integer and bool promote to
/// float64 for the result.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes.
/// * `axes` - Explicit axis list.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - "propagate" or "ignore".
///
/// # Returns
///
/// Mean array (float64 for integer/bool input).
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — empty slice, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([2, 4, 6])
/// assert np.mean(a) == 4.0
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn mean(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "mean", (axis, axes), keepdims, nan_policy)
}

/// Population variance (`ddof=0`) over the given axes.
///
/// Always returns float64. Not supported for complex arrays.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes.
/// * `axes` - Explicit axis list.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - "propagate" or "ignore".
///
/// # Returns
///
/// Variance array as float64.
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex dtype.
/// * `ValueError` — empty slice, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1.0, 2.0, 3.0])
/// assert abs(np.var(a) - 1.0) < 1e-10
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn var(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "var", (axis, axes), keepdims, nan_policy)
}

/// Population standard deviation over the given axes.
///
/// Square root of [`var`]. Always returns float64. Not supported for
/// complex arrays.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes.
/// * `axes` - Explicit axis list.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
/// * `nan_policy` - `'propagate'` or `'ignore'`.
///
/// # Returns
///
/// Standard deviation array as float64.
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex dtype.
/// * `ValueError` — empty slice, invalid axis, or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1.0, 2.0, 3.0])
/// assert abs(np.std(a) - 1.0) < 1e-10
/// ```
#[pyfunction]
#[pyo3(name = "std", signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
pub fn py_std(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    axes: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    nan_policy: &str,
) -> PyResult<PyObject> {
    reduce_axis_fn(py, a, "std", (axis, axes), keepdims, nan_policy)
}

/// True if any element is logically true over the given axes.
///
/// Non-boolean dtypes are interpreted via their truthiness.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes, or `None` for all axes.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
///
/// # Returns
///
/// Boolean array (or Python `bool` when the result is 0-D).
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axis or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([0, 0, 1])
/// assert np.any(a) is True
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, keepdims=false))]
pub fn any(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("any")?;
    let ax = coerce_optional_axes(axis)?;
    reject_empty_axes(ax.as_deref())?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::any(arr, ax.as_deref(), keepdims))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::any(arr, ax.as_deref(), keepdims))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::any(arr, ax.as_deref(), keepdims))?)
        }
        ArrayInner::C64(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::any(arr, ax.as_deref(), keepdims))?)
        }
    };
    wrap_result(py, inner)
}

/// True if all elements are logically true over the given axes.
///
/// Non-boolean dtypes are interpreted via their truthiness.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Single axis or tuple of axes, or `None` for all axes.
/// * `keepdims` - Keep reduced axes as length-1 dimensions.
///
/// # Returns
///
/// Boolean array (or Python `bool` when the result is 0-D).
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axis or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 1, 1])
/// assert np.all(a) is True
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, keepdims=false))]
pub fn all(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("all")?;
    let ax = coerce_optional_axes(axis)?;
    reject_empty_axes(ax.as_deref())?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::all(arr, ax.as_deref(), keepdims))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::all(arr, ax.as_deref(), keepdims))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::all(arr, ax.as_deref(), keepdims))?)
        }
        ArrayInner::C64(arr) => {
            ArrayInner::Bool(map_sdnp(sdnp::all(arr, ax.as_deref(), keepdims))?)
        }
    };
    wrap_result(py, inner)
}

/// Index of the minimum value along an axis or over the whole array.
///
/// Returns int64 indices. For float64 with `nan_policy='ignore'`, an
/// all-NaN slice raises `ValueError`. For bool input the Rust core stops at
/// the first `false`, so cost depends on the winning position rather than the
/// full axis length.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to find minima, or `None` for flat index.
/// * `nan_policy` - `'propagate'` or `'ignore'`.
///
/// # Returns
///
/// int64 index array (or Python int when 0-D).
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex dtype.
/// * `ValueError` — empty slice, all-NaN slice, or invalid axis.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([3, 1, 2])
/// assert np.argmin(a) == 1
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn argmin(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("argmin")?;
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argmin(arr, ax, policy))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argmin(arr, ax, policy))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argmin(arr, ax, policy))?)
        }
        ArrayInner::C64(_) => {
            return Err(value_error(
                "argmin is not supported for complex arrays",
            ))
        }
    };
    wrap_result(py, inner)
}

/// Index of the maximum value along an axis or over the whole array.
///
/// Returns int64 indices. For float64 with `nan_policy='ignore'`, an
/// all-NaN slice raises `ValueError`. For bool input the Rust core stops at
/// the first `true`, so cost depends on the winning position rather than the
/// full axis length.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to find maxima, or `None` for flat index.
/// * `nan_policy` - `'propagate'` or `'ignore'`.
///
/// # Returns
///
/// int64 index array (or Python int when 0-D).
///
/// # Errors
///
/// * `TypeError` — 0-D input or complex dtype.
/// * `ValueError` — empty slice, all-NaN slice, or invalid axis.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([3, 1, 2])
/// assert np.argmax(a) == 0
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn argmax(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("argmax")?;
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argmax(arr, ax, policy))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argmax(arr, ax, policy))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::argmax(arr, ax, policy))?)
        }
        ArrayInner::C64(_) => {
            return Err(value_error(
                "argmax is not supported for complex arrays",
            ))
        }
    };
    wrap_result(py, inner)
}

/// Cumulative sum along an axis or in flat C order.
///
/// Boolean input promotes to int64. Output shape matches the input.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to cumulate, or `None` for flat order.
/// * `nan_policy` - `'propagate'` or `'ignore'` for floating dtypes.
///
/// # Returns
///
/// Cumulative sum array (int64 for bool input).
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axis or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3])
/// assert np.cumsum(a).to_list() == [1, 3, 6]
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn cumsum(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("cumsum")?;
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::cumsum(arr, ax, policy))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::cumsum(arr, ax, policy))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::F64(map_sdnp(sdnp::cumsum(arr, ax, policy))?)
        }
        ArrayInner::C64(arr) => {
            ArrayInner::C64(map_sdnp(sdnp::cumsum(arr, ax, policy))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Cumulative product along an axis or in flat C order.
///
/// Boolean input promotes to int64. Output shape matches the input.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to cumulate, or `None` for flat order.
/// * `nan_policy` - `'propagate'` or `'ignore'` for floating dtypes.
///
/// # Returns
///
/// Cumulative product array (int64 for bool input).
///
/// # Errors
///
/// * `TypeError` — 0-D input.
/// * `ValueError` — invalid axis or core failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3])
/// assert np.cumprod(a).to_list() == [1, 2, 6]
/// ```
#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn cumprod(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    a.reject_zero_dim_input("cumprod")?;
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    let inner = match &a.inner {
        ArrayInner::Bool(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::cumprod(arr, ax, policy))?)
        }
        ArrayInner::I64(arr) => {
            ArrayInner::I64(map_sdnp(sdnp::cumprod(arr, ax, policy))?)
        }
        ArrayInner::F64(arr) => {
            ArrayInner::F64(map_sdnp(sdnp::cumprod(arr, ax, policy))?)
        }
        ArrayInner::C64(arr) => {
            ArrayInner::C64(map_sdnp(sdnp::cumprod(arr, ax, policy))?)
        }
    };
    crate::array::into_pyobject(py, array_from_inner(inner))
}

/// Register reduction callables on the extension module.
///
/// Adds axis reductions, boolean reductions, arg reductions, and cumulative
/// ops to the `sdnp` module object.
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
/// assert callable(np.sum)
/// assert callable(np.cumsum)
/// ```
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(py_std, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(cumprod, m)?)?;
    Ok(())
}
