//! Reduction free functions.

use pyo3::prelude::*;
use sdnp::NanPolicy;

use crate::array::{array_from_inner, wrap_result};
use crate::coerce::{coerce_optional_axes, coerce_optional_axis};
use crate::error::{map_sdnp, value_error};
use crate::inner::ArrayInner;
use crate::validate::{
    check_arg_nonempty, check_axis_xor_axes, check_nonempty_reduction,
    check_optional_axes, check_optional_axis,
};

fn parse_nan_policy(s: &str) -> PyResult<NanPolicy> {
    match s {
        "propagate" => Ok(NanPolicy::Propagate),
        "ignore" => Ok(NanPolicy::Ignore),
        other => Err(value_error(format!(
            "nan_policy must be 'propagate' or 'ignore', got '{other}'"
        ))),
    }
}

fn check_arg_nan_slice(
    name: &str,
    inner: &ArrayInner,
    axis: Option<isize>,
    policy: NanPolicy,
) -> PyResult<()> {
    if policy != NanPolicy::Ignore {
        return Ok(());
    }
    let ArrayInner::F64(array) = inner else {
        return Ok(());
    };
    let values = array.to_vec();
    if axis.is_none() {
        if !values.is_empty() && values.iter().all(|value| value.is_nan()) {
            return Err(value_error(format!("{name} of all-NaN slice")));
        }
        return Ok(());
    }

    let axis = crate::validate::normalize_axis(axis.unwrap(), array.ndim())?;
    let outer = array.shape()[..axis].iter().product::<usize>();
    let axis_len = array.shape()[axis];
    let inner_len = array.shape()[axis + 1..].iter().product::<usize>();
    for outer_index in 0..outer {
        for inner_index in 0..inner_len {
            let all_nan = (0..axis_len).all(|axis_index| {
                values[(outer_index * axis_len + axis_index) * inner_len
                    + inner_index]
                    .is_nan()
            });
            if all_nan {
                return Err(value_error(format!("{name} of all-NaN slice")));
            }
        }
    }
    Ok(())
}

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
            _ => unreachable!(),
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
            _ => unreachable!(),
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
            _ => unreachable!(),
        },
    })
}

macro_rules! reduce_fn {
    ($name:ident) => {
        #[pyfunction]
        #[pyo3(signature = (a, *, axis=None, axes=None, keepdims=false, nan_policy="propagate"))]
        pub fn $name(
            py: Python<'_>,
            a: PyRef<crate::array::PyArray>,
            axis: Option<&Bound<'_, PyAny>>,
            axes: Option<&Bound<'_, PyAny>>,
            keepdims: bool,
            nan_policy: &str,
        ) -> PyResult<PyObject> {
            let policy = parse_nan_policy(nan_policy)?;
            check_axis_xor_axes(axis.is_some(), axes.is_some())?;
            let ndim = a.inner.ndim();
            let ax = if axes.is_some() {
                let ax = coerce_optional_axes(axes)?;
                check_optional_axes(ax.as_deref(), ndim)?;
                ax
            } else {
                let ax = coerce_optional_axes(axis)?;
                check_optional_axes(ax.as_deref(), ndim)?;
                ax
            };
            if matches!(stringify!($name), "min" | "max" | "mean" | "var") {
                check_nonempty_reduction(stringify!($name), &a.inner, ax.as_deref())?;
            }
            wrap_result(
                py,
                reduce_inner(&a.inner, stringify!($name), ax.as_deref(), keepdims, policy)?,
            )
        }
    };
}

reduce_fn!(sum);
reduce_fn!(prod);
reduce_fn!(min);
reduce_fn!(max);
reduce_fn!(mean);
reduce_fn!(var);

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
    let policy = parse_nan_policy(nan_policy)?;
    check_axis_xor_axes(axis.is_some(), axes.is_some())?;
    let ndim = a.inner.ndim();
    let ax = if axes.is_some() {
        let ax = coerce_optional_axes(axes)?;
        check_optional_axes(ax.as_deref(), ndim)?;
        ax
    } else {
        let ax = coerce_optional_axes(axis)?;
        check_optional_axes(ax.as_deref(), ndim)?;
        ax
    };
    check_nonempty_reduction("std", &a.inner, ax.as_deref())?;
    wrap_result(
        py,
        reduce_inner(&a.inner, "std", ax.as_deref(), keepdims, policy)?,
    )
}

#[pyfunction]
#[pyo3(signature = (a, *, axis=None, keepdims=false))]
pub fn any(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<PyObject> {
    let ax = coerce_optional_axes(axis)?;
    check_optional_axes(ax.as_deref(), a.inner.ndim())?;
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

#[pyfunction]
#[pyo3(signature = (a, *, axis=None, keepdims=false))]
pub fn all(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<PyObject> {
    let ax = coerce_optional_axes(axis)?;
    check_optional_axes(ax.as_deref(), a.inner.ndim())?;
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

#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn argmin(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    check_optional_axis(ax, a.inner.ndim())?;
    check_arg_nonempty("argmin", &a.inner, ax)?;
    check_arg_nan_slice("argmin", &a.inner, ax, policy)?;
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

#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn argmax(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    check_optional_axis(ax, a.inner.ndim())?;
    check_arg_nonempty("argmax", &a.inner, ax)?;
    check_arg_nan_slice("argmax", &a.inner, ax, policy)?;
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

#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn cumsum(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    check_optional_axis(ax, a.inner.ndim())?;
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

#[pyfunction]
#[pyo3(signature = (a, *, axis=None, nan_policy="propagate"))]
pub fn cumprod(
    py: Python<'_>,
    a: PyRef<crate::array::PyArray>,
    axis: Option<&Bound<'_, PyAny>>,
    nan_policy: &str,
) -> PyResult<PyObject> {
    let policy = parse_nan_policy(nan_policy)?;
    let ax = coerce_optional_axis(axis)?;
    check_optional_axis(ax, a.inner.ndim())?;
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
