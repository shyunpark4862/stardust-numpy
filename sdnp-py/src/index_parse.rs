//! Parse Python index objects into core `IndexSpec` lists.
//!
//! Translates `int`, `slice`, `Ellipsis`, `None` (newaxis), tuples, and fancy
//! integer/boolean arrays into the indexing IR consumed by `sdnp::gather` and
//! `sdnp::scatter`. Validation runs here before the core sees the spec.

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use sdnp::{gather, scatter, scatter_array, IndexSpec};

use crate::array::PyArray;
use crate::coerce::coerce_scalar;
use crate::dispatch::cast_inner;
use crate::error::{index_error, map_sdnp, type_error, value_error};
use crate::inner::ArrayInner;
use crate::unwrap::{finish, PyScalar};
use crate::validate::check_slice_step;

/// Implement `Array.__getitem__`: parse, validate, gather, unwrap.
///
/// Translates Python index objects into core [`IndexSpec`] lists, validates
/// bounds and mask shapes, then returns a scalar or new array.
///
/// # Arguments
///
/// * `py` - Python interpreter token.
/// * `array` - Source `PyArray`.
/// * `index` - NumPy-style index object.
///
/// # Returns
///
/// A Python scalar (0-D unwrap) or `PyArray` wrapper.
///
/// # Errors
///
/// * `TypeError` — invalid index type or 0-D fancy index array.
/// * `IndexError` — out-of-bounds, too many indices, or mask mismatch.
/// * `ValueError` — invalid slice step or core gather failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([10, 20, 30])
/// assert a[1] == 20
/// assert a[1:3].to_list() == [20, 30]
/// ```
pub fn get_item(
    py: Python<'_>,
    array: &PyArray,
    index: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let spec = parse_index(index)?;
    validate_index(array.inner.shape(), &spec)?;
    let inner = match &array.inner {
        ArrayInner::Bool(a) => ArrayInner::Bool(map_sdnp(gather(a, &spec))?),
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(gather(a, &spec))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(gather(a, &spec))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(gather(a, &spec))?),
    };
    finish(py, inner)
}

/// Implement `Array.__setitem__`: parse, validate, scatter scalar or array.
///
/// Accepts Python scalars with NumPy-like coercion or same-broadcast-shape
/// array values. Mutates the source array in place.
///
/// # Arguments
///
/// * `array` - Mutable target `PyArray`.
/// * `index` - NumPy-style index object.
/// * `value` - Scalar or `Array` to write.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * `TypeError` — invalid index type or 0-D index/value array.
/// * `IndexError` — out-of-bounds or boolean mask shape mismatch.
/// * `ValueError` — incompatible assignment dtype or scatter failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3])
/// a[0] = 99
/// assert a.to_list() == [99, 2, 3]
/// ```
pub fn set_item(
    array: &mut PyArray,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let spec = parse_index(index)?;
    validate_index(array.inner.shape(), &spec)?;
    if let Ok(arr) = value.extract::<PyRef<PyArray>>() {
        arr.reject_zero_dim_input("array assignment")?;
        return set_item_array(array, &spec, &arr.inner);
    }
    let scalar = coerce_scalar(value)?;
    set_item_scalar(array, &spec, scalar)
}

/// Scatter a Python scalar with permissive cross-dtype assignment rules.
///
/// Mirrors NumPy-like narrowing and widening when the stored dtype differs
/// from the Python scalar type (e.g. `float` into `int`, `bool` into `int`).
///
/// # Arguments
///
/// * `array` - Mutable target `PyArray`.
/// * `spec` - Parsed index specification.
/// * `scalar` - Coerced Python scalar value.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * `ValueError` — incompatible assignment to `bool` storage or scatter
///   failure from the core.
fn set_item_scalar(
    array: &mut PyArray,
    spec: &[IndexSpec],
    scalar: PyScalar,
) -> PyResult<()> {
    match (&mut array.inner, scalar) {
        (ArrayInner::Bool(a), PyScalar::Bool(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::I64(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::F64(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        (ArrayInner::C64(a), PyScalar::C64(v)) => {
            map_sdnp(scatter(a, spec, v)).map(|_| ())
        }
        // Narrowing/widening assignments mirror NumPy-like coercion.
        (ArrayInner::Bool(a), PyScalar::I64(v)) => {
            map_sdnp(scatter(a, spec, v != 0)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::Bool(v)) => {
            map_sdnp(scatter(a, spec, i64::from(v))).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::I64(v)) => {
            map_sdnp(scatter(a, spec, v as f64)).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::Bool(v)) => {
            map_sdnp(scatter(a, spec, if v { 1.0 } else { 0.0 })).map(|_| ())
        }
        (ArrayInner::C64(a), other) => {
            map_sdnp(scatter(a, spec, scalar_to_c64(other)?)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::F64(v)) => {
            map_sdnp(scatter(a, spec, v as i64)).map(|_| ())
        }
        (ArrayInner::F64(a), PyScalar::C64(v)) => {
            map_sdnp(scatter(a, spec, v.re)).map(|_| ())
        }
        (ArrayInner::I64(a), PyScalar::C64(v)) => {
            map_sdnp(scatter(a, spec, v.re as i64)).map(|_| ())
        }
        (ArrayInner::Bool(_), _) => {
            Err(value_error("cannot assign value to bool array"))
        }
    }
}

/// Promote a scalar to `Complex64` for complex-array assignment.
///
/// Real-valued scalars gain a zero imaginary part.
///
/// # Arguments
///
/// * `s` - Coerced Python scalar.
///
/// # Returns
///
/// A `Complex64` suitable for complex storage scatter.
///
/// # Errors
///
/// None; all scalar variants are representable as complex.
fn scalar_to_c64(s: PyScalar) -> PyResult<sdnp::Complex64> {
    use sdnp::Complex64;
    Ok(match s {
        PyScalar::Bool(v) => Complex64::new(if v { 1.0 } else { 0.0 }, 0.0),
        PyScalar::I64(v) => Complex64::new(v as f64, 0.0),
        PyScalar::F64(v) => Complex64::new(v, 0.0),
        PyScalar::C64(v) => v,
    })
}

/// Scatter an array value; promote source dtype when needed.
///
/// When dtypes differ, the source is cast to the destination dtype and the
/// scatter is retried recursively.
///
/// # Arguments
///
/// * `array` - Mutable target `PyArray`.
/// * `spec` - Parsed index specification.
/// * `values` - Source typed storage to write.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * `ValueError` — cast or scatter failure from the core.
fn set_item_array(
    array: &mut PyArray,
    spec: &[IndexSpec],
    values: &ArrayInner,
) -> PyResult<()> {
    match (&mut array.inner, values) {
        (ArrayInner::Bool(a), ArrayInner::Bool(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (ArrayInner::I64(a), ArrayInner::I64(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (ArrayInner::F64(a), ArrayInner::F64(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (ArrayInner::C64(a), ArrayInner::C64(v)) => {
            map_sdnp(scatter_array(a, spec, v)).map(|_| ())
        }
        (dst, src) => {
            let promoted = cast_inner(src.clone(), dst.dtype())?;
            set_item_array(array, spec, &promoted)
        }
    }
}

/// Parse a top-level index: bare item or tuple of items (no outer wrapper).
///
/// Tuple elements are concatenated into one flat `IndexSpec` list, matching
/// NumPy's `(i, j, ...)` indexing semantics.
///
/// # Arguments
///
/// * `obj` - Python index object (`int`, `slice`, tuple, etc.).
///
/// # Returns
///
/// A vector of core [`IndexSpec`] entries.
///
/// # Errors
///
/// * `TypeError` — unsupported index type or 0-D fancy index array.
/// * `ValueError` — invalid slice step.
fn parse_index(obj: &Bound<'_, PyAny>) -> PyResult<Vec<IndexSpec>> {
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut specs = Vec::new();
        for item in tuple.iter() {
            specs.extend(parse_index_item(&item)?);
        }
        return Ok(specs);
    }
    parse_index_item(obj)
}

/// Parse one index slot: int, slice, ellipsis, newaxis, or fancy array.
///
/// Fancy arrays become [`IndexSpec::IntegerArray`] or
/// [`IndexSpec::BoolArray`]; float/complex index arrays are rejected.
///
/// # Arguments
///
/// * `obj` - One index component.
///
/// # Returns
///
/// A one-element (or fancy) `IndexSpec` vector for this slot.
///
/// # Errors
///
/// * `TypeError` — invalid index type, 0-D index array, or float/complex
///   fancy index.
fn parse_index_item(obj: &Bound<'_, PyAny>) -> PyResult<Vec<IndexSpec>> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(vec![IndexSpec::NewAxis]);
    }
    if obj.is(&py.Ellipsis()) {
        return Ok(vec![IndexSpec::Ellipsis]);
    }
    if let Ok(slice) = obj.downcast::<PySlice>() {
        return Ok(vec![parse_slice(slice)?]);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(vec![IndexSpec::Index(i)]);
    }
    if let Ok(arr) = obj.extract::<PyRef<PyArray>>() {
        arr.reject_zero_dim_input("array index")?;
        return fancy_spec(&arr);
    }
    Err(type_error(format!(
        "index must be int, slice, ellipsis, None, or array; got {obj}"
    )))
}

/// Convert a Python slice object into `IndexSpec::Slice`.
///
/// `None` components remain `None` so the core applies NumPy default bounds.
///
/// # Arguments
///
/// * `slice` - Python `slice` object.
///
/// # Returns
///
/// An [`IndexSpec::Slice`] with optional start/stop/step.
///
/// # Errors
///
/// * `TypeError` — non-integer slice component.
/// * `ValueError` — zero slice step.
fn parse_slice(slice: &Bound<'_, PySlice>) -> PyResult<IndexSpec> {
    let start = optional_index(&slice.getattr("start")?)?;
    let stop = optional_index(&slice.getattr("stop")?)?;
    let step = optional_index(&slice.getattr("step")?)?;
    check_slice_step(step)?;
    Ok(IndexSpec::Slice { start, stop, step })
}

/// Extract `None` → `None`, otherwise an integer index component.
///
/// Used for slice `start`, `stop`, and `step` attributes.
///
/// # Arguments
///
/// * `obj` - Slice component attribute value.
///
/// # Returns
///
/// `None` when the attribute is `None`, else `Some(i64)`.
///
/// # Errors
///
/// * `TypeError` — non-integer, non-`None` component.
fn optional_index(obj: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if obj.is_none() {
        return Ok(None);
    }
    Ok(Some(obj.extract::<i64>()?))
}

/// Build a fancy-index spec from an integer or boolean index array.
///
/// Translates a `PyArray` used as an index into the core gather/scatter IR.
/// Only integer and boolean dtypes are permitted for fancy indexing.
///
/// # Arguments
///
/// * `arr` - Index array (`ndim >= 1`).
///
/// # Returns
///
/// A one-element vector containing [`IndexSpec::IntegerArray`] or
/// [`IndexSpec::BoolArray`].
///
/// # Errors
///
/// * `TypeError` — float or complex fancy index array.
fn fancy_spec(arr: &PyRef<PyArray>) -> PyResult<Vec<IndexSpec>> {
    match &arr.inner {
        ArrayInner::Bool(a) => Ok(vec![IndexSpec::BoolArray(a.clone())]),
        ArrayInner::I64(a) => Ok(vec![IndexSpec::IntegerArray(a.clone())]),
        ArrayInner::F64(_) | ArrayInner::C64(_) => Err(type_error(
            "fancy index must be an integer or boolean array",
        )),
    }
}

/// How many source axes one spec entry consumes (0 for newaxis/ellipsis).
///
/// Boolean masks consume `mask.ndim()` axes; scalar indices consume one.
///
/// # Arguments
///
/// * `spec` - One parsed index specification entry.
///
/// # Returns
///
/// Number of source axes consumed by this entry.
///
/// # Errors
///
/// None.
fn axes_consumed(spec: &IndexSpec) -> usize {
    match spec {
        IndexSpec::NewAxis | IndexSpec::Ellipsis => 0,
        IndexSpec::BoolArray(mask) => mask.ndim(),
        IndexSpec::Index(_)
        | IndexSpec::Slice { .. }
        | IndexSpec::IntegerArray(_) => 1,
    }
}

/// Check ellipsis count, axis count, and boolean mask shape alignment.
///
/// Ensures at most one ellipsis, not too many indices, and that boolean
/// masks exactly cover the indexed sub-shape.
///
/// # Arguments
///
/// * `shape` - Source array shape.
/// * `specs` - Parsed index specification list.
///
/// # Returns
///
/// `Ok(())` when the index is structurally valid.
///
/// # Errors
///
/// * `IndexError` — multiple ellipses, too many indices, or boolean mask
///   shape mismatch.
fn validate_index(shape: &[usize], specs: &[IndexSpec]) -> PyResult<()> {
    let ellipsis_count = specs
        .iter()
        .filter(|spec| matches!(spec, IndexSpec::Ellipsis))
        .count();
    if ellipsis_count > 1 {
        return Err(index_error("an index can only have a single ellipsis"));
    }

    let used: usize = specs.iter().map(axes_consumed).sum();
    if used > shape.len() {
        return Err(index_error(format!(
            "too many indices for array: array is {}-dimensional, but {used} were indexed",
            shape.len()
        )));
    }
    // Ellipsis expands to consume all remaining unindexed axes.
    let missing = shape.len() - used;
    let mut source_axis = 0usize;
    for spec in specs {
        match spec {
            IndexSpec::NewAxis => {}
            IndexSpec::Ellipsis => source_axis += missing,
            IndexSpec::BoolArray(mask) => {
                let end = source_axis + mask.ndim();
                if end > shape.len() || mask.shape() != &shape[source_axis..end]
                {
                    return Err(index_error(format!(
                        "boolean index shape {:?} does not match indexed dimensions {:?}",
                        mask.shape(),
                        shape.get(source_axis..end).unwrap_or(&[])
                    )));
                }
                source_axis = end;
            }
            IndexSpec::Index(_)
            | IndexSpec::Slice { .. }
            | IndexSpec::IntegerArray(_) => {
                source_axis += 1;
            }
        }
    }
    Ok(())
}
