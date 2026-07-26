//! Coerce Python objects into [`ArrayInner`] and helper types.
//!
//! Bridges dynamic Python values (nested lists, scalars, existing `Array`s)
//! to statically typed storage. Nested-sequence parsing infers shape and
//! dtype; explicit `dtype=` overrides inference. 0-D arrays are never
//! created from Python literals — scalars stay internal until unwrap.

use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyComplex, PyFloat, PyInt, PyList,
    PySequence, PyString, PyTuple,
};
use sdnp::Array;
use sdnp::Complex64;

use crate::array::PyArray;
use crate::dtype::{
    scalar_to_bool, scalar_to_c64, scalar_to_f64, scalar_to_i64, PyDType,
};
use crate::error::{map_sdnp, type_error, value_error};
use crate::inner::ArrayInner;
use crate::unwrap::PyScalar;

/// Parse a creation `shape` from an int or sequence of non-negative ints.
///
/// Used by `array(..., shape=...)`, `zeros`, and other factory functions.
/// A bare int becomes a 1-D shape; tuples and lists become multi-D shapes.
/// Empty sequences and scalar-only shapes are rejected (no 0-D from Python).
///
/// # Arguments
///
/// * `obj` - Python int, tuple, list, or other finite sequence of ints.
///
/// # Returns
///
/// A validated `Vec<usize>` with at least one dimension.
///
/// # Errors
///
/// * `ValueError` — negative dimension, empty shape, 0-D request, or
///   product overflow (`usize`).
/// * `TypeError` — element is not an integer (via PyO3 extract).
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.zeros(3)          # shape parsed as (3,)
/// b = np.zeros((2, 4))     # shape parsed as (2, 4)
/// ```
pub fn parse_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(seq) = obj.downcast::<PySequence>() {
        let mut shape = Vec::with_capacity(seq.len()?);
        for item in seq.try_iter()? {
            let item = item?;
            let dim: isize = item.extract()?;
            if dim < 0 {
                return Err(value_error(
                    "shape dimensions must be non-negative",
                ));
            }
            shape.push(dim as usize);
        }
        if shape.is_empty() {
            return Err(value_error(
                "0-dimensional arrays cannot be created from Python",
            ));
        }
        validate_shape_size(&shape)?;
        return Ok(shape);
    }
    let dim: isize = obj.extract()?;
    if dim < 0 {
        return Err(value_error("shape dimensions must be non-negative"));
    }
    let shape = vec![dim as usize];
    validate_shape_size(&shape)?;
    Ok(shape)
}

/// Parse a reshape target shape, allowing `-1` for one inferred dimension.
///
/// Unlike [`parse_shape`], dimensions may be negative: exactly one `-1` is
/// resolved by the reshape kernel from the total element count. Integers are
/// signed because NumPy accepts `-1` in reshape tuples.
///
/// # Arguments
///
/// * `obj` - Python int or sequence of ints (each dimension or `-1`).
///
/// # Returns
///
/// Signed dimension list passed to the core reshape planner.
///
/// # Errors
///
/// * `ValueError` — element is not an integer.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([1, 2, 3, 4, 5, 6])
/// b = a.reshape(2, -1)     # -1 inferred as 3
/// assert b.shape == (2, 3)
/// ```
pub fn coerce_reshape_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(seq) = obj.downcast::<PySequence>() {
        let mut dimensions = Vec::with_capacity(seq.len()?);
        for item in seq.try_iter()? {
            dimensions.push(item?.extract::<isize>().map_err(|_| {
                value_error("reshape dimensions must be integers")
            })?);
        }
        Ok(dimensions)
    } else {
        Ok(vec![obj
            .extract::<isize>()
            .map_err(|_| value_error("reshape dimensions must be integers"))?])
    }
}

/// Guard against shape products overflowing `usize`.
///
/// Multiplies all dimensions with checked arithmetic before allocation.
/// Called from [`parse_shape`] so huge shape tuples fail early.
///
/// # Arguments
///
/// * `shape` - Proposed output dimensions (already non-negative).
///
/// # Returns
///
/// `Ok(())` when the product fits in `usize`.
///
/// # Errors
///
/// * `ValueError` — product exceeds `usize::MAX`.
fn validate_shape_size(shape: &[usize]) -> PyResult<()> {
    shape
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
        .ok_or_else(|| value_error("shape size overflows usize"))?;
    Ok(())
}

/// Coerce a Python object to a typed [`PyScalar`].
///
/// Checks `bool` before `int` because Python `bool` is a subclass of `int`.
/// Only bare literals are accepted — not `Array` instances (use
/// [`coerce_array_like`] for array-like input).
///
/// # Arguments
///
/// * `obj` - Python bool, int, float, or complex literal.
///
/// # Returns
///
/// A tagged scalar in the narrowest matching storage type.
///
/// # Errors
///
/// * `TypeError` — object is not a supported scalar literal.
/// * `ValueError` — int/complex conversion out of range (via dtype helpers).
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// # Scalars feed array() only with shape=; coercion happens internally.
/// a = np.array(3.5, shape=(2,))
/// assert a.dtype == np.float64
/// ```
pub fn coerce_scalar(obj: &Bound<'_, PyAny>) -> PyResult<PyScalar> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(PyScalar::Bool(scalar_to_bool(obj)?));
    }
    // bool is a subclass of int in Python — exclude it here.
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        return Ok(PyScalar::I64(scalar_to_i64(obj)?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(PyScalar::F64(scalar_to_f64(obj)?));
    }
    if obj.is_instance_of::<PyComplex>() {
        return Ok(PyScalar::C64(scalar_to_c64(obj)?));
    }
    Err(type_error(format!(
        "expected a scalar (bool, int, float, complex), got {obj}"
    )))
}

/// Coerce array-like Python input into a typed [`PyArray`].
///
/// Resolution order: existing `sdnp.Array` (optional cast), Python scalar
/// (becomes internal 0-D storage), or nested list/tuple (shape inference +
/// flatten). Nested sequences infer dtype from leaf values; explicit
/// `dtype=` overrides inference and may cast leaves.
///
/// # Arguments
///
/// * `obj` - `Array`, scalar literal, or nested list/tuple of scalars.
/// * `dtype` - Optional target dtype; when set, arrays are cast via
///   [`crate::dispatch::cast_inner`].
///
/// # Returns
///
/// A [`PyArray`] wrapping [`ArrayInner`]. Scalars become 0-D internally
/// (callers apply the unwrap policy before returning to Python).
///
/// # Errors
///
/// * `ValueError` — 0-D from bare scalar/list, inhomogeneous nesting,
///   invalid cast, or core allocation failure.
/// * `TypeError` — unsupported element type (strings, bytes, etc.).
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// a = np.array([[1, 2], [3, 4]])       # nested list → (2, 2) int
/// b = np.array([1.0, 2.0], dtype=np.float64)
/// c = np.add(a, 1)                     # uses coercion on operands
/// ```
pub fn coerce_array_like(
    obj: &Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyArray> {
    if let Ok(array) = obj.extract::<PyRef<PyArray>>() {
        array.reject_zero_dim_input("array input")?;
        let inner = array.inner.clone();
        return Ok(PyArray {
            inner: match dtype {
                Some(dtype) => crate::dispatch::cast_inner(inner, dtype)?,
                None => inner,
            },
        });
    }
    if is_python_scalar(obj) {
        let scalar = coerce_scalar(obj)?;
        let want = dtype.unwrap_or(scalar.dtype());
        return scalar_to_array(scalar, want);
    }
    let (shape, values, inferred) = flatten_nested(obj)?;
    if shape.is_empty() {
        return Err(value_error(
            "0-dimensional arrays cannot be created from Python",
        ));
    }
    let dt = dtype.unwrap_or(inferred);
    build_array_from_flat(&shape, &values, dt)
}

/// Return whether `obj` is a bare Python scalar literal.
///
/// Distinguishes literals from `sdnp.Array` and nested sequences. Used to
/// route ufunc operands and to reject bare scalars in `array()` without
/// `shape=`. `bool` is checked separately from `int` (subclass rule).
///
/// # Arguments
///
/// * `obj` - Any Python object.
///
/// # Returns
///
/// `true` for bool/int/float/complex literals; `false` otherwise.
///
/// # Errors
///
/// Never fails.
pub fn is_python_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyBool>()
        || (obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>())
        || obj.is_instance_of::<PyFloat>()
        || obj.is_instance_of::<PyComplex>()
}

/// Build internal 0-D storage from one scalar and a target dtype.
///
/// Maps each `(PyScalar, PyDType)` pair to the correct [`ArrayInner`] variant.
/// Some casts are rejected (e.g. int → bool) to match NumPy-like rules.
/// Used when a scalar is coerced as array-like input (ufuncs, `shape=` fill).
///
/// # Arguments
///
/// * `scalar` - Already-coerced Python scalar.
/// * `dtype` - Desired element type for the internal 0-D array.
///
/// # Returns
///
/// A [`PyArray`] whose inner storage holds exactly one element.
///
/// # Errors
///
/// * `ValueError` — unsupported scalar→dtype cast or allocation failure.
fn scalar_to_array(scalar: PyScalar, dtype: PyDType) -> PyResult<PyArray> {
    let inner = match (scalar, dtype) {
        (PyScalar::Bool(v), PyDType::Bool) => {
            ArrayInner::Bool(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
        (PyScalar::Bool(v), PyDType::I64) => {
            ArrayInner::I64(map_sdnp(Array::from_vec(vec![i64::from(v)], &[]))?)
        }
        (PyScalar::Bool(v), PyDType::F64) => ArrayInner::F64(map_sdnp(
            Array::from_vec(vec![if v { 1.0 } else { 0.0 }], &[]),
        )?),
        (PyScalar::Bool(v), PyDType::C64) => {
            ArrayInner::C64(map_sdnp(Array::from_vec(
                vec![Complex64::new(if v { 1.0 } else { 0.0 }, 0.0)],
                &[],
            ))?)
        }
        (PyScalar::I64(_v), PyDType::Bool) => {
            return Err(value_error("cannot cast int scalar to bool array"))
        }
        (PyScalar::I64(v), PyDType::I64) => {
            ArrayInner::I64(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
        (PyScalar::I64(v), PyDType::F64) => {
            ArrayInner::F64(map_sdnp(Array::from_vec(vec![v as f64], &[]))?)
        }
        (PyScalar::I64(v), PyDType::C64) => ArrayInner::C64(map_sdnp(
            Array::from_vec(vec![Complex64::new(v as f64, 0.0)], &[]),
        )?),
        (PyScalar::F64(_v), PyDType::Bool) => {
            return Err(value_error("cannot cast float scalar to bool array"))
        }
        (PyScalar::F64(v), PyDType::I64) => {
            ArrayInner::I64(map_sdnp(Array::from_vec(vec![v as i64], &[]))?)
        }
        (PyScalar::F64(v), PyDType::F64) => {
            ArrayInner::F64(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
        (PyScalar::F64(v), PyDType::C64) => ArrayInner::C64(map_sdnp(
            Array::from_vec(vec![Complex64::new(v, 0.0)], &[]),
        )?),
        (PyScalar::C64(_v), PyDType::Bool)
        | (PyScalar::C64(_v), PyDType::I64) => {
            return Err(value_error("cannot cast complex scalar to real array"))
        }
        (PyScalar::C64(v), PyDType::F64) => {
            ArrayInner::F64(map_sdnp(Array::from_vec(vec![v.re], &[]))?)
        }
        (PyScalar::C64(v), PyDType::C64) => {
            ArrayInner::C64(map_sdnp(Array::from_vec(vec![v], &[]))?)
        }
    };
    Ok(PyArray { inner })
}

/// Temporary flat value while parsing nested Python sequences.
enum FlatValue {
    Bool(bool),
    I64(i64),
    F64(f64),
    C64(Complex64),
}

/// Maximum nesting depth for list/tuple parsing (DoS guard).
const MAX_NESTING_DEPTH: usize = 64;

/// Infer shape, flatten nested leaves, and pick a common dtype.
///
/// Two-pass nested-sequence pipeline: [`infer_shape`] validates homogeneous
/// nesting, then [`flatten_into`] collects leaves in C order. Dtype is the
/// promotion max over all leaves (complex > float > int > bool).
///
/// # Arguments
///
/// * `obj` - Nested list, tuple, or generic sequence of scalars.
///
/// # Returns
///
/// Tuple of `(shape, flat_values, inferred_dtype)`.
///
/// # Errors
///
/// * `ValueError` — inhomogeneous nesting, depth limit, or 0-D literal.
/// * `TypeError` — non-sequence or unsupported leaf type.
fn flatten_nested(
    obj: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, Vec<FlatValue>, PyDType)> {
    let mut values = Vec::new();
    let shape = infer_shape(obj, 0)?;
    flatten_into(obj, &shape, 0, &mut values)?;
    let dtype = infer_dtype_from_values(&values)?;
    Ok((shape, values, dtype))
}

/// Recursively infer a homogeneous nested shape (reject strings/bytes).
///
/// Walks list/tuple structure depth-first. Every sibling sub-sequence must
/// share the same inferred trailing shape. Strings and bytes are sequences
/// in Python but are rejected here as array literals. Depth is capped at
/// [`MAX_NESTING_DEPTH`] to limit DoS from pathological nesting.
///
/// # Arguments
///
/// * `obj` - Current node in the nested structure.
/// * `depth` - Current recursion depth (0 at the root).
///
/// # Returns
///
/// Shape suffix for this node: `[]` at a scalar leaf, `[len, …]` at lists.
///
/// # Errors
///
/// * `ValueError` — inhomogeneous rows, max depth exceeded, or 0-D root.
/// * `TypeError` — str/bytes or non-sequence where nested list expected.
fn infer_shape(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<Vec<usize>> {
    if depth > MAX_NESTING_DEPTH {
        return Err(value_error(format!(
            "nested sequence exceeds maximum depth {MAX_NESTING_DEPTH}"
        )));
    }
    if is_python_scalar(obj) {
        if depth == 0 {
            return Err(value_error(
                "0-dimensional arrays cannot be created from Python",
            ));
        }
        return Ok(vec![]);
    }
    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
    {
        // Strings are sequences in Python but not array literals here.
        let message = if depth == 0 {
            format!("expected nested list/tuple, got {obj}")
        } else {
            format!("unsupported element type: {obj}")
        };
        return Err(type_error(message));
    }
    let seq = if let Ok(list) = obj.downcast::<PyList>() {
        list.as_sequence()
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        tuple.as_sequence()
    } else if let Ok(seq) = obj.downcast::<PySequence>() {
        seq
    } else {
        return Err(type_error(format!(
            "expected nested list/tuple, got {obj}"
        )));
    };
    let len = seq.len()?;
    if len == 0 {
        let mut shape = vec![0usize];
        shape.extend(std::iter::repeat_n(0, depth));
        return Ok(shape);
    }
    let sub = infer_shape(&seq.get_item(0)?, depth + 1)?;
    if !sub.is_empty() {
        let first_len = sub[0];
        for i in 1..len {
            let other = infer_shape(&seq.get_item(i)?, depth + 1)?;
            if other != sub {
                return Err(value_error("inhomogeneous nested sequence"));
            }
            if other[0] != first_len && !other.is_empty() {
                return Err(value_error("inhomogeneous nested sequence"));
            }
        }
    }
    let mut shape = vec![len];
    shape.extend(sub);
    Ok(shape)
}

/// Walk a nested structure and collect leaf values in C order.
///
/// Assumes `shape` was produced by [`infer_shape`] for the same `obj`.
/// Recurses one dimension at a time until `depth == shape.len()`, then
/// extracts a [`FlatValue`] at each leaf.
///
/// # Arguments
///
/// * `obj` - Current subtree root.
/// * `shape` - Full inferred shape (shared across the walk).
/// * `depth` - Index into `shape` (0 = outermost dimension).
/// * `out` - Accumulator for leaf values in row-major (C) order.
///
/// # Returns
///
/// `Ok(())` after appending all leaves to `out`.
///
/// # Errors
///
/// * `TypeError` — node is not a sequence at an interior level.
/// * `ValueError` — leaf is not a supported scalar (via
///   [`extract_flat_value`]).
fn flatten_into(
    obj: &Bound<'_, PyAny>,
    shape: &[usize],
    depth: usize,
    out: &mut Vec<FlatValue>,
) -> PyResult<()> {
    if depth == shape.len() {
        out.push(extract_flat_value(obj)?);
        return Ok(());
    }
    let seq = if let Ok(list) = obj.downcast::<PyList>() {
        list.as_sequence()
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        tuple.as_sequence()
    } else {
        obj.downcast::<PySequence>()?
    };
    for i in 0..seq.len()? {
        flatten_into(&seq.get_item(i)?, shape, depth + 1, out)?;
    }
    Ok(())
}

/// Extract one nested-sequence leaf as a temporary [`FlatValue`].
///
/// Applies the same bool-before-int rule as [`coerce_scalar`]. Leaves are
/// widened later by [`infer_dtype_from_values`] and [`build_array_from_flat`].
///
/// # Arguments
///
/// * `obj` - Scalar leaf at the bottom of a nested literal.
///
/// # Returns
///
/// A tagged leaf value before final dtype materialization.
///
/// # Errors
///
/// * `TypeError` — leaf is not bool/int/float/complex.
fn extract_flat_value(obj: &Bound<'_, PyAny>) -> PyResult<FlatValue> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(FlatValue::Bool(scalar_to_bool(obj)?));
    }
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        return Ok(FlatValue::I64(scalar_to_i64(obj)?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(FlatValue::F64(scalar_to_f64(obj)?));
    }
    if obj.is_instance_of::<PyComplex>() {
        return Ok(FlatValue::C64(scalar_to_c64(obj)?));
    }
    Err(type_error(format!("unsupported element type: {obj}")))
}

/// Pick the widest dtype among flattened nested leaves.
///
/// Promotion order: complex > float > int > bool. Any complex leaf forces
/// `complex128`; otherwise the max over int/float/bool applies. Empty
/// `values` yields `bool` (only reachable for empty-shaped arrays).
///
/// # Arguments
///
/// * `values` - Leaves collected by [`flatten_into`].
///
/// # Returns
///
/// Inferred [`PyDType`] unless overridden by an explicit `dtype=` argument.
///
/// # Errors
///
/// Never fails for valid [`FlatValue`] slices.
fn infer_dtype_from_values(values: &[FlatValue]) -> PyResult<PyDType> {
    let mut dt = PyDType::Bool;
    for v in values {
        dt = match v {
            FlatValue::Bool(_) => dt.max(PyDType::Bool),
            FlatValue::I64(_) => dt.max(PyDType::I64),
            FlatValue::F64(_) => dt.max(PyDType::F64),
            FlatValue::C64(_) => PyDType::C64,
        };
    }
    Ok(dt)
}

/// Materialize typed [`ArrayInner`] storage from flattened nested literals.
///
/// Converts each [`FlatValue`] to the target element type, then calls
/// `Array::from_vec`. Narrowing casts (float→int without explicit dtype)
/// are rejected to avoid silent data loss.
///
/// # Arguments
///
/// * `shape` - Inferred output dimensions.
/// * `values` - Flat leaf buffer in C order; length must match shape product.
/// * `dtype` - Target element type (inferred or explicit).
///
/// # Returns
///
/// A [`PyArray`] wrapping fully typed storage.
///
/// # Errors
///
/// * `ValueError` — incompatible leaf for target dtype or allocation error.
fn build_array_from_flat(
    shape: &[usize],
    values: &[FlatValue],
    dtype: PyDType,
) -> PyResult<PyArray> {
    match dtype {
        PyDType::Bool => {
            let data: Vec<bool> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => Ok(*b),
                    _ => {
                        Err(value_error("cannot convert mixed values to bool"))
                    }
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::Bool(map_sdnp(Array::from_vec(
                    data, shape,
                ))?),
            })
        }
        PyDType::I64 => {
            let data: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => Ok(i64::from(*b)),
                    FlatValue::I64(i) => Ok(*i),
                    _ => Err(value_error(
                        "cannot convert float/complex to int without dtype",
                    )),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::I64(map_sdnp(Array::from_vec(data, shape))?),
            })
        }
        PyDType::F64 => {
            let data: Vec<f64> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
                    FlatValue::I64(i) => Ok(*i as f64),
                    FlatValue::F64(f) => Ok(*f),
                    _ => Err(value_error(
                        "cannot convert complex to float without dtype",
                    )),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::F64(map_sdnp(Array::from_vec(data, shape))?),
            })
        }
        PyDType::C64 => {
            let data: Vec<Complex64> = values
                .iter()
                .map(|v| match v {
                    FlatValue::Bool(b) => {
                        Ok(Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0))
                    }
                    FlatValue::I64(i) => Ok(Complex64::new(*i as f64, 0.0)),
                    FlatValue::F64(f) => Ok(Complex64::new(*f, 0.0)),
                    FlatValue::C64(c) => Ok(*c),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyArray {
                inner: ArrayInner::C64(map_sdnp(Array::from_vec(data, shape))?),
            })
        }
    }
}

/// Parse a single axis index as a signed integer.
///
/// Axes follow NumPy convention: negative indices count from the end. Used by
/// reductions, sorting, and manipulation entry points.
///
/// # Arguments
///
/// * `obj` - Python int (or bool coerced by PyO3 — callers should validate).
///
/// # Returns
///
/// The axis value as `isize`.
///
/// # Errors
///
/// * `TypeError` — object is not an integer.
pub fn coerce_axis(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    obj.extract::<isize>()
        .map_err(|_| type_error("axis must be an integer"))
}

/// Parse one axis or a sequence of axes.
///
/// Accepts either a bare int (`axis=0`) or a tuple/list of ints
/// (`axis=(0, 1)`). Each element is validated via [`coerce_axis`].
///
/// # Arguments
///
/// * `obj` - Python int or sequence of ints.
///
/// # Returns
///
/// Normalized list of axis indices (length 1 for a scalar input).
///
/// # Errors
///
/// * `TypeError` — not an int and not a sequence, or bad sequence element.
pub fn coerce_axes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(axis) = obj.extract::<isize>() {
        return Ok(vec![axis]);
    }
    let seq = obj.downcast::<PySequence>()?;
    let mut axes = Vec::with_capacity(seq.len()?);
    for item in seq.try_iter()? {
        let item = item?;
        axes.push(coerce_axis(&item)?);
    }
    Ok(axes)
}

/// Parse an optional axis list; absent means reduce over all axes.
///
/// Treats Rust `None` and Python `None` the same. When present, delegates
/// to [`coerce_axes`]. Matches NumPy `axis=None` semantics at the boundary.
///
/// # Arguments
///
/// * `obj` - Optional Python object from an optional `axis=` parameter.
///
/// # Returns
///
/// `None` when axis is omitted; otherwise `Some(Vec<isize>)`.
///
/// # Errors
///
/// * `TypeError` — present value is not a valid axis or axis sequence.
pub fn coerce_optional_axes(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<isize>>> {
    match obj {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(None),
        Some(obj) => Ok(Some(coerce_axes(obj)?)),
    }
}

/// Parse an optional single axis index.
///
/// Treats Rust `None` and Python `None` as absent. When present, delegates
/// to [`coerce_axis`]. Used by APIs that accept at most one axis.
///
/// # Arguments
///
/// * `obj` - Optional Python object from an optional `axis=` parameter.
///
/// # Returns
///
/// `None` when axis is omitted; otherwise `Some(isize)`.
///
/// # Errors
///
/// * `TypeError` — present value is not an integer.
pub fn coerce_optional_axis(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<isize>> {
    match obj {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(None),
        Some(obj) => Ok(Some(coerce_axis(obj)?)),
    }
}

/// Extract a [`PyRef<PyArray>`] or raise a contextual `TypeError`.
///
/// Validates that the object is an `sdnp.Array` and rejects internal 0-D
/// arrays where Python callers expect ndim ≥ 1 (message includes `context`).
///
/// # Arguments
///
/// * `obj` - Candidate array argument.
/// * `context` - Parameter name for error messages (e.g. `"concatenate"`).
///
/// # Returns
///
/// Borrowed [`PyArray`] reference for the duration of the GIL borrow.
///
/// # Errors
///
/// * `TypeError` — object is not an `sdnp.Array`.
/// * `ValueError` — array is 0-D when disallowed for this API.
pub fn require_pyarray<'py>(
    obj: &Bound<'py, PyAny>,
    context: &str,
) -> PyResult<PyRef<'py, PyArray>> {
    let array = obj
        .extract::<PyRef<PyArray>>()
        .map_err(|_| type_error(format!("{context} must be an sdnp.Array")))?;
    array.reject_zero_dim_input(context)?;
    Ok(array)
}

/// Collect [`ArrayInner`] values from a Python sequence of arrays.
///
/// Used by `concatenate`, `stack`, and similar multi-array APIs. Clones
/// inner storage from each element after dtype/shape validation at the
/// Python boundary.
///
/// # Arguments
///
/// * `seq` - Python list/tuple of `sdnp.Array` instances.
/// * `context` - API name for error messages.
///
/// # Returns
///
/// Owned `Vec<ArrayInner>` in sequence order.
///
/// # Errors
///
/// * `TypeError` — `seq` is not a sequence or an element is not an Array.
/// * `ValueError` — empty sequence or 0-D element where disallowed.
pub fn collect_pyarrays(
    seq: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<ArrayInner>> {
    let sequence = seq.downcast::<PySequence>().map_err(|_| {
        type_error(format!("{context} argument must be a sequence of arrays"))
    })?;
    if sequence.len()? == 0 {
        return Err(value_error(format!(
            "{context} requires at least one array"
        )));
    }
    let mut out = Vec::with_capacity(sequence.len()?);
    for (index, item) in sequence.try_iter()?.enumerate() {
        let item = item?;
        let array = item.extract::<PyRef<PyArray>>().map_err(|_| {
            type_error(format!(
                "{context} element {index} must be an sdnp.Array"
            ))
        })?;
        array.reject_zero_dim_input(context)?;
        out.push(array.inner.clone());
    }
    Ok(out)
}
