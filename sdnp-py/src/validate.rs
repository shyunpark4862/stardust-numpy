//! User-input validation at the Python boundary.
//!
//! Domain invariants (index OOB, broadcast failure, read-only writes, …) remain
//! in the Rust core and surface as Python exceptions via `map_sdnp`.

use pyo3::prelude::*;
use pyo3::types::PySequence;

use crate::array::PyArray;
use crate::dtype::PyDType;
use crate::error::{index_error, type_error, value_error};
use crate::inner::ArrayInner;

/// Normalize one axis; returns the canonical index in `0..ndim`.
pub fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    if ndim == 0 {
        return Err(index_error(format!(
            "axis {axis} is out of bounds for array of dimension 0"
        )));
    }
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(index_error(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
}

/// Validate every axis is in range and unique.
pub fn check_axes(axes: &[isize], ndim: usize) -> PyResult<()> {
    if axes.is_empty() {
        return Err(value_error(
            "axes must be a non-empty sequence of integers",
        ));
    }
    let mut seen = vec![false; ndim.max(1)];
    for &axis in axes {
        let normalized = normalize_axis(axis, ndim)?;
        if ndim > 0 && seen[normalized] {
            return Err(value_error("axes must not contain duplicates"));
        }
        if ndim > 0 {
            seen[normalized] = true;
        }
    }
    Ok(())
}

/// Validate optional axis list before calling reduction kernels.
pub fn check_optional_axes(
    axes: Option<&[isize]>,
    ndim: usize,
) -> PyResult<()> {
    if let Some(axes) = axes {
        check_axes(axes, ndim)?;
    }
    Ok(())
}

/// Validate a single optional axis (cumsum, argmin, sort, …).
pub fn check_optional_axis(axis: Option<isize>, ndim: usize) -> PyResult<()> {
    if let Some(axis) = axis {
        normalize_axis(axis, ndim)?;
    }
    Ok(())
}

pub fn check_nonempty_reduction(
    name: &str,
    inner: &ArrayInner,
    axes: Option<&[isize]>,
) -> PyResult<()> {
    let reduced = match axes {
        None => (0..inner.ndim()).collect::<Vec<_>>(),
        Some(axes) => axes
            .iter()
            .map(|&axis| normalize_axis(axis, inner.ndim()))
            .collect::<PyResult<Vec<_>>>()?,
    };
    let output_len = inner
        .shape()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !reduced.contains(axis))
        .map(|(_, &dim)| dim)
        .product::<usize>();
    if output_len > 0 && reduced.iter().any(|&axis| inner.shape()[axis] == 0) {
        return Err(value_error(format!("{name} of empty array / empty axis")));
    }
    Ok(())
}

pub fn check_arg_nonempty(
    name: &str,
    inner: &ArrayInner,
    axis: Option<isize>,
) -> PyResult<()> {
    match axis {
        None if inner.size() == 0 => {
            Err(value_error(format!("{name} of empty array")))
        }
        Some(axis) => check_nonempty_reduction(name, inner, Some(&[axis])),
        _ => Ok(()),
    }
}

pub fn check_axis_xor_axes(axis: bool, axes: bool) -> PyResult<()> {
    if axis && axes {
        return Err(value_error("cannot specify both axis and axes"));
    }
    Ok(())
}

pub fn require_pyarray<'py>(
    obj: &Bound<'py, PyAny>,
    context: &str,
) -> PyResult<PyRef<'py, PyArray>> {
    obj.extract()
        .map_err(|_| type_error(format!("{context} must be an sdnp.Array")))
}

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
    for (i, item) in sequence.try_iter()?.enumerate() {
        let item = item?;
        let arr = item.extract::<PyRef<PyArray>>().map_err(|_| {
            type_error(format!("{context} element {i} must be an sdnp.Array"))
        })?;
        out.push(arr.inner.clone());
    }
    Ok(out)
}

pub fn check_same_dtype(arrays: &[ArrayInner], op: &str) -> PyResult<PyDType> {
    let dt = arrays[0].dtype();
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        if arr.dtype() != dt {
            return Err(value_error(format!(
                "all arrays must have the same dtype in {op}; array 0 has dtype {}, \
                 array {i} has dtype {}",
                dt.name(),
                arr.dtype().name()
            )));
        }
    }
    Ok(dt)
}

pub fn check_concatenate(
    arrays: &[ArrayInner],
    axis: isize,
) -> PyResult<isize> {
    if arrays.is_empty() {
        return Err(value_error("concatenate requires at least one array"));
    }
    let first = &arrays[0];
    if first.ndim() == 0 {
        return Err(value_error("cannot concatenate 0-D arrays"));
    }
    let axis = normalize_axis(axis, first.ndim())?;
    let ndim = first.ndim();
    let first_shape = first.shape();
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        if arr.ndim() != ndim {
            return Err(value_error(format!(
                "all arrays must have the same rank; array 0 has rank {ndim}, \
                 array {i} has rank {}",
                arr.ndim()
            )));
        }
        for dim in 0..ndim {
            if dim != axis && arr.shape()[dim] != first_shape[dim] {
                return Err(value_error(format!(
                    "array dimensions must match except along axis {axis}; array 0 has \
                     shape {first_shape:?}, array {i} has shape {:?}",
                    arr.shape()
                )));
            }
        }
    }
    Ok(axis as isize)
}

pub fn check_stack(arrays: &[ArrayInner], axis: isize) -> PyResult<()> {
    if arrays.is_empty() {
        return Err(value_error("stack requires at least one array"));
    }
    check_same_dtype(arrays, "stack")?;
    let reference = arrays[0].shape();
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        if arr.shape() != reference {
            return Err(value_error(format!(
                "all arrays must have the same shape; array 0 has shape {reference:?}, \
                 array {i} has shape {:?}",
                arr.shape()
            )));
        }
    }
    normalize_axis(axis, reference.len().saturating_add(1))?;
    Ok(())
}

pub fn check_vstack(arrays: &[ArrayInner]) -> PyResult<()> {
    check_promoted_join(arrays, true)
}

pub fn check_hstack(arrays: &[ArrayInner]) -> PyResult<()> {
    check_promoted_join(arrays, false)
}

fn check_promoted_join(arrays: &[ArrayInner], vertical: bool) -> PyResult<()> {
    if arrays.is_empty() {
        return Err(value_error("stacking requires at least one array"));
    }
    let promoted = arrays
        .iter()
        .map(|array| {
            if vertical {
                match array.ndim() {
                    0 => vec![1, 1],
                    1 => vec![1, array.shape()[0]],
                    _ => array.shape().to_vec(),
                }
            } else if array.ndim() == 0 {
                vec![1]
            } else {
                array.shape().to_vec()
            }
        })
        .collect::<Vec<_>>();
    let axis = if vertical || promoted[0].len() == 1 {
        0
    } else {
        1
    };
    for (index, shape) in promoted.iter().enumerate().skip(1) {
        if shape.len() != promoted[0].len()
            || shape.iter().zip(&promoted[0]).enumerate().any(
                |(dimension, (left, right))| dimension != axis && left != right,
            )
        {
            return Err(value_error(format!(
                "array dimensions must match for stacking; array 0 has shape {:?}, \
                 array {index} has shape {shape:?}",
                promoted[0]
            )));
        }
    }
    Ok(())
}

pub fn check_arange_step(step: i64) -> PyResult<()> {
    if step == 0 {
        return Err(value_error("arange step must not be zero"));
    }
    Ok(())
}

pub fn check_finite_bounds(name: &str, start: f64, stop: f64) -> PyResult<()> {
    if !start.is_finite() || !stop.is_finite() {
        return Err(value_error(format!("{name} bounds must be finite")));
    }
    Ok(())
}

pub fn check_logspace_base(base: f64) -> PyResult<()> {
    if !base.is_finite() || base <= 0.0 {
        return Err(value_error(
            "logspace base must be finite and greater than zero",
        ));
    }
    Ok(())
}

pub fn check_geomspace_bounds(start: f64, stop: f64) -> PyResult<()> {
    check_finite_bounds("geomspace", start, stop)?;
    if start == 0.0 || stop == 0.0 {
        return Err(value_error("geomspace bounds must not be zero"));
    }
    if start.is_sign_negative() != stop.is_sign_negative() {
        return Err(value_error("geomspace bounds must have the same sign"));
    }
    Ok(())
}

pub fn check_slice_step(step: Option<i64>) -> PyResult<()> {
    if step == Some(0) {
        return Err(value_error("slice step cannot be zero"));
    }
    Ok(())
}

/// Parse reshape dimensions, allowing exactly one `-1` to infer size.
pub fn parse_reshape_shape(
    obj: &Bound<'_, PyAny>,
    size: usize,
) -> PyResult<Vec<isize>> {
    let dims = if let Ok(seq) = obj.downcast::<PySequence>() {
        let mut out = Vec::with_capacity(seq.len()?);
        for item in seq.try_iter()? {
            out.push(parse_one_dim(&item?)?);
        }
        out
    } else {
        vec![parse_one_dim(obj)?]
    };

    if dims.is_empty() {
        return Err(value_error("reshape target shape must be non-empty"));
    }

    let mut inferred = None;
    let mut known = 1_usize;
    let mut resolved: Vec<isize> = Vec::with_capacity(dims.len());

    for (i, d) in dims.iter().copied().enumerate() {
        if d == -1 {
            if inferred.is_some() {
                return Err(value_error(
                    "only one reshape dimension may be -1",
                ));
            }
            inferred = Some(i);
            resolved.push(-1);
        } else if d < 0 {
            return Err(value_error(format!("invalid reshape dimension {d}")));
        } else {
            let dim = d as usize;
            known = known.checked_mul(dim).ok_or_else(|| {
                value_error("reshape shape size overflows usize")
            })?;
            resolved.push(d);
        }
    }

    if let Some(idx) = inferred {
        if known == 0 {
            return Err(value_error(
                "cannot infer reshape dimension when another is 0",
            ));
        }
        if !size.is_multiple_of(known) {
            return Err(value_error(format!(
                "cannot reshape array of size {size} into shape {dims:?}"
            )));
        }
        resolved[idx] = (size / known) as isize;
    } else {
        let total: usize = resolved.iter().map(|&d| d as usize).product();
        if total != size {
            return Err(value_error(format!(
                "cannot reshape array of size {size} into shape {dims:?}"
            )));
        }
    }

    Ok(resolved)
}

fn parse_one_dim(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    obj.extract::<isize>()
        .map_err(|_| value_error("reshape dimensions must be integers"))
}

pub fn check_permute_axes(axes: &[isize], ndim: usize) -> PyResult<()> {
    if axes.len() != ndim {
        return Err(value_error(format!(
            "axes length {} does not match ndim {ndim}",
            axes.len()
        )));
    }
    let mut seen = vec![false; ndim];
    for &axis in axes {
        let normalized = normalize_axis(axis, ndim)?;
        if seen[normalized] {
            return Err(value_error("axes must be a permutation of 0..ndim-1"));
        }
        seen[normalized] = true;
    }
    if !seen.iter().all(|&v| v) {
        return Err(value_error("axes must be a permutation of 0..ndim-1"));
    }
    Ok(())
}

pub fn check_squeeze_axes(
    inner: &ArrayInner,
    axes: Option<&[isize]>,
) -> PyResult<()> {
    if let Some(axes) = axes {
        check_axes(axes, inner.ndim())?;
        for &axis in axes {
            let normalized = normalize_axis(axis, inner.ndim())?;
            let length = inner.shape()[normalized];
            if length != 1 {
                return Err(value_error(format!(
                    "cannot squeeze axis {axis} with length {length}"
                )));
            }
        }
    }
    Ok(())
}

pub fn require_bool_array(inner: &ArrayInner, name: &str) -> PyResult<()> {
    if inner.dtype() != PyDType::Bool {
        return Err(type_error(format!("{name} must be a bool array")));
    }
    Ok(())
}

pub fn check_meshgrid_indexing(indexing: &str) -> PyResult<()> {
    match indexing {
        "xy" | "ij" => Ok(()),
        other => Err(value_error(format!(
            "meshgrid indexing must be 'xy' or 'ij', got '{other}'"
        ))),
    }
}

pub fn check_nditer_operands(n: usize) -> PyResult<()> {
    if n == 0 || n > 2 {
        return Err(value_error(
            "nditer supports 1-2 operands with the same dtype",
        ));
    }
    Ok(())
}

pub fn check_nditer_same_dtype(arrays: &[PyRef<PyArray>]) -> PyResult<()> {
    let dt = arrays[0].inner.dtype();
    for arr in &arrays[1..] {
        if arr.inner.dtype() != dt {
            return Err(value_error(
                "nditer requires operands with the same dtype",
            ));
        }
    }
    Ok(())
}

pub fn check_broadcastable(
    operation: &str,
    shapes: &[&[usize]],
) -> PyResult<()> {
    let ndim = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    for offset in 0..ndim {
        let mut expected = 1usize;
        for shape in shapes {
            let dimension = shape
                .len()
                .checked_sub(offset + 1)
                .map(|axis| shape[axis])
                .unwrap_or(1);
            if dimension != 1 {
                if expected != 1 && expected != dimension {
                    return Err(value_error(format!(
                        "{operation} operands could not be broadcast together: {:?}",
                        shapes
                    )));
                }
                expected = dimension;
            }
        }
    }
    Ok(())
}

pub fn check_matmul(left: &ArrayInner, right: &ArrayInner) -> PyResult<()> {
    if left.ndim() == 0 || right.ndim() == 0 {
        return Err(value_error("matmul does not support 0-D operands"));
    }
    let left_k = left.shape()[left.ndim() - 1];
    let right_k = if right.ndim() == 1 {
        right.shape()[0]
    } else {
        right.shape()[right.ndim() - 2]
    };
    if left_k != right_k {
        return Err(value_error(format!(
            "matmul inner dimensions differ: {left_k} != {right_k}"
        )));
    }

    let left_batch = &left.shape()[..left.ndim().saturating_sub(2)];
    let right_batch = &right.shape()[..right.ndim().saturating_sub(2)];
    let rank = left_batch.len().max(right_batch.len());
    for offset in 0..rank {
        let l = left_batch
            .len()
            .checked_sub(offset + 1)
            .map(|i| left_batch[i])
            .unwrap_or(1);
        let r = right_batch
            .len()
            .checked_sub(offset + 1)
            .map(|i| right_batch[i])
            .unwrap_or(1);
        if l != r && l != 1 && r != 1 {
            return Err(value_error(format!(
                "matmul batch dimensions are not broadcast-compatible: {:?} and {:?}",
                left_batch, right_batch
            )));
        }
    }
    Ok(())
}

pub fn check_dot(left: &ArrayInner, right: &ArrayInner) -> PyResult<()> {
    if !(1..=2).contains(&left.ndim()) || !(1..=2).contains(&right.ndim()) {
        return Err(value_error("dot supports only 1-D or 2-D operands"));
    }
    check_matmul(left, right)
}

pub fn check_vdot(left: &ArrayInner, right: &ArrayInner) -> PyResult<()> {
    if left.size() != right.size() {
        return Err(value_error(format!(
            "vdot requires equal flattened sizes: {} != {}",
            left.size(),
            right.size()
        )));
    }
    Ok(())
}

pub fn check_diagonal_axes(
    ndim: usize,
    axis1: isize,
    axis2: isize,
) -> PyResult<()> {
    if ndim < 2 {
        return Err(value_error(
            "diagonal and trace require an array of at least two dimensions",
        ));
    }
    let axis1 = normalize_axis(axis1, ndim)?;
    let axis2 = normalize_axis(axis2, ndim)?;
    if axis1 == axis2 {
        return Err(value_error("axis1 and axis2 must be different"));
    }
    Ok(())
}

pub fn check_triangle_input(name: &str, inner: &ArrayInner) -> PyResult<()> {
    if inner.ndim() == 0 {
        return Err(value_error(format!(
            "{name} requires an array of at least one dimension"
        )));
    }
    Ok(())
}

pub fn check_diag_input(inner: &ArrayInner) -> PyResult<()> {
    if !matches!(inner.ndim(), 1 | 2) {
        return Err(value_error("diag requires a 1-D or 2-D array"));
    }
    Ok(())
}

pub fn check_meshgrid_arrays(arrays: &[ArrayInner]) -> PyResult<()> {
    if arrays.iter().any(|array| array.ndim() != 1) {
        return Err(value_error("meshgrid inputs must be 1-D arrays"));
    }
    check_same_dtype(arrays, "meshgrid").map(|_| ())
}
