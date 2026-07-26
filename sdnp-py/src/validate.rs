//! User-input validation at the Python boundary.
//!
//! Checks argument shapes, dtypes, and API constraints before calling the
//! generic Rust core. Domain invariants (OOB index, broadcast failure,
//! read-only writes) remain in `sdnp` and surface via [`crate::error::map_sdnp`].
//! Negative axes are validated here but canonicalized in the core.

use pyo3::prelude::*;

use crate::array::PyArray;
use crate::dtype::PyDType;
use crate::error::{index_error, type_error, value_error};
use crate::inner::ArrayInner;

/// Validate that a single axis index is in bounds for an array of `ndim`.
///
/// Reduction and manipulation entry points accept negative axes the same way
/// NumPy does: `-1` refers to the last dimension. This helper rejects out-of-
/// bounds values at the Python boundary so the core never sees invalid axis
/// indices. Canonicalization to non-negative indices happens later in `sdnp`.
///
/// # Arguments
///
/// * `axis` — Python `axis` keyword, possibly negative.
/// * `ndim` — Rank of the target [`ArrayInner`].
///
/// # Returns
///
/// `Ok(())` when `axis` refers to a dimension in `0..ndim` (inclusive of
/// negative indexing rules).
///
/// # Errors
///
/// * [`PyIndexError`] — axis is out of bounds for the given rank.
///
/// # Examples
///
/// ```rust
/// use sdnp_py::validate::check_axis;
///
/// assert!(check_axis(0, 2).is_ok());
/// assert!(check_axis(-1, 2).is_ok());
/// assert!(check_axis(2, 2).is_err());
/// ```
pub fn check_axis(axis: isize, ndim: usize) -> PyResult<()> {
    let in_bounds = if axis < 0 {
        axis.unsigned_abs() <= ndim
    } else {
        (axis as usize) < ndim
    };
    if !in_bounds {
        return Err(index_error(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(())
}

/// Test whether a validated axis refers to a specific dimension index.
///
/// After [`check_axis`] succeeds, this predicate resolves negative axes to
/// the canonical dimension they denote. It is used to detect duplicate axes
/// in multi-axis APIs without mutating the original axis values passed to the
/// core.
///
/// # Arguments
///
/// * `axis` — Already validated axis index (may be negative).
/// * `candidate` — Zero-based dimension index to test.
/// * `ndim` — Rank of the array under validation.
///
/// # Returns
///
/// `true` when `axis` and `candidate` refer to the same dimension.
///
/// # Examples
///
/// ```rust
/// use sdnp_py::validate::axis_refers_to;
///
/// assert!(axis_refers_to(-1, 1, 2));
/// assert!(!axis_refers_to(0, 1, 2));
/// ```
pub fn axis_refers_to(axis: isize, candidate: usize, ndim: usize) -> bool {
    if axis < 0 {
        ndim.checked_sub(axis.unsigned_abs()) == Some(candidate)
    } else {
        axis as usize == candidate
    }
}

/// Validate a non-empty axis list: each entry in bounds and pairwise distinct.
///
/// Multi-axis reductions and `permute_axes`-style APIs accept a sequence of
/// integer axes from Python. This helper enforces NumPy-like rules: the list
/// must not be empty, every axis must be valid, and no two entries may refer
/// to the same dimension (including via negative indexing).
///
/// # Arguments
///
/// * `axes` — Slice of axis indices from a Python sequence.
/// * `ndim` — Rank of the target array.
///
/// # Returns
///
/// `Ok(())` when every axis is in bounds and the list has no duplicates.
///
/// # Errors
///
/// * [`PyValueError`] — empty axis list or duplicate axes.
/// * [`PyIndexError`] — any axis is out of bounds (via [`check_axis`]).
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// a = np.zeros((2, 3, 4))
/// a.sum(axis=(0, 2))   # ok
/// a.sum(axis=(0, 0))   # ValueError: axes must not contain duplicates
/// ```
pub fn check_axes(axes: &[isize], ndim: usize) -> PyResult<()> {
    if axes.is_empty() {
        return Err(value_error(
            "axes must be a non-empty sequence of integers",
        ));
    }
    for (index, &axis) in axes.iter().enumerate() {
        check_axis(axis, ndim)?;
        if axes[..index].iter().any(|&previous| {
            (0..ndim).any(|dimension| {
                axis_refers_to(axis, dimension, ndim)
                    && axis_refers_to(previous, dimension, ndim)
            })
        }) {
            return Err(value_error("axes must not contain duplicates"));
        }
    }
    Ok(())
}

/// Validate an optional multi-axis argument for reductions.
///
/// Python reduction methods accept `axes=None` (reduce over all dimensions)
/// or a non-empty sequence. When present, the sequence is checked with
/// [`check_axes`].
///
/// # Arguments
///
/// * `axes` — Optional slice from the Python `axes` keyword.
/// * `ndim` — Rank of the input array.
///
/// # Returns
///
/// `Ok(())` when `axes` is `None` or passes [`check_axes`].
///
/// # Errors
///
/// * [`PyValueError`] — invalid or duplicate axes.
/// * [`PyIndexError`] — any axis out of bounds.
pub fn check_optional_axes(
    axes: Option<&[isize]>,
    ndim: usize,
) -> PyResult<()> {
    if let Some(axes) = axes {
        check_axes(axes, ndim)?;
    }
    Ok(())
}

/// Validate an optional single-axis argument.
///
/// Many Python methods take `axis=None` for a global reduction or an integer
/// for a single dimension. When `axis` is provided, [`check_axis`] runs.
///
/// # Arguments
///
/// * `axis` — Optional axis from the Python `axis` keyword.
/// * `ndim` — Rank of the input array.
///
/// # Returns
///
/// `Ok(())` when `axis` is `None` or in bounds.
///
/// # Errors
///
/// * [`PyIndexError`] — axis out of bounds.
pub fn check_optional_axis(axis: Option<isize>, ndim: usize) -> PyResult<()> {
    if let Some(axis) = axis {
        check_axis(axis, ndim)?;
    }
    Ok(())
}

/// Reject reductions that would combine a non-empty output with a zero-length
/// reduced axis.
///
/// NumPy raises when reducing over an axis of length zero while other output
/// dimensions remain (e.g. `np.sum(np.zeros((0, 3)), axis=0)`). This check
/// mirrors that behavior before the reduction kernel runs.
///
/// # Arguments
///
/// * `name` — Operation name for the error message (e.g. `"sum"`).
/// * `inner` — Input [`ArrayInner`] being reduced.
/// * `axes` — Optional reduced axes; `None` means reduce all dimensions.
///
/// # Returns
///
/// `Ok(())` when the reduction is well-defined for the given shape.
///
/// # Errors
///
/// * [`PyValueError`] — non-empty output shape with at least one empty reduced
///   axis.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// np.sum(np.zeros((0, 3)), axis=0)  # ValueError
/// ```
pub fn check_nonempty_reduction(
    name: &str,
    inner: &ArrayInner,
    axes: Option<&[isize]>,
) -> PyResult<()> {
    let is_reduced = |dimension| {
        axes.is_none_or(|axes| {
            axes.iter()
                .any(|&axis| axis_refers_to(axis, dimension, inner.ndim()))
        })
    };
    let output_len = inner
        .shape()
        .iter()
        .enumerate()
        .filter(|(dimension, _)| !is_reduced(*dimension))
        .map(|(_, &length)| length)
        .product::<usize>();
    let reduction_is_empty = inner
        .shape()
        .iter()
        .enumerate()
        .any(|(dimension, &length)| is_reduced(dimension) && length == 0);
    if output_len > 0 && reduction_is_empty {
        return Err(value_error(format!("{name} of empty array / empty axis")));
    }
    Ok(())
}

/// Reject `argmin` / `argmax` on empty inputs or empty reduced slices.
///
/// Global arg reductions require a non-zero element count. Axis reductions
/// delegate to [`check_nonempty_reduction`] so empty slices along the chosen
/// axis are caught with the same message shape as other reductions.
///
/// # Arguments
///
/// * `name` — `"argmin"` or `"argmax"` for error messages.
/// * `inner` — Input array.
/// * `axis` — Optional single axis; `None` reduces the entire array.
///
/// # Returns
///
/// `Ok(())` when the input has elements along the reduction path.
///
/// # Errors
///
/// * [`PyValueError`] — empty global input or empty axis slice.
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

/// Reject passing both `axis` and `axes` to one reduction call.
///
/// NumPy allows only one multi-axis specification style per call. Bindings
/// normalize this at the boundary so downstream code never has to resolve
/// conflicting keyword arguments.
///
/// # Arguments
///
/// * `axis` — Whether the Python `axis` keyword was supplied.
/// * `axes` — Whether the Python `axes` keyword was supplied.
///
/// # Returns
///
/// `Ok(())` when at most one of the flags is true.
///
/// # Errors
///
/// * [`PyValueError`] — both `axis` and `axes` were provided.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// a = np.arange(6).reshape(2, 3)
/// a.sum(axis=0, axes=(1,))  # ValueError
/// ```
pub fn check_axis_xor_axes(axis: bool, axes: bool) -> PyResult<()> {
    if axis && axes {
        return Err(value_error("cannot specify both axis and axes"));
    }
    Ok(())
}

/// Require identical dtypes across a sequence of arrays.
///
/// Variadic Python APIs (`concatenate`, `stack`, `meshgrid`, …) expect
/// homogeneous element types unless explicit casting is documented. This
/// helper compares [`PyDType`] tags and reports the first mismatch.
///
/// # Arguments
///
/// * `arrays` — Operands already coerced to [`ArrayInner`].
/// * `op` — Operation name for the error message.
///
/// # Returns
///
/// `Ok(())` when every array shares the dtype of `arrays[0]`.
///
/// # Errors
///
/// * [`PyValueError`] — dtype mismatch between operands.
pub fn check_same_dtype(arrays: &[ArrayInner], op: &str) -> PyResult<()> {
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
    Ok(())
}

/// Validate `concatenate` operands: shared rank and matching trailing shape.
///
/// Along every dimension except the join axis, all inputs must agree in
/// length. Rank and axis bounds are checked before the core allocates the
/// output buffer.
///
/// # Arguments
///
/// * `arrays` — Sequence of arrays to join (non-empty).
/// * `axis` — Dimension along which to concatenate (may be negative).
///
/// # Returns
///
/// `Ok(())` when operands satisfy NumPy concatenate rules.
///
/// # Errors
///
/// * [`PyValueError`] — empty list, 0-D input, rank mismatch, or shape
///   mismatch off the join axis.
/// * [`PyIndexError`] — axis out of bounds.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// np.concatenate([np.zeros(3), np.ones(3)])       # ok (1-D)
/// np.concatenate([np.zeros((2, 3)), np.ones(4)])  # ValueError
/// ```
pub fn check_concatenate(arrays: &[ArrayInner], axis: isize) -> PyResult<()> {
    if arrays.is_empty() {
        return Err(value_error("concatenate requires at least one array"));
    }
    let first = &arrays[0];
    if first.ndim() == 0 {
        return Err(value_error("cannot concatenate 0-D arrays"));
    }
    check_axis(axis, first.ndim())?;
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
            if !axis_refers_to(axis, dim, ndim)
                && arr.shape()[dim] != first_shape[dim]
            {
                return Err(value_error(format!(
                    "array dimensions must match except along axis {axis}; array 0 has \
                     shape {first_shape:?}, array {i} has shape {:?}",
                    arr.shape()
                )));
            }
        }
    }
    Ok(())
}

/// Validate `stack` operands: identical shapes and valid axis for rank+1.
///
/// `stack` inserts a new dimension; operands must match exactly and the axis
/// must be valid for the promoted rank (`ndim + 1`). Dtype homogeneity is
/// enforced via [`check_same_dtype`].
///
/// # Arguments
///
/// * `arrays` — Sequence of same-shaped arrays (non-empty).
/// * `axis` — Index of the new dimension in the result (may be negative).
///
/// # Returns
///
/// `Ok(())` when stack preconditions hold.
///
/// # Errors
///
/// * [`PyValueError`] — empty list, shape mismatch, or dtype mismatch.
/// * [`PyIndexError`] — axis out of bounds for `ndim + 1`.
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
    check_axis(axis, reference.len().saturating_add(1))?;
    Ok(())
}

/// Validate `vstack` operands under NumPy-like rank promotion rules.
///
/// Vertical stacking treats 1-D inputs as row vectors by promoting rank to
/// at least two. Shape compatibility is checked on promoted dimensions via
/// [`check_promoted_join`].
///
/// # Arguments
///
/// * `arrays` — Sequence of arrays to stack vertically (non-empty).
///
/// # Returns
///
/// `Ok(())` when promoted shapes align for `vstack`.
///
/// # Errors
///
/// * [`PyValueError`] — empty list or incompatible promoted shapes.
pub fn check_vstack(arrays: &[ArrayInner]) -> PyResult<()> {
    check_promoted_join(arrays, true)
}

/// Validate `hstack` operands under NumPy-like rank promotion rules.
///
/// Horizontal stacking promotes 0-D scalars to length-1 vectors and joins
/// along axis 1 when rank is at least two (otherwise axis 0). See
/// [`check_promoted_join`] for the shared logic.
///
/// # Arguments
///
/// * `arrays` — Sequence of arrays to stack horizontally (non-empty).
///
/// # Returns
///
/// `Ok(())` when promoted shapes align for `hstack`.
///
/// # Errors
///
/// * [`PyValueError`] — empty list or incompatible promoted shapes.
pub fn check_hstack(arrays: &[ArrayInner]) -> PyResult<()> {
    check_promoted_join(arrays, false)
}

/// Shared shape check for [`check_vstack`] and [`check_hstack`].
///
/// NumPy promotes low-rank operands before comparing trailing dimensions.
/// This helper computes the effective join axis and verifies that every
/// non-join dimension matches across operands after promotion.
///
/// # Arguments
///
/// * `arrays` — Non-empty operand list.
/// * `vertical` — `true` for `vstack` (join axis 0), `false` for `hstack`.
///
/// # Returns
///
/// `Ok(())` when all promoted shapes are compatible.
///
/// # Errors
///
/// * [`PyValueError`] — empty list or shape mismatch after promotion.
fn check_promoted_join(arrays: &[ArrayInner], vertical: bool) -> PyResult<()> {
    if arrays.is_empty() {
        return Err(value_error("stacking requires at least one array"));
    }
    let reference_rank = promoted_rank(&arrays[0], vertical);
    // vstack joins on axis 0; hstack on axis 1 when rank ≥ 2.
    let axis = if vertical || reference_rank == 1 {
        0
    } else {
        1
    };
    for (index, array) in arrays.iter().enumerate().skip(1) {
        let rank = promoted_rank(array, vertical);
        let incompatible = rank != reference_rank
            || (0..reference_rank).any(|dimension| {
                dimension != axis
                    && promoted_length(array, vertical, dimension)
                        != promoted_length(&arrays[0], vertical, dimension)
            });
        if incompatible {
            return Err(value_error(format!(
                "array dimensions must match for stacking; array 0 has shape {:?}, \
                 array {index} has shape {:?}",
                arrays[0].shape(),
                array.shape()
            )));
        }
    }
    Ok(())
}

/// Effective rank after NumPy-like vstack/hstack promotion.
///
/// `vstack` treats vectors as `(1, n)` by promoting to at least rank 2;
/// `hstack` promotes 0-D arrays to rank 1. Used only during stacking
/// validation, not when building the output array.
///
/// # Arguments
///
/// * `array` — Operand whose logical rank is needed.
/// * `vertical` — `true` for vstack promotion, `false` for hstack.
///
/// # Returns
///
/// Promoted rank used for pairwise shape comparison.
fn promoted_rank(array: &ArrayInner, vertical: bool) -> usize {
    if vertical {
        array.ndim().max(2)
    } else {
        array.ndim().max(1)
    }
}

/// Effective axis length after rank promotion for stacking checks.
///
/// Low-rank operands gain synthetic length-1 dimensions during vstack/hstack
/// validation. This maps a promoted dimension index to the length that
/// [`check_promoted_join`] compares across operands.
///
/// # Arguments
///
/// * `array` — Operand array.
/// * `vertical` — `true` for vstack rules, `false` for hstack rules.
/// * `dimension` — Index in the promoted shape space.
///
/// # Returns
///
/// Length along `dimension` after NumPy-like promotion.
fn promoted_length(
    array: &ArrayInner,
    vertical: bool,
    dimension: usize,
) -> usize {
    if vertical {
        match array.ndim() {
            0 => 1,
            1 => {
                if dimension == 0 {
                    1
                } else {
                    array.shape()[0]
                }
            }
            _ => array.shape()[dimension],
        }
    } else if array.ndim() == 0 {
        1
    } else {
        array.shape()[dimension]
    }
}

/// Reject a zero step in three-argument `arange`.
///
/// Python's `range`-like constructors forbid `step=0` because the iteration
/// would not terminate. This is checked before any core range builder runs.
///
/// # Arguments
///
/// * `step` — Integer step from the Python `arange(start, stop, step)` call.
///
/// # Returns
///
/// `Ok(())` when `step != 0`.
///
/// # Errors
///
/// * [`PyValueError`] — step is zero.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// np.arange(0, 10, 0)  # ValueError
/// ```
pub fn check_arange_step(step: i64) -> PyResult<()> {
    if step == 0 {
        return Err(value_error("arange step must not be zero"));
    }
    Ok(())
}

/// Require finite floating-point endpoints for spaced sequence constructors.
///
/// `linspace`, `logspace`, and related APIs divide or interpolate between
/// bounds; non-finite values would propagate NaN through the entire output.
/// Shared by multiple creation entry points.
///
/// # Arguments
///
/// * `name` — API name for the error message (e.g. `"linspace"`).
/// * `start` — First bound from Python.
/// * `stop` — Second bound from Python.
///
/// # Returns
///
/// `Ok(())` when both bounds are finite.
///
/// # Errors
///
/// * [`PyValueError`] — either bound is NaN or infinity.
pub fn check_finite_bounds(name: &str, start: f64, stop: f64) -> PyResult<()> {
    if !start.is_finite() || !stop.is_finite() {
        return Err(value_error(format!("{name} bounds must be finite")));
    }
    Ok(())
}

/// Require a positive finite base for `logspace`.
///
/// Logarithmic spacing computes `base ** t` for `t` in a linear interval.
/// Non-positive or non-finite bases would yield undefined or complex values
/// outside the real `float64` dtype this crate exposes.
///
/// # Arguments
///
/// * `base` — Logarithm base from the Python `logspace` keyword.
///
/// # Returns
///
/// `Ok(())` when `base` is finite and strictly positive.
///
/// # Errors
///
/// * [`PyValueError`] — base is non-finite, zero, or negative.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// np.logspace(0, 1, base=0)   # ValueError
/// np.logspace(0, 1, base=-2)  # ValueError
/// ```
pub fn check_logspace_base(base: f64) -> PyResult<()> {
    if !base.is_finite() || base <= 0.0 {
        return Err(value_error(
            "logspace base must be finite and greater than zero",
        ));
    }
    Ok(())
}

/// Validate `geomspace` endpoints: finite, non-zero, and same sign.
///
/// Geometric sequences multiply ratios across the interval; zero endpoints
/// or mixed signs would break the log-linear construction used internally.
/// Delegates finiteness to [`check_finite_bounds`].
///
/// # Arguments
///
/// * `start` — First geometric endpoint from Python.
/// * `stop` — Last geometric endpoint from Python.
///
/// # Returns
///
/// `Ok(())` when endpoints are valid for geometric spacing.
///
/// # Errors
///
/// * [`PyValueError`] — non-finite bounds, zero endpoint, or opposite signs.
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

/// Reject a slice step of zero before index parsing continues.
///
/// NumPy slice semantics treat `step=0` as an error. Index parsing may see
/// `None` (default step 1) or a concrete integer; only `Some(0)` fails.
///
/// # Arguments
///
/// * `step` — Optional step from a Python slice object.
///
/// # Returns
///
/// `Ok(())` when step is absent or non-zero.
///
/// # Errors
///
/// * [`PyValueError`] — step is exactly zero.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// a = np.arange(10)
/// a[0:10:0]  # ValueError
/// ```
pub fn check_slice_step(step: Option<i64>) -> PyResult<()> {
    if step == Some(0) {
        return Err(value_error("slice step cannot be zero"));
    }
    Ok(())
}

/// Validate a `reshape` target: at most one `-1`, product matches size.
///
/// NumPy allows one inferred dimension marked `-1`. This helper validates
/// the Python shape tuple before the core reallocates or returns a view,
/// including overflow checks on the known dimension product.
///
/// # Arguments
///
/// * `dims` — Target shape from Python (may contain one `-1`).
/// * `size` — Total element count of the source array.
///
/// # Returns
///
/// `Ok(())` when the shape is compatible with `size`.
///
/// # Errors
///
/// * [`PyValueError`] — empty shape, multiple `-1`, negative dim (other than
///   `-1`), product mismatch, or size overflow.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// a = np.arange(12)
/// a.reshape(3, 4)     # ok
/// a.reshape(3, -1)    # ok (infers 4)
/// a.reshape(3, 5)     # ValueError
/// ```
pub fn check_reshape_shape(dims: &[isize], size: usize) -> PyResult<()> {
    if dims.is_empty() {
        return Err(value_error("reshape target shape must be non-empty"));
    }

    let mut inferred = None;
    let mut known = 1_usize;
    for (i, d) in dims.iter().copied().enumerate() {
        if d == -1 {
            if inferred.is_some() {
                return Err(value_error(
                    "only one reshape dimension may be -1",
                ));
            }
            inferred = Some(i);
        } else if d < 0 {
            return Err(value_error(format!("invalid reshape dimension {d}")));
        } else {
            let dim = d as usize;
            known = known.checked_mul(dim).ok_or_else(|| {
                value_error("reshape shape size overflows usize")
            })?;
        }
    }

    if inferred.is_some() {
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
    } else {
        if known != size {
            return Err(value_error(format!(
                "cannot reshape array of size {size} into shape {dims:?}"
            )));
        }
    }

    Ok(())
}

/// Validate `permute_axes`: length matches `ndim` and entries form a permutation.
///
/// Transpose-like operations require a complete reordering of dimensions.
/// Every axis must be in bounds, and no two entries may refer to the same
/// dimension (stricter than [`check_axes`], which allows duplicates in other
/// contexts).
///
/// # Arguments
///
/// * `axes` — Desired axis order from Python.
/// * `ndim` — Rank of the input array.
///
/// # Returns
///
/// `Ok(())` when `axes` is a permutation of `0..ndim-1`.
///
/// # Errors
///
/// * [`PyValueError`] — wrong length or not a permutation.
/// * [`PyIndexError`] — any axis out of bounds.
pub fn check_permute_axes(axes: &[isize], ndim: usize) -> PyResult<()> {
    if axes.len() != ndim {
        return Err(value_error(format!(
            "axes length {} does not match ndim {ndim}",
            axes.len()
        )));
    }
    for (index, &axis) in axes.iter().enumerate() {
        check_axis(axis, ndim)?;
        if axes[..index].iter().any(|&previous| {
            (0..ndim).any(|dimension| {
                axis_refers_to(axis, dimension, ndim)
                    && axis_refers_to(previous, dimension, ndim)
            })
        }) {
            return Err(value_error("axes must be a permutation of 0..ndim-1"));
        }
    }
    Ok(())
}

/// When axes are given to `squeeze`, each must refer to a length-1 dimension.
///
/// Explicit-axis squeeze rejects dimensions that are not singletons. When
/// `axes` is `None`, no check runs here (the core removes all length-1 dims).
///
/// # Arguments
///
/// * `inner` — Array being squeezed.
/// * `axes` — Optional explicit axis list from Python.
///
/// # Returns
///
/// `Ok(())` when every listed axis has length 1.
///
/// # Errors
///
/// * [`PyValueError`] — axis length is not 1, or axis list invalid.
/// * [`PyIndexError`] — axis out of bounds.
pub fn check_squeeze_axes(
    inner: &ArrayInner,
    axes: Option<&[isize]>,
) -> PyResult<()> {
    if let Some(axes) = axes {
        check_axes(axes, inner.ndim())?;
        for &axis in axes {
            let length = inner
                .shape()
                .iter()
                .enumerate()
                .find_map(|(dimension, &length)| {
                    axis_refers_to(axis, dimension, inner.ndim())
                        .then_some(length)
                })
                .expect("axis was validated above");
            if length != 1 {
                return Err(value_error(format!(
                    "cannot squeeze axis {axis} with length {length}"
                )));
            }
        }
    }
    Ok(())
}

/// Require bool storage dtype (e.g. for `where` condition arrays).
///
/// Boolean masking APIs expect [`PyDType::Bool`] storage, not integer 0/1
/// arrays. This produces a clear [`PyTypeError`] before typed kernels run.
///
/// # Arguments
///
/// * `inner` — Array coerced from a Python operand.
/// * `name` — Parameter name for the error message (e.g. `"condition"`).
///
/// # Returns
///
/// `Ok(())` when `inner.dtype()` is bool.
///
/// # Errors
///
/// * [`PyTypeError`] — array is not bool dtype.
pub fn require_bool_array(inner: &ArrayInner, name: &str) -> PyResult<()> {
    if inner.dtype() != PyDType::Bool {
        return Err(type_error(format!("{name} must be a bool array")));
    }
    Ok(())
}

/// Accept only `"xy"` or `"ij"` meshgrid indexing modes.
///
/// NumPy's `meshgrid` switches whether the first input varies along rows or
/// columns. Unknown strings are rejected at the boundary.
///
/// # Arguments
///
/// * `indexing` — Python `indexing` keyword value.
///
/// # Returns
///
/// `Ok(())` for `"xy"` or `"ij"`.
///
/// # Errors
///
/// * [`PyValueError`] — any other string.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// np.meshgrid([1, 2], [3, 4], indexing="xy")  # ok
/// np.meshgrid([1, 2], [3, 4], indexing="ij")  # ok
/// ```
pub fn check_meshgrid_indexing(indexing: &str) -> PyResult<()> {
    match indexing {
        "xy" | "ij" => Ok(()),
        other => Err(value_error(format!(
            "meshgrid indexing must be 'xy' or 'ij', got '{other}'"
        ))),
    }
}

/// Restrict `nditer` to one or two operands.
///
/// The current Python binding supports binary ufunc-style iteration only.
/// Operand count is validated before dtype and broadcast checks.
///
/// # Arguments
///
/// * `n` — Number of [`PyArray`] operands passed from Python.
///
/// # Returns
///
/// `Ok(())` when `n` is 1 or 2.
///
/// # Errors
///
/// * [`PyValueError`] — zero operands or more than two.
pub fn check_nditer_operands(n: usize) -> PyResult<()> {
    if n == 0 || n > 2 {
        return Err(value_error(
            "nditer supports 1-2 operands with the same dtype",
        ));
    }
    Ok(())
}

/// Require all `nditer` operands to share one dtype.
///
/// Mixed-type iteration is not implemented; every operand must match the first
/// array's [`PyDType`] tag after coercion.
///
/// # Arguments
///
/// * `arrays` — One or two [`PyArray`] references from Python.
///
/// # Returns
///
/// `Ok(())` when all operands share the first dtype.
///
/// # Errors
///
/// * [`PyValueError`] — dtype mismatch between operands.
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

/// NumPy-style trailing-axis broadcast check for several shapes.
///
/// Operands are aligned from the right; size-1 dimensions broadcast. This
/// mirrors NumPy's shape rules for ufuncs and is used before core broadcast
/// planning for multi-input Python APIs.
///
/// # Arguments
///
/// * `operation` — Name for error messages (e.g. `"add"`).
/// * `shapes` — Slice of shape slices to compare pairwise by trailing axis.
///
/// # Returns
///
/// `Ok(())` when every shape is mutually broadcastable.
///
/// # Errors
///
/// * [`PyValueError`] — incompatible trailing dimensions.
///
/// # Examples
///
/// ```rust
/// use sdnp_py::validate::check_broadcastable;
///
/// assert!(check_broadcastable("add", &[&[3, 1], &[3, 4]]).is_ok());
/// assert!(check_broadcastable("add", &[&[3, 4], &[3, 5]]).is_err());
/// ```
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
            // Length 1 broadcasts; otherwise sizes must agree.
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

/// Validate matrix multiply contraction and batch broadcast rules.
///
/// `@` / `matmul` requires matching inner dimensions and broadcast-compatible
/// batch prefixes. 0-D operands are rejected here; vector cases follow NumPy
/// after this check passes.
///
/// # Arguments
///
/// * `left` — Left-hand [`ArrayInner`] operand.
/// * `right` — Right-hand [`ArrayInner`] operand.
///
/// # Returns
///
/// `Ok(())` when contraction and batch shapes are compatible.
///
/// # Errors
///
/// * [`PyValueError`] — 0-D operand, inner dim mismatch, or batch mismatch.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// a = np.ones((2, 3))
/// b = np.ones((3, 4))
/// a @ b  # ok
/// ```
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

/// Validate `dot` operands: rank 1–2 only, then [`check_matmul`] rules.
///
/// Python `dot` is restricted to vector and matrix cases in this binding.
/// Higher-rank tensors must use `@` instead; rank is checked before delegating
/// to the shared matmul shape validator.
///
/// # Arguments
///
/// * `left` — Left-hand operand.
/// * `right` — Right-hand operand.
///
/// # Returns
///
/// `Ok(())` when ranks are 1 or 2 and matmul rules pass.
///
/// # Errors
///
/// * [`PyValueError`] — invalid rank or matmul shape failure.
pub fn check_dot(left: &ArrayInner, right: &ArrayInner) -> PyResult<()> {
    if !(1..=2).contains(&left.ndim()) || !(1..=2).contains(&right.ndim()) {
        return Err(value_error("dot supports only 1-D or 2-D operands"));
    }
    check_matmul(left, right)
}

/// Validate `vdot`: equal flattened element counts.
///
/// Vector dot products flatten both operands logically; mismatched sizes are
/// rejected before the reduction kernel accumulates.
///
/// # Arguments
///
/// * `left` — First 1-D (or treated-as-1-D) operand.
/// * `right` — Second operand.
///
/// # Returns
///
/// `Ok(())` when `left.size() == right.size()`.
///
/// # Errors
///
/// * [`PyValueError`] — size mismatch.
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

/// Validate `diagonal` / `trace` axes: ndim ≥ 2 and distinct axes.
///
/// Both APIs pick two dimensions to form a matrix slice; they must differ
/// after negative-axis resolution. Rank-0 and rank-1 inputs are rejected.
///
/// # Arguments
///
/// * `ndim` — Rank of the input array.
/// * `axis1` — First diagonal axis from Python.
/// * `axis2` — Second diagonal axis from Python.
///
/// # Returns
///
/// `Ok(())` when axes are valid and distinct.
///
/// # Errors
///
/// * [`PyValueError`] — `ndim < 2` or `axis1` equals `axis2`.
/// * [`PyIndexError`] — either axis out of bounds.
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
    check_axis(axis1, ndim)?;
    check_axis(axis2, ndim)?;
    if (0..ndim).any(|dimension| {
        axis_refers_to(axis1, dimension, ndim)
            && axis_refers_to(axis2, dimension, ndim)
    }) {
        return Err(value_error("axis1 and axis2 must be different"));
    }
    Ok(())
}

/// Validate `tril` / `triu` input: at least one dimension.
///
/// Triangular masks are undefined for 0-D scalars in NumPy-style APIs.
///
/// # Arguments
///
/// * `name` — `"tril"` or `"triu"` for the error message.
/// * `inner` — Input array.
///
/// # Returns
///
/// `Ok(())` when `inner.ndim() >= 1`.
///
/// # Errors
///
/// * [`PyValueError`] — 0-D input.
pub fn check_triangle_input(name: &str, inner: &ArrayInner) -> PyResult<()> {
    if inner.ndim() == 0 {
        return Err(value_error(format!(
            "{name} requires an array of at least one dimension"
        )));
    }
    Ok(())
}

/// Validate `diag` input: dtype must be numeric and rank must be 1 or 2.
///
/// Extracting or constructing diagonals applies to vectors and matrices only;
/// boolean arrays and higher-rank tensors are rejected at the Python boundary.
///
/// # Arguments
///
/// * `inner` — Input array.
///
/// # Returns
///
/// `Ok(())` for a non-boolean array whose rank is 1 or 2.
///
/// # Errors
///
/// * [`PyValueError`] — input is boolean or rank is not 1 or 2.
///
/// # Examples
///
/// ```python
/// import sdnp as np
/// np.diag(np.arange(3))        # ok
/// np.diag(np.eye(3))           # ok
/// np.diag(np.zeros((2, 2, 2))) # ValueError
/// ```
pub fn check_diag_input(inner: &ArrayInner) -> PyResult<()> {
    if matches!(inner, ArrayInner::Bool(_)) {
        return Err(value_error("diag does not support boolean arrays"));
    }
    if !matches!(inner.ndim(), 1 | 2) {
        return Err(value_error("diag requires a 1-D or 2-D array"));
    }
    Ok(())
}

/// Validate `meshgrid` inputs: all 1-D with a common dtype.
///
/// Coordinate grids are built from 1-D coordinate vectors. Rank and dtype
/// checks run before the core expands shapes.
///
/// # Arguments
///
/// * `arrays` — Sequence of coordinate arrays from Python.
///
/// # Returns
///
/// `Ok(())` when every input is 1-D and dtypes match.
///
/// # Errors
///
/// * [`PyValueError`] — non-1-D input or dtype mismatch.
pub fn check_meshgrid_arrays(arrays: &[ArrayInner]) -> PyResult<()> {
    if arrays.iter().any(|array| array.ndim() != 1) {
        return Err(value_error("meshgrid inputs must be 1-D arrays"));
    }
    check_same_dtype(arrays, "meshgrid")
}
