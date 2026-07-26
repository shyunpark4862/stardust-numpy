//! User-input validation at the Python boundary.
//!
//! The remaining checks intentionally enforce Python-only dispatch and surface
//! policies before calling the generic Rust core. Shared semantic invariants
//! remain in `sdnp` and surface via [`crate::error::map_sdnp`].

use pyo3::prelude::*;

use crate::array::PyArray;
use crate::dtype::PyDType;
use crate::error::{type_error, value_error};
use crate::inner::ArrayInner;

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
