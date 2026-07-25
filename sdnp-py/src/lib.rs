//! PyO3 root for the `sdnp` Python package.
//!
//! This crate is the Python-facing shell around the generic Rust `sdnp` core.
//! Submodules handle dtype-tagged storage, coercion, validation, and PyErr
//! mapping; the core stays monomorphized over `bool`, `i64`, `f64`, and
//! `Complex64`. The module entry point registers classes, free functions, and
//! a stable `__all__` export list for IDE autocompletion.

mod array;
mod coerce;
mod creation;
mod dispatch;
mod dtype;
mod error;
mod index_parse;
mod inner;
mod iterators;
mod linalg;
mod manipulation;
mod reduction;
mod repr;
mod selection;
mod sorting;
mod ufunc;
mod unwrap;
mod validate;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use array::{Axis0Iter, FlatIter, PyArray};

/// Public names re-exported by `from sdnp import *`.
const __ALL__: &[&str] = &[
    "Array",
    "array",
    "zeros",
    "ones",
    "full",
    "arange",
    "linspace",
    "logspace",
    "geomspace",
    "meshgrid",
    "eye",
    "eye_with",
    "tri",
    "tri_with",
    "tril",
    "triu",
    "diag",
    "add",
    "subtract",
    "multiply",
    "divide",
    "trunc_divide",
    "remainder",
    "power",
    "negative",
    "absolute",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_not",
    "isnan",
    "isinf",
    "isfinite",
    "conj",
    "real",
    "imag",
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "var",
    "std",
    "any",
    "all",
    "argmin",
    "argmax",
    "cumsum",
    "cumprod",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "where",
    "nonzero",
    "clip",
    "sort",
    "argsort",
    "unique",
    "dot",
    "matmul",
    "vdot",
    "outer",
    "diagonal",
    "trace",
    "ndindex",
    "ndenumerate",
    "nditer",
];

/// Educational NumPy-style arrays backed by the Rust `sdnp` core.
///
/// Registers the `Array` class, iterator types, and all module-level free
/// functions. Exposes `__optimized__`, `__build_profile__`, and `__all__`
/// for tooling and IDE autocompletion.
///
/// # Arguments
///
/// * `m` - Bound reference to the freshly created extension module.
///
/// # Returns
///
/// The initialized `sdnp` Python module.
///
/// # Errors
///
/// Returns `PyErr` if any class or function registration fails.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert hasattr(np, "Array")
/// assert np.__optimized__ in (True, False)
/// assert "sum" in np.__all__
/// ```
#[pymodule]
fn sdnp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Surface build profile so benchmarks and tests can branch on it.
    m.add("__optimized__", !cfg!(debug_assertions))?;
    m.add(
        "__build_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    m.add_class::<PyArray>()?;
    m.add_class::<FlatIter>()?;
    m.add_class::<Axis0Iter>()?;
    m.add_class::<iterators::PyNdIndex>()?;
    m.add_class::<iterators::PyNdEnumerate>()?;
    m.add_class::<iterators::PyNdIter>()?;

    // Each submodule registers its own free functions on the module object.
    creation::register(m)?;
    ufunc::register(m)?;
    reduction::register(m)?;
    manipulation::register(m)?;
    selection::register(m)?;
    sorting::register(m)?;
    linalg::register(m)?;
    iterators::register(m)?;

    m.add("__all__", __ALL__)?;
    Ok(())
}
