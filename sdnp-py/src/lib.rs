//! PyO3 module entry point for the `sdnp` Python package.

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
#[pymodule]
fn sdnp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_class::<FlatIter>()?;
    m.add_class::<Axis0Iter>()?;
    m.add_class::<iterators::PyNdIndex>()?;
    m.add_class::<iterators::PyNdEnumerate>()?;
    m.add_class::<iterators::PyNdIter>()?;

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
