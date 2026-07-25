//! Stardust NumPy (`sdnp`): an educational reimplementation of ndarray
//! internals in Rust.
//!
//! The crate models NumPy's core ideas — shape, strides, broadcasting,
//! copy-on-write views, and dtype promotion — without a Python runtime.
//! Compile-time generics pick element types; a future PyO3 layer can wrap
//! this core for Python-facing APIs. Treat broadcast views as read-only when
//! planning buffer export or in-place mutation.
#![deny(missing_docs)]

/// Core [`Array`] type, element access, and view helpers.
pub mod array;
mod axis;
/// NumPy-style broadcasting: shape alignment and broadcast views.
pub mod broadcast;
pub(crate) mod creation;
pub(crate) mod dtype;
/// Error types and the crate-wide [`Result`] alias.
pub mod error;
/// Indexing, slicing, gather/scatter, and bounds resolution.
pub mod index;
pub(crate) mod iteration;
pub(crate) mod linalg;
pub(crate) mod manipulation;
pub(crate) mod reduction;
pub(crate) mod selection;
/// Shape products, C-order strides, and layout helpers.
pub mod shape;
pub(crate) mod sorting;
pub(crate) mod traversal;
/// Element-wise and binary universal functions (ufuncs).
pub mod ufunc;

/// N-dimensional array with shared storage and element-unit strides.
pub use array::Array;
/// Compute the broadcast-compatible shape of two input shapes.
pub use broadcast::{broadcast_arrays, broadcast_shape, broadcast_shapes};
/// Array constructors: fill, range, grid, and triangular helpers.
pub use creation::{
    arange, arange_stop, diag, eye, eye_with, full, geomspace, linspace,
    logspace, meshgrid, ones, tri, tri_with, tril, triu, zeros,
    MeshgridIndexing,
};
/// Scalar markers, promotion traits, and casting helpers.
pub use dtype::{ArrayCast, AsBool, CastTo, Complex64, Promote, Scalar};
/// Unified error enum and result alias for the crate.
pub use error::{Error, Result};
/// Gather/scatter indexing and the [`IndexSpec`] description type.
pub use index::{gather, scatter, scatter_array, IndexSpec};
/// Flat, axis-0, and N-dimensional iteration helpers.
pub use iteration::{
    ndenumerate, ndindex, nditer, Axis0Iter, FlatIter, NdEnumerate, NdIndex,
    NdIter,
};
/// Linear algebra: dot, matmul, trace, outer product, and diagonal ops.
pub use linalg::{diagonal, dot, matmul, outer, trace, vdot, ContractElement};
/// Stack, concatenate, and axis-wise joining of arrays.
pub use manipulation::{concatenate, hstack, stack, vstack};
/// Reductions, cumulative scans, and NaN-aware statistics.
pub use reduction::{
    all, any, argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum,
    var, ExtremumReduce, LogicalReduce, MeanReduce, NanPolicy, ProdReduce,
    SumReduce, VarReduce,
};
/// Selection utilities: clipping, nonzero indices, and where-masks.
pub use selection::{clip, nonzero, where_};
/// Compute C-order strides and the element count of a shape.
pub use shape::{c_order_strides, size_of_shape};
/// Sorting, argsort, and unique-value extraction.
pub use sorting::{
    argsort, sort, unique, unique_with, SortElement, UniqueElement,
    UniqueOptions, UniqueResult,
};
/// Arithmetic, comparison, and logical ufuncs over arrays.
pub use ufunc::{
    absolute, add, conj, divide, equal, greater, greater_equal, imag, isfinite,
    isinf, isnan, less, less_equal, logical_and, logical_not, logical_or,
    multiply, negative, not_equal, power, real, remainder, subtract,
    trunc_divide,
};
