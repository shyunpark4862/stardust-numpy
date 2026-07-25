//! Educational NumPy-style array library (`sdnp`).
//!
//! Core model: [`Array<T>`] with compile-time generics and automatic numeric
//! promotion (`bool < i64 < f64 < Complex<f64>`).
//!
//! A future **PyO3** binding can wrap this crate: Rust keeps the numeric core;
//! Python supplies dtype dispatch, dunders, and sequence conversion. Do not
//! expose writable zero-copy buffers to NumPy without an explicit ownership
//! policy — views use copy-on-write.
#![deny(missing_docs)]

pub mod array;
mod axis;
pub mod broadcast;
pub(crate) mod creation;
pub(crate) mod dtype;
pub mod error;
pub mod index;
pub(crate) mod iteration;
pub(crate) mod linalg;
pub(crate) mod manipulation;
pub(crate) mod reduction;
pub(crate) mod selection;
pub mod shape;
pub(crate) mod sorting;
pub(crate) mod traversal;
pub mod ufunc;

pub use array::Array;
pub use broadcast::{broadcast_arrays, broadcast_shape, broadcast_shapes};
pub use creation::{
    arange, arange_stop, diag, eye, eye_with, full, geomspace, linspace,
    logspace, meshgrid, ones, tri, tri_with, tril, triu, zeros,
    MeshgridIndexing,
};
pub use dtype::{ArrayCast, AsBool, CastTo, Complex64, Promote, Scalar};
pub use error::{Error, Result};
pub use index::{gather, scatter, scatter_array, IndexSpec};
pub use iteration::{
    ndenumerate, ndindex, nditer, Axis0Iter, FlatIter, NdEnumerate, NdIndex,
    NdIter,
};
pub use linalg::{diagonal, dot, matmul, outer, trace, vdot, ContractElement};
pub use manipulation::{concatenate, hstack, stack, vstack};
pub use reduction::{
    all, any, argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum,
    var, ExtremumReduce, LogicalReduce, MeanReduce, NanPolicy, ProdReduce,
    SumReduce, VarReduce,
};
pub use selection::{clip, nonzero, where_};
pub use shape::{c_order_strides, size_of_shape};
pub use sorting::{
    argsort, sort, unique, unique_with, SortElement, UniqueElement,
    UniqueOptions, UniqueResult,
};
pub use ufunc::{
    absolute, add, conj, divide, equal, greater, greater_equal, imag, isfinite,
    isinf, isnan, less, less_equal, logical_and, logical_not, logical_or,
    multiply, negative, not_equal, power, real, remainder, subtract,
    trunc_divide,
};
