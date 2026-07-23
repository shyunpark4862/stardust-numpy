//! Educational NumPy-style array library (`sdnp`).
//!
//! Core model: [`Array<T>`] with compile-time generics and automatic numeric
//! promotion (`bool < i64 < f64 < Complex<f64>`).
//!
//! A future **PyO3** binding (see `plan.md` Phase 8) is expected to wrap this
//! crate: Rust keeps the numeric core; Python supplies dtype dispatch, dunders,
//! and sequence conversion. Do not expose writable zero-copy buffers to NumPy
//! without an explicit ownership policy — views use copy-on-write.
#![deny(missing_docs)]

pub mod array;
pub mod broadcast;
pub mod create;
pub mod dtype;
pub mod error;
pub(crate) mod format;
pub mod index;
pub mod join;
pub(crate) mod layout;
pub(crate) mod linalg;
pub mod reduce;
pub mod select;
pub mod shape;
pub mod sort;
pub(crate) mod stride_iter;
pub mod ufunc;

pub use array::Array;
pub use broadcast::{broadcast_arrays, broadcast_shape, broadcast_shapes};
pub use create::{
    arange, arange_stop, eye, eye_with, full, geomspace, linspace, logspace,
    meshgrid, ones, zeros, MeshgridIndexing,
};
pub use dtype::{AsBool, CastTo, Complex64, Promote, Scalar};
pub use error::{Error, Result};
pub use index::{gather, scatter, scatter_array, IndexSpec};
pub use join::{concatenate, hstack, stack, vstack};
pub use reduce::{
    all, any, argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum,
    var,
};
pub use select::{clip, nonzero, where_};
pub use shape::{c_order_strides, size_of_shape};
pub use sort::{
    argsort, sort, sort_in_place, unique, unique_with, UniqueOptions,
    UniqueResult,
};
pub use ufunc::{
    absolute, add, conj, divide, equal, greater, greater_equal, imag, isfinite,
    isinf, isnan, less, less_equal, logical_and, logical_not, logical_or,
    multiply, negative, not_equal, power, real, remainder, subtract,
    trunc_divide,
};
