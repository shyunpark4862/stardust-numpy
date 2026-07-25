//! Batched linear algebra: dot, matrix multiply, and related operations.
//!
//! Shape and stride planning live in [`geometry`] and [`diagonal_geometry`].
//! Numeric kernels in [`kernels`] execute prepared contraction plans.
//! The public API in [`ops`] mirrors common NumPy entry points (`dot`,
//! `matmul`, `vdot`, `outer`, `diagonal`, `trace`).

pub(crate) mod diagonal_geometry;
mod geometry;
mod kernels;
mod ops;
mod traits;

pub use ops::{diagonal, dot, matmul, outer, trace, vdot};
pub use traits::ContractElement;
