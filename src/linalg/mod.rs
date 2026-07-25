//! Linear algebra operations and contraction kernels.

pub(crate) mod diagonal_geometry;
mod geometry;
mod kernels;
mod ops;
mod traits;

pub use ops::{diagonal, dot, matmul, outer, trace, vdot};
pub use traits::ContractElement;
