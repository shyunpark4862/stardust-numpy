//! Element-wise ufuncs and promoted binary ops.
//!
//! - [`kernels`]: stride-aware map loops
//! - [`traits`]: per-dtype scalar arithmetic / classify helpers
//! - [`ops`]: public free functions (`add`, `divide`, …)

pub(crate) mod kernels;
pub(crate) mod ops;
pub(crate) mod traits;

pub use ops::{
    absolute, add, conj, divide, equal, greater, greater_equal, imag, isfinite,
    isinf, isnan, less, less_equal, logical_and, logical_not, logical_or,
    multiply, negative, not_equal, power, real, remainder, subtract,
    trunc_divide,
};
