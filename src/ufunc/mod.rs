//! Universal functions: element-wise operations on arrays.
//!
//! A ufunc applies one scalar operation to every pair of broadcast-aligned
//! elements, like NumPy's `np.add` or `np.divide`. Public entry points in
//! [`ops`] delegate to stride-aware [`kernels`] and per-dtype [`traits`].
//!
//! Traversal uses coalesced [`RunPlan`](crate::traversal::RunPlan) iteration
//! when operands are not C-contiguous; dtype promotion happens before kernels
//! invoke scalar trait methods.

pub(crate) mod kernels;
mod ops;
mod traits;

pub use ops::{
    absolute, add, conj, divide, equal, greater, greater_equal, imag, isfinite,
    isinf, isnan, less, less_equal, logical_and, logical_not, logical_or,
    multiply, negative, not_equal, power, real, remainder, subtract,
    trunc_divide,
};
