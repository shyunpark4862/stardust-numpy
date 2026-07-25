//! Axis reductions, cumulative scans, and statistical aggregates.
//!
//! Public functions dispatch through dtype-specific traits in `traits`.
//! `plan` resolves which axes collapse; `kernels` pick contiguous,
//! prefix/suffix, or general strided traversal. NaN policy is applied at the
//! trait layer so propagate and ignore paths stay separate in hot loops.

mod kernels;
mod ops;
mod plan;
mod traits;

pub use ops::{
    all, any, argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum,
    var,
};
pub use traits::{
    ExtremumReduce, LogicalReduce, MeanReduce, NanPolicy, ProdReduce,
    SumReduce, VarReduce,
};
