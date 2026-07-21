//! Reductions and cumulative operations (`sum`, `mean`, `cumsum`, …).

mod axis;
mod kernels;
mod ops;
mod traits;

pub use ops::{
    all, any, argmax, argmin, cumprod, cumsum, max, mean, min, prod, std, sum,
    var,
};
pub use traits::{
    ExtremumReduce, LogicalReduce, MeanReduce, ProdReduce, SumReduce, VarReduce,
};
