//! Public reduction free functions (generic, trait-dispatched).

use crate::array::Array;
use crate::error::Result;
use crate::reduction::kernels::transform_owned_c_order;
use crate::reduction::traits::{
    ExtremumReduce, LogicalReduce, MeanReduce, NanPolicy, ProdReduce,
    SumReduce, VarReduce,
};

/// Sum of elements along `axes` (bool accumulates as `i64`).
pub fn sum<T: SumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_sum(a, axes, keepdims, nan_policy)
}

/// Product of elements along `axes` (bool accumulates as `i64`).
pub fn prod<T: ProdReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_prod(a, axes, keepdims, nan_policy)
}

/// Minimum along `axes` (NaN propagates for `f64`).
pub fn min<T: ExtremumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T>> {
    T::reduce_min(a, axes, keepdims, nan_policy)
}

/// Maximum along `axes` (NaN propagates for `f64`).
pub fn max<T: ExtremumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T>> {
    T::reduce_max(a, axes, keepdims, nan_policy)
}

/// Arithmetic mean along `axes`.
///
/// - `output_len == 0` → empty result
/// - `reduction_len == 0` → error
pub fn mean<T: MeanReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_mean(a, axes, keepdims, nan_policy)
}

/// Argmin: flat C-order index if `axis is None`, else index along `axis`.
pub fn argmin<T: ExtremumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<i64>> {
    T::reduce_argmin(a, axis, nan_policy)
}

/// Argmax: flat C-order index if `axis is None`, else index along `axis`.
pub fn argmax<T: ExtremumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<i64>> {
    T::reduce_argmax(a, axis, nan_policy)
}

/// Cumulative sum along `axis`, or flattened C-order if `None`.
///
/// Bool promotes to `i64` (NumPy `int_`).
pub fn cumsum<T: SumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_cumsum(a, axis, nan_policy)
}

/// Cumulative product along `axis`, or flattened C-order if `None`.
///
/// Bool promotes to `i64` (NumPy `int_`).
pub fn cumprod<T: ProdReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_cumprod(a, axis, nan_policy)
}

/// Population variance (`ddof=0`) → `f64`.
pub fn var<T: VarReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<f64>> {
    T::reduce_var(a, axes, keepdims, nan_policy)
}

/// Population standard deviation (`sqrt(var)`).
pub fn std<T: VarReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<f64>> {
    let v = var(a, axes, keepdims, nan_policy)?;
    Ok(transform_owned_c_order(v, f64::sqrt))
}

/// Logical OR reduction.
pub fn any<T: LogicalReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    T::reduce_any(a, axes, keepdims)
}

/// Logical AND reduction.
pub fn all<T: LogicalReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    T::reduce_all(a, axes, keepdims)
}
