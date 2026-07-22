//! Public reduction free functions (generic, trait-dispatched).

use crate::array::Array;
use crate::error::{Error, Result};
use crate::reduce::axis::ReducePlan;
use crate::reduce::kernels::{
    arg_extremum_axis, arg_extremum_flat, cumulate, map_owned_contiguous,
    reduce_extremum, reduce_fold, reduce_sum, reduce_sum_plan, reduce_var,
};
use crate::reduce::traits::{
    ExtremumReduce, LogicalReduce, MeanReduce, ProdReduce, SumReduce, VarReduce,
};

/// Sum of elements along `axes` (bool accumulates as `i64`).
pub fn sum<T: SumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T::Acc>> {
    reduce_sum(a, axes, keepdims, T::identity(), T::accumulate, T::combine)
}

/// Product of elements along `axes` (bool accumulates as `i64`).
pub fn prod<T: ProdReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T::Acc>> {
    reduce_fold(a, axes, keepdims, T::identity(), T::accumulate)
}

/// Minimum along `axes` (NaN propagates for `f64`).
pub fn min<T: ExtremumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T>> {
    reduce_extremum(a, axes, keepdims, "min", |c, b| c < b, T::is_nan)
}

/// Maximum along `axes` (NaN propagates for `f64`).
pub fn max<T: ExtremumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T>> {
    reduce_extremum(a, axes, keepdims, "max", |c, b| c > b, T::is_nan)
}

/// Arithmetic mean along `axes`.
///
/// - `outer_n == 0` → empty result
/// - `inner_n == 0` → error
pub fn mean<T: MeanReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T::Acc>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.outer_n == 0 {
        return Array::from_vec(Vec::new(), &plan.out_shape);
    }
    if plan.inner_is_empty() {
        return Err(Error::InvalidArgument("mean of empty array".into()));
    }
    let count = plan.inner_n as f64;
    let sums =
        reduce_sum_plan(a, &plan, T::identity(), T::accumulate, T::combine)?;
    Ok(map_owned_contiguous(sums, |x| T::divide_by_count(x, count)))
}

/// Argmin: flat C-order index if `axis is None`, else index along `axis`.
pub fn argmin<T: ExtremumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<i64>> {
    match axis {
        None => arg_extremum_flat(a, "argmin", |c, b| c < b, T::is_nan),
        Some(ax) => arg_extremum_axis(a, ax, |c, b| c < b, T::is_nan),
    }
}

/// Argmax: flat C-order index if `axis is None`, else index along `axis`.
pub fn argmax<T: ExtremumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<i64>> {
    match axis {
        None => arg_extremum_flat(a, "argmax", |c, b| c > b, T::is_nan),
        Some(ax) => arg_extremum_axis(a, ax, |c, b| c > b, T::is_nan),
    }
}

/// Cumulative sum along `axis`, or flattened C-order if `None`.
///
/// Bool promotes to `i64` (NumPy `int_`).
pub fn cumsum<T: SumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<T::Acc>> {
    cumulate(a, axis, T::to_acc, T::accumulate)
}

/// Cumulative product along `axis`, or flattened C-order if `None`.
///
/// Bool promotes to `i64` (NumPy `int_`).
pub fn cumprod<T: ProdReduce>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<T::Acc>> {
    cumulate(a, axis, T::to_acc, T::accumulate)
}

/// Population variance (`ddof=0`) → `f64`.
pub fn var<T: VarReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<f64>> {
    reduce_var(a, axes, keepdims, T::to_f64)
}

/// Population standard deviation (`sqrt(var)`).
pub fn std<T: VarReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<f64>> {
    let v = var(a, axes, keepdims)?;
    Ok(map_owned_contiguous(v, f64::sqrt))
}

/// Logical OR reduction.
pub fn any<T: LogicalReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    reduce_fold(a, axes, keepdims, false, |acc, v| acc || v.as_bool())
}

/// Logical AND reduction.
pub fn all<T: LogicalReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    reduce_fold(a, axes, keepdims, true, |acc, v| acc && v.as_bool())
}
