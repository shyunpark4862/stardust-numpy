//! Public reduction and cumulative entry points.
//!
//! Each function is a thin generic wrapper over a dtype trait method.
//! Axis lists, `keepdims`, and [`NanPolicy`] are forwarded unchanged so
//! Python bindings and Rust callers share one dispatch path.

use crate::array::Array;
use crate::error::Result;
use crate::reduction::kernels::transform_owned_c_order;
use crate::reduction::traits::{
    ExtremumReduce, LogicalReduce, MeanReduce, NanPolicy, ProdReduce,
    SumReduce, VarReduce,
};

/// Sum elements over one or more axes.
///
/// Boolean input accumulates into `i64`, matching NumPy `int_`. Floating
/// and complex reductions honor [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
///
/// # Returns
///
/// An array of sums with dtype `T::Acc` (e.g. `i64` for `bool` input).
///
/// # Errors
///
/// Returns an error when axis indices are out of range or duplicate.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, sum};
///
/// let a = Array::from_vec(vec![1_i64, 2, 3], &[3]).unwrap();
/// let total = sum(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert_eq!(total.item().unwrap(), 6);
/// ```
pub fn sum<T: SumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_sum(a, axes, keepdims, nan_policy)
}

/// Product over one or more axes.
///
/// Boolean input accumulates into `i64`, matching NumPy `int_`. Floating
/// and complex reductions honor [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
///
/// # Returns
///
/// An array of products with dtype `T::Acc` (e.g. `i64` for `bool` input).
///
/// # Errors
///
/// Returns an error when axis indices are out of range or duplicate.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, prod};
///
/// let a = Array::from_vec(vec![2_i64, 3, 4], &[3]).unwrap();
/// let total = prod(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert_eq!(total.item().unwrap(), 24);
/// ```
pub fn prod<T: ProdReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_prod(a, axes, keepdims, nan_policy)
}

/// Element-wise minimum over one or more axes.
///
/// For `f64`, NaN handling follows `nan_policy`: [`NanPolicy::Propagate`]
/// poisons the result; [`NanPolicy::Ignore`] skips NaN elements.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for `f64` input.
///
/// # Returns
///
/// An array of minima with the same element type as the input.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or a reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, min};
///
/// let a = Array::from_vec(vec![3_i64, 1, 2], &[3]).unwrap();
/// let m = min(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert_eq!(m.item().unwrap(), 1);
/// ```
pub fn min<T: ExtremumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T>> {
    T::reduce_min(a, axes, keepdims, nan_policy)
}

/// Element-wise maximum over one or more axes.
///
/// For `f64`, NaN handling follows `nan_policy`: [`NanPolicy::Propagate`]
/// poisons the result; [`NanPolicy::Ignore`] skips NaN elements.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for `f64` input.
///
/// # Returns
///
/// An array of maxima with the same element type as the input.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or a reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, max};
///
/// let a = Array::from_vec(vec![3_i64, 1, 2], &[3]).unwrap();
/// let m = max(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert_eq!(m.item().unwrap(), 3);
/// ```
pub fn max<T: ExtremumReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T>> {
    T::reduce_max(a, axes, keepdims, nan_policy)
}

/// Arithmetic mean over one or more axes.
///
/// Empty output shape yields an empty array. A zero-length reduced block
/// is an error at the kernel layer. Floating reductions honor
/// [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
///
/// # Returns
///
/// An array of means with dtype `T::Acc` (typically `f64`).
///
/// # Errors
///
/// Returns an error when axis indices are invalid or a reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, mean};
///
/// let a = Array::from_vec(vec![2_i64, 4, 6], &[3]).unwrap();
/// let m = mean(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert_eq!(m.item().unwrap(), 4.0);
/// ```
pub fn mean<T: MeanReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_mean(a, axes, keepdims, nan_policy)
}

/// Index of the minimum element along an axis or over the whole array.
///
/// With `axis = None`, returns a flat C-order index. Otherwise the index
/// is along the chosen axis. For `f64`, NaN handling follows `nan_policy`.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to find minima, or `None` for a flat index.
/// * `nan_policy` - Propagate or ignore NaN for `f64` input.
///
/// # Returns
///
/// An `i64` array of indices (0-D when reducing all axes).
///
/// # Errors
///
/// Returns an error when the axis is out of range or the reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, argmin};
///
/// let a = Array::from_vec(vec![3_i64, 1, 2], &[3]).unwrap();
/// let idx = argmin(&a, None, NanPolicy::Propagate).unwrap();
/// assert_eq!(idx.item().unwrap(), 1);
/// ```
pub fn argmin<T: ExtremumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<i64>> {
    T::reduce_argmin(a, axis, nan_policy)
}

/// Index of the maximum element along an axis or over the whole array.
///
/// With `axis = None`, returns a flat C-order index. Otherwise the index
/// is along the chosen axis. For `f64`, NaN handling follows `nan_policy`.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axis` - Axis along which to find maxima, or `None` for a flat index.
/// * `nan_policy` - Propagate or ignore NaN for `f64` input.
///
/// # Returns
///
/// An `i64` array of indices (0-D when reducing all axes).
///
/// # Errors
///
/// Returns an error when the axis is out of range or the reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, argmax};
///
/// let a = Array::from_vec(vec![3_i64, 1, 2], &[3]).unwrap();
/// let idx = argmax(&a, None, NanPolicy::Propagate).unwrap();
/// assert_eq!(idx.item().unwrap(), 0);
/// ```
pub fn argmax<T: ExtremumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<i64>> {
    T::reduce_argmax(a, axis, nan_policy)
}

/// Prefix sum along one axis or in flat C order.
///
/// Boolean input promotes to `i64`, matching NumPy `int_`. Floating
/// reductions honor [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array (shape preserved in the output).
/// * `axis` - Axis along which to cumulate, or `None` for flat C order.
/// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
///
/// # Returns
///
/// An array of the same shape as the input with dtype `T::Acc`.
///
/// # Errors
///
/// Returns an error when the axis index is out of range.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, cumsum};
///
/// let a = Array::from_vec(vec![1_i64, 2, 3], &[3]).unwrap();
/// let c = cumsum(&a, None, NanPolicy::Propagate).unwrap();
/// assert_eq!(c.to_vec(), vec![1, 3, 6]);
/// ```
pub fn cumsum<T: SumReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_cumsum(a, axis, nan_policy)
}

/// Prefix product along one axis or in flat C order.
///
/// Boolean input promotes to `i64`, matching NumPy `int_`. Floating
/// reductions honor [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array (shape preserved in the output).
/// * `axis` - Axis along which to cumulate, or `None` for flat C order.
/// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
///
/// # Returns
///
/// An array of the same shape as the input with dtype `T::Acc`.
///
/// # Errors
///
/// Returns an error when the axis index is out of range.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, cumprod};
///
/// let a = Array::from_vec(vec![1_i64, 2, 3], &[3]).unwrap();
/// let c = cumprod(&a, None, NanPolicy::Propagate).unwrap();
/// assert_eq!(c.to_vec(), vec![1, 2, 6]);
/// ```
pub fn cumprod<T: ProdReduce>(
    a: &Array<T>,
    axis: Option<isize>,
    nan_policy: NanPolicy,
) -> Result<Array<T::Acc>> {
    T::reduce_cumprod(a, axis, nan_policy)
}

/// Population variance (`ddof = 0`), always returned as `f64`.
///
/// Floating reductions honor [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for `f64` input.
///
/// # Returns
///
/// An `f64` array of population variances.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or a reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, var};
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let v = var(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert!((v.item().unwrap() - 2.0 / 3.0).abs() < 1e-10);
/// ```
pub fn var<T: VarReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<f64>> {
    T::reduce_var(a, axes, keepdims, nan_policy)
}

/// Population standard deviation: square root of [`var`].
///
/// Floating reductions honor [`NanPolicy`].
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan_policy` - Propagate or ignore NaN for `f64` input.
///
/// # Returns
///
/// An `f64` array of population standard deviations.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or a reduced slice is
/// empty.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, std};
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let s = std(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert!((s.item().unwrap() - (2.0_f64 / 3.0).sqrt()).abs() < 1e-10);
/// ```
pub fn std<T: VarReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan_policy: NanPolicy,
) -> Result<Array<f64>> {
    let v = var(a, axes, keepdims, nan_policy)?;
    Ok(transform_owned_c_order(v, f64::sqrt))
}

/// True if any element is logically true over one or more axes.
///
/// # Arguments
///
/// * `a` - Input array (any scalar type with a logical interpretation).
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// A `bool` array of reduction results.
///
/// # Errors
///
/// Returns an error when axis indices are out of range or duplicate.
///
/// # Examples
///
/// ```
/// use sdnp::{any, Array};
///
/// let a = Array::from_vec(vec![0_i64, 0, 1], &[3]).unwrap();
/// let result = any(&a, None, false).unwrap();
/// assert_eq!(result.item().unwrap(), true);
/// ```
pub fn any<T: LogicalReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    T::reduce_any(a, axes, keepdims)
}

/// True if all elements are logically true over one or more axes.
///
/// # Arguments
///
/// * `a` - Input array (any scalar type with a logical interpretation).
/// * `axes` - Axes to reduce, or `None` to reduce all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// A `bool` array of reduction results.
///
/// # Errors
///
/// Returns an error when axis indices are out of range or duplicate.
///
/// # Examples
///
/// ```
/// use sdnp::{all, Array};
///
/// let a = Array::from_vec(vec![1_i64, 1, 1], &[3]).unwrap();
/// let result = all(&a, None, false).unwrap();
/// assert_eq!(result.item().unwrap(), true);
/// ```
pub fn all<T: LogicalReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>> {
    T::reduce_all(a, axes, keepdims)
}
