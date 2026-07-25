//! Per-dtype reduction dispatch through traits.
//!
//! Each scalar type implements the traits for the operations it supports.
//! `NanPolicy` selects propagate vs ignore paths for floating reductions;
//! integer and boolean types ignore the policy. Trait methods delegate to
//! the shared kernels in `kernels` with type-specific accumulate rules.

use crate::array::Array;
use crate::dtype::{AsBool, CastTo, Complex64, Scalar};
use crate::error::Result;
use crate::reduction::kernels::{
    arg_extremum_axis, arg_extremum_axis_ignore, arg_extremum_flat,
    arg_extremum_flat_ignore, cumulate, cumulate_ignore, reduce_associative,
    reduce_associative_with_plan, reduce_bool_extremum, reduce_bool_logical,
    reduce_f64_extremum, reduce_f64_extremum_ignore, reduce_fold,
    reduce_i64_max, reduce_i64_min, reduce_ignore_with_counts, reduce_var,
    reduce_var_ignore_f64, transform_owned_c_order,
};
use crate::reduction::plan::ReducePlan;

/// How floating reductions treat NaN inputs.
///
/// Integer and boolean types ignore this policy. For `f64` and complex
/// dtypes, each variant selects a different kernel path in sum, product,
/// mean, variance, extremum, and cumulative operations.
///
/// # Examples
///
/// ```
/// use sdnp::{Array, NanPolicy, sum};
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0], &[3]).unwrap();
/// let p = sum(&a, None, false, NanPolicy::Propagate).unwrap();
/// assert!(p.item().unwrap().is_nan());
/// let i = sum(&a, None, false, NanPolicy::Ignore).unwrap();
/// assert_eq!(i.item().unwrap(), 4.0);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NanPolicy {
    /// NaN in the reduced slice poisons the result (NumPy default).
    ///
    /// Any NaN encountered during reduction makes the output NaN for that
    /// slot, even when other elements are finite.
    Propagate,
    /// Skip NaN elements; all-NaN non-empty slices yield NaN.
    ///
    /// Finite values are combined normally. When every element in a
    /// non-empty reduced slice is NaN, the result is NaN.
    Ignore,
}

/// Associative sum reduction for a scalar dtype.
///
/// The accumulator type `Acc` may differ from `Self` (e.g. `bool` → `i64`).
/// Implementors provide identity, promotion, and merge rules used by the
/// shared associative reduction kernels.
pub trait SumReduce: Scalar {
    /// Accumulator and output element type.
    type Acc: Scalar;

    /// Additive identity for an empty reduction.
    ///
    /// Returned when a reduced slice has length zero (e.g. `sum` over an
    /// empty axis). Integer types use wrapping zero; floats use `0.0`.
    ///
    /// # Arguments
    ///
    /// None — this is a type-level constant query.
    ///
    /// # Returns
    ///
    /// The neutral element for [`Self::accumulate`].
    fn identity() -> Self::Acc;

    /// Promote one input element into the accumulator type.
    ///
    /// Boolean input casts to `i64`; other types typically pass through.
    ///
    /// # Arguments
    ///
    /// * `self` - Input element before it enters the reduction kernel.
    ///
    /// # Returns
    ///
    /// `self` widened or cast to [`Self::Acc`].
    fn to_acc(self) -> Self::Acc;

    /// Combine one element into a running accumulator.
    ///
    /// Used by sequential folds and as the per-element step in associative
    /// eight-lane unrolling.
    ///
    /// # Arguments
    ///
    /// * `acc` - Partial sum so far.
    /// * `x` - Next input element (still type `Self`).
    ///
    /// # Returns
    ///
    /// Updated accumulator after adding `x`.
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;

    /// Merge two partial accumulators from independent chains.
    ///
    /// Must be associative with [`Self::accumulate`] so suffix-chunk kernels
    /// can combine eight lanes in parallel.
    ///
    /// # Arguments
    ///
    /// * `left` - Partial sum from one sub-chain.
    /// * `right` - Partial sum from another sub-chain.
    ///
    /// # Returns
    ///
    /// Combined accumulator value.
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc;

    /// Axis sum implementation for this dtype.
    ///
    /// Dispatches to associative or NaN-ignore kernels depending on
    /// [`NanPolicy`]. Floating and complex types branch at this boundary so
    /// hot loops stay branch-free.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
    /// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
    /// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
    ///
    /// # Returns
    ///
    /// An array of sums with dtype [`Self::Acc`].
    ///
    /// # Errors
    ///
    /// Returns an error when axis indices are invalid.
    fn reduce_sum(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;

    /// Prefix sum along one axis or in flat C order.
    ///
    /// Output shape matches the input. With `axis = None`, elements are
    /// scanned in C-order linear index.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axis` - Axis to scan, or `None` for flat C order.
    /// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
    ///
    /// # Returns
    ///
    /// Cumulative sums with dtype [`Self::Acc`], same shape as `a`.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis index is out of range.
    fn reduce_cumsum(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
}

/// Associative product reduction for a scalar dtype.
///
/// Mirrors [`SumReduce`]: `Acc` may differ from `Self` (e.g. `bool` →
/// `i64`), and floating dtypes honor [`NanPolicy`].
pub trait ProdReduce: Scalar {
    /// Accumulator and output element type.
    type Acc: Scalar;

    /// Multiplicative identity for an empty reduction.
    ///
    /// # Arguments
    ///
    /// None — this is a type-level constant query.
    ///
    /// # Returns
    ///
    /// The neutral element for [`Self::accumulate`] (typically `1`).
    fn identity() -> Self::Acc;

    /// Promote one input element into the accumulator type.
    ///
    /// # Arguments
    ///
    /// * `self` - Input element before it enters the reduction kernel.
    ///
    /// # Returns
    ///
    /// `self` widened or cast to [`Self::Acc`].
    fn to_acc(self) -> Self::Acc;

    /// Multiply one element into a running accumulator.
    ///
    /// # Arguments
    ///
    /// * `acc` - Partial product so far.
    /// * `x` - Next input element.
    ///
    /// # Returns
    ///
    /// Updated accumulator after multiplying by `x`.
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;

    /// Merge two partial accumulators from independent chains.
    ///
    /// # Arguments
    ///
    /// * `left` - Partial product from one sub-chain.
    /// * `right` - Partial product from another sub-chain.
    ///
    /// # Returns
    ///
    /// Combined accumulator value.
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc;

    /// Axis product implementation for this dtype.
    ///
    /// Mirrors [`SumReduce::reduce_sum`] but uses multiplicative identity and
    /// [`NanPolicy`] branches for floating types.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
    /// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
    /// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
    ///
    /// # Returns
    ///
    /// An array of products with dtype [`Self::Acc`].
    ///
    /// # Errors
    ///
    /// Returns an error when axis indices are invalid.
    fn reduce_prod(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;

    /// Prefix product along one axis or in flat C order.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axis` - Axis to scan, or `None` for flat C order.
    /// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
    ///
    /// # Returns
    ///
    /// Cumulative products with dtype [`Self::Acc`], same shape as `a`.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis index is out of range.
    fn reduce_cumprod(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
}

/// Orderable reductions: min, max, argmin, and argmax.
///
/// For `f64`, [`NanPolicy`] selects propagate vs ignore paths. Integer
/// and boolean types treat every value as finite.
pub trait ExtremumReduce: Scalar + PartialOrd {
    /// Minimum over selected axes.
    ///
    /// For `f64`, [`NanPolicy::Propagate`] poisons the result when any NaN is
    /// seen; [`NanPolicy::Ignore`] skips NaN elements.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
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
    fn reduce_min(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self>>;

    /// Maximum over selected axes.
    ///
    /// NaN semantics match [`Self::reduce_min`].
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
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
    fn reduce_max(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self>>;

    /// Whether this scalar should be treated as NaN.
    ///
    /// Only `f64` returns `true`; integer and boolean types always return
    /// `false`, letting shared arg/extremum kernels compile once.
    ///
    /// # Arguments
    ///
    /// * `self` - Scalar value to classify before extremum/arg reductions.
    ///
    /// # Returns
    ///
    /// `true` when the value is NaN for this dtype.
    fn is_nan(self) -> bool;

    /// Index of the minimum element along an axis or over the whole array.
    ///
    /// With `axis = None`, returns a flat C-order index (0-D output). With
    /// an axis, indices are relative to that axis length.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axis` - Axis along which to find minima, or `None` for flat index.
    /// * `nan_policy` - Propagate or ignore NaN for `f64` input.
    ///
    /// # Returns
    ///
    /// An `i64` array of indices.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is out of range or the slice is empty.
    fn reduce_argmin(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<i64>>;

    /// Index of the maximum element along an axis or over the whole array.
    ///
    /// Index semantics match [`Self::reduce_argmin`].
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axis` - Axis along which to find maxima, or `None` for flat index.
    /// * `nan_policy` - Propagate or ignore NaN for `f64` input.
    ///
    /// # Returns
    ///
    /// An `i64` array of indices.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis is out of range or the slice is empty.
    fn reduce_argmax(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<i64>>;
}

/// Sum-then-divide mean reduction.
///
/// Accumulates in type `Acc`, then divides by the element count (or the
/// count of non-NaN elements when [`NanPolicy::Ignore`] applies).
pub trait MeanReduce: Scalar {
    /// Mean accumulator and output type.
    type Acc: Scalar;

    /// Additive identity for the mean accumulator.
    ///
    /// # Arguments
    ///
    /// None — this is a type-level constant query.
    ///
    /// # Returns
    ///
    /// Zero in [`Self::Acc`] before summing.
    fn identity() -> Self::Acc;

    /// Cast an element into the mean accumulator.
    ///
    /// # Arguments
    ///
    /// * `self` - Input element before summing for the mean.
    ///
    /// # Returns
    ///
    /// `self` promoted to [`Self::Acc`].
    fn to_acc(self) -> Self::Acc;

    /// Add one element into the running sum used for the mean.
    ///
    /// # Arguments
    ///
    /// * `acc` - Partial sum so far.
    /// * `x` - Next input element.
    ///
    /// # Returns
    ///
    /// Updated sum accumulator.
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;

    /// Merge two partial sum accumulators.
    ///
    /// # Arguments
    ///
    /// * `left` - Partial sum from one sub-chain.
    /// * `right` - Partial sum from another sub-chain.
    ///
    /// # Returns
    ///
    /// Combined sum.
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc;

    /// Divide a sum accumulator by the number of contributing elements.
    ///
    /// Called once per output slot after summation (or after NaN-ignore
    /// counting).
    ///
    /// # Arguments
    ///
    /// * `acc` - Total sum for one output slot.
    /// * `count` - Number of finite elements folded into that slot.
    ///
    /// # Returns
    ///
    /// The arithmetic mean.
    fn divide_by_count(acc: Self::Acc, count: f64) -> Self::Acc;

    /// Axis mean implementation for this dtype.
    ///
    /// Sums via associative kernels, then divides by per-slot counts.
    /// [`NanPolicy::Ignore`] skips NaN and leaves all-NaN slots as NaN.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
    /// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
    /// * `nan_policy` - Propagate or ignore NaN for floating dtypes.
    ///
    /// # Returns
    ///
    /// An array of means with dtype [`Self::Acc`].
    ///
    /// # Errors
    ///
    /// Returns an error when axis indices are invalid or a reduced slice is
    /// empty.
    fn reduce_mean(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
}

/// Population variance and standard deviation (`ddof = 0`).
///
/// All outputs are `f64`. Floating input honors [`NanPolicy`].
pub trait VarReduce: Scalar {
    /// Cast to the `f64` value used in the two-pass variance algorithm.
    ///
    /// Every input dtype is converted to `f64` before mean and squared-
    /// deviation passes in the variance kernels.
    ///
    /// # Arguments
    ///
    /// * `self` - Input element to convert into an unbiased sample value.
    ///
    /// # Returns
    ///
    /// `self` as an `f64` sample value.
    fn to_f64(self) -> f64;

    /// Population variance (`ddof = 0`) over selected axes.
    ///
    /// Always returns `f64`. Uses a two-pass algorithm: mean, then sum of
    /// squared deviations. [`NanPolicy::Ignore`] applies only to `f64`
    /// input via a dedicated kernel path.
    ///
    /// # Arguments
    ///
    /// * `a` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
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
    fn reduce_var(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<f64>>;
}

/// Logical OR / AND reductions for [`crate::any`] and [`crate::all`].
///
/// Non-boolean scalars are converted via [`AsBool`] before folding.
/// [`NanPolicy`] does not apply to logical reductions.
pub trait LogicalReduce: Scalar + AsBool {
    /// True if any element is logically true over selected axes.
    ///
    /// Non-boolean scalars are converted via [`AsBool`]. [`NanPolicy`] does
    /// not apply. Empty reduced axes yield `false`.
    ///
    /// # Arguments
    ///
    /// * `array` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
    /// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
    ///
    /// # Returns
    ///
    /// A `bool` array of OR-reduction results.
    ///
    /// # Errors
    ///
    /// Returns an error when axis indices are invalid.
    fn reduce_any(
        array: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<bool>>;

    /// True if all elements are logically true over selected axes.
    ///
    /// Empty reduced axes yield `true` (vacuous truth), matching NumPy.
    ///
    /// # Arguments
    ///
    /// * `array` - Input array.
    /// * `axes` - Axes to reduce, or `None` for all axes.
    /// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
    ///
    /// # Returns
    ///
    /// A `bool` array of AND-reduction results.
    ///
    /// # Errors
    ///
    /// Returns an error when axis indices are invalid.
    fn reduce_all(
        array: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<bool>>;
}

impl LogicalReduce for bool {
    fn reduce_any(
        array: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<bool>> {
        reduce_bool_logical::<false>(array, axes, keepdims)
    }

    fn reduce_all(
        array: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<bool>> {
        reduce_bool_logical::<true>(array, axes, keepdims)
    }
}

macro_rules! logical_reduce {
    ($type:ty) => {
        impl LogicalReduce for $type {
            fn reduce_any(
                array: &Array<Self>,
                axes: Option<&[isize]>,
                keepdims: bool,
            ) -> Result<Array<bool>> {
                reduce_fold(array, axes, keepdims, false, |acc, value| {
                    acc || value.as_bool()
                })
            }

            fn reduce_all(
                array: &Array<Self>,
                axes: Option<&[isize]>,
                keepdims: bool,
            ) -> Result<Array<bool>> {
                reduce_fold(array, axes, keepdims, true, |acc, value| {
                    acc && value.as_bool()
                })
            }
        }
    };
}

logical_reduce!(i64);
logical_reduce!(f64);
logical_reduce!(Complex64);

macro_rules! sum_prod_same {
    ($t:ty, $zero:expr, $one:expr) => {
        impl SumReduce for $t {
            type Acc = $t;
            #[inline]
            fn identity() -> Self::Acc {
                $zero
            }
            #[inline]
            fn to_acc(self) -> Self::Acc {
                self
            }
            #[inline]
            fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
                acc.wrapping_add(x)
            }
            #[inline]
            fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
                left.wrapping_add(right)
            }
            fn reduce_sum(
                a: &Array<Self>,
                axes: Option<&[isize]>,
                keepdims: bool,
                _nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                reduce_associative(
                    a,
                    axes,
                    keepdims,
                    <Self as SumReduce>::identity(),
                    <Self as SumReduce>::accumulate,
                    <Self as SumReduce>::combine,
                )
            }
            fn reduce_cumsum(
                a: &Array<Self>,
                axis: Option<isize>,
                _nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                cumulate(
                    a,
                    axis,
                    <Self as SumReduce>::to_acc,
                    <Self as SumReduce>::accumulate,
                )
            }
        }
        impl ProdReduce for $t {
            type Acc = $t;
            #[inline]
            fn identity() -> Self::Acc {
                $one
            }
            #[inline]
            fn to_acc(self) -> Self::Acc {
                self
            }
            #[inline]
            fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
                acc.wrapping_mul(x)
            }
            #[inline]
            fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
                left.wrapping_mul(right)
            }
            fn reduce_prod(
                a: &Array<Self>,
                axes: Option<&[isize]>,
                keepdims: bool,
                _nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                reduce_associative(
                    a,
                    axes,
                    keepdims,
                    <Self as ProdReduce>::identity(),
                    <Self as ProdReduce>::accumulate,
                    <Self as ProdReduce>::combine,
                )
            }
            fn reduce_cumprod(
                a: &Array<Self>,
                axis: Option<isize>,
                _nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                cumulate(
                    a,
                    axis,
                    <Self as ProdReduce>::to_acc,
                    <Self as ProdReduce>::accumulate,
                )
            }
        }
    };
}

sum_prod_same!(i64, 0_i64, 1_i64);

macro_rules! sum_prod_nan {
    ($t:ty, $zero:expr, $one:expr, $nan:expr, $is_nan:expr) => {
        impl SumReduce for $t {
            type Acc = $t;
            fn identity() -> Self::Acc {
                $zero
            }
            fn to_acc(self) -> Self::Acc {
                self
            }
            fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
                acc + x
            }
            fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
                left + right
            }
            fn reduce_sum(
                a: &Array<Self>,
                axes: Option<&[isize]>,
                keepdims: bool,
                nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                match nan_policy {
                    NanPolicy::Propagate => reduce_associative(
                        a,
                        axes,
                        keepdims,
                        $zero,
                        <Self as SumReduce>::accumulate,
                        <Self as SumReduce>::combine,
                    ),
                    NanPolicy::Ignore => {
                        let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
                        let (result, _) = reduce_ignore_with_counts(
                            a,
                            &plan,
                            $zero,
                            $nan,
                            <Self as SumReduce>::accumulate,
                            <Self as SumReduce>::combine,
                            $is_nan,
                        )?;
                        Ok(result)
                    }
                }
            }
            fn reduce_cumsum(
                a: &Array<Self>,
                axis: Option<isize>,
                nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                match nan_policy {
                    NanPolicy::Propagate => cumulate(
                        a,
                        axis,
                        <Self as SumReduce>::to_acc,
                        <Self as SumReduce>::accumulate,
                    ),
                    NanPolicy::Ignore => cumulate_ignore(
                        a,
                        axis,
                        $nan,
                        <Self as SumReduce>::to_acc,
                        <Self as SumReduce>::accumulate,
                        $is_nan,
                    ),
                }
            }
        }
        impl ProdReduce for $t {
            type Acc = $t;
            fn identity() -> Self::Acc {
                $one
            }
            fn to_acc(self) -> Self::Acc {
                self
            }
            fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
                acc * x
            }
            fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
                left * right
            }
            fn reduce_prod(
                a: &Array<Self>,
                axes: Option<&[isize]>,
                keepdims: bool,
                nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                match nan_policy {
                    NanPolicy::Propagate => reduce_associative(
                        a,
                        axes,
                        keepdims,
                        $one,
                        <Self as ProdReduce>::accumulate,
                        <Self as ProdReduce>::combine,
                    ),
                    NanPolicy::Ignore => {
                        let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
                        let (result, _) = reduce_ignore_with_counts(
                            a,
                            &plan,
                            $one,
                            $nan,
                            <Self as ProdReduce>::accumulate,
                            <Self as ProdReduce>::combine,
                            $is_nan,
                        )?;
                        Ok(result)
                    }
                }
            }
            fn reduce_cumprod(
                a: &Array<Self>,
                axis: Option<isize>,
                nan_policy: NanPolicy,
            ) -> Result<Array<Self::Acc>> {
                match nan_policy {
                    NanPolicy::Propagate => cumulate(
                        a,
                        axis,
                        <Self as ProdReduce>::to_acc,
                        <Self as ProdReduce>::accumulate,
                    ),
                    NanPolicy::Ignore => cumulate_ignore(
                        a,
                        axis,
                        $nan,
                        <Self as ProdReduce>::to_acc,
                        <Self as ProdReduce>::accumulate,
                        $is_nan,
                    ),
                }
            }
        }
    };
}

sum_prod_nan!(f64, 0.0, 1.0, f64::NAN, f64::is_nan);
sum_prod_nan!(
    Complex64,
    Complex64::new(0.0, 0.0),
    Complex64::new(1.0, 0.0),
    Complex64::new(f64::NAN, f64::NAN),
    |value: Complex64| value.re.is_nan() || value.im.is_nan()
);

impl SumReduce for bool {
    type Acc = i64;
    #[inline]
    fn identity() -> Self::Acc {
        0
    }
    #[inline]
    fn to_acc(self) -> Self::Acc {
        CastTo::<i64>::cast_to(self)
    }
    #[inline]
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
        acc + CastTo::<i64>::cast_to(x)
    }
    #[inline]
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left + right
    }
    fn reduce_sum(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        reduce_associative(
            a,
            axes,
            keepdims,
            <Self as SumReduce>::identity(),
            <Self as SumReduce>::accumulate,
            <Self as SumReduce>::combine,
        )
    }
    fn reduce_cumsum(
        a: &Array<Self>,
        axis: Option<isize>,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        cumulate(
            a,
            axis,
            <Self as SumReduce>::to_acc,
            <Self as SumReduce>::accumulate,
        )
    }
}

impl ProdReduce for bool {
    type Acc = i64;
    #[inline]
    fn identity() -> Self::Acc {
        1
    }
    #[inline]
    fn to_acc(self) -> Self::Acc {
        CastTo::<i64>::cast_to(self)
    }
    #[inline]
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
        acc * CastTo::<i64>::cast_to(x)
    }
    #[inline]
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left * right
    }
    fn reduce_prod(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        reduce_associative(
            a,
            axes,
            keepdims,
            <Self as ProdReduce>::identity(),
            <Self as ProdReduce>::accumulate,
            <Self as ProdReduce>::combine,
        )
    }
    fn reduce_cumprod(
        a: &Array<Self>,
        axis: Option<isize>,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        cumulate(
            a,
            axis,
            <Self as ProdReduce>::to_acc,
            <Self as ProdReduce>::accumulate,
        )
    }
}

impl ExtremumReduce for i64 {
    fn reduce_min(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self>> {
        reduce_i64_min(a, axes, keepdims)
    }
    fn reduce_max(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self>> {
        reduce_i64_max(a, axes, keepdims)
    }
    #[inline]
    fn is_nan(self) -> bool {
        false
    }
    fn reduce_argmin(
        a: &Array<Self>,
        axis: Option<isize>,
        _nan_policy: NanPolicy,
    ) -> Result<Array<i64>> {
        match axis {
            None => arg_extremum_flat(
                a,
                "argmin",
                |candidate, best| candidate < best,
                |_| false,
            ),
            Some(axis) => arg_extremum_axis(
                a,
                axis,
                |candidate, best| candidate < best,
                |_| false,
            ),
        }
    }
    fn reduce_argmax(
        a: &Array<Self>,
        axis: Option<isize>,
        _nan_policy: NanPolicy,
    ) -> Result<Array<i64>> {
        match axis {
            None => arg_extremum_flat(
                a,
                "argmax",
                |candidate, best| candidate > best,
                |_| false,
            ),
            Some(axis) => arg_extremum_axis(
                a,
                axis,
                |candidate, best| candidate > best,
                |_| false,
            ),
        }
    }
}

impl ExtremumReduce for bool {
    fn reduce_min(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self>> {
        reduce_bool_extremum::<true>(a, axes, keepdims)
    }
    fn reduce_max(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self>> {
        reduce_bool_extremum::<false>(a, axes, keepdims)
    }
    #[inline]
    fn is_nan(self) -> bool {
        false
    }
    fn reduce_argmin(
        a: &Array<Self>,
        axis: Option<isize>,
        _nan_policy: NanPolicy,
    ) -> Result<Array<i64>> {
        match axis {
            None => arg_extremum_flat(
                a,
                "argmin",
                |candidate, best| !candidate && best,
                |_| false,
            ),
            Some(axis) => arg_extremum_axis(
                a,
                axis,
                |candidate, best| !candidate && best,
                |_| false,
            ),
        }
    }
    fn reduce_argmax(
        a: &Array<Self>,
        axis: Option<isize>,
        _nan_policy: NanPolicy,
    ) -> Result<Array<i64>> {
        match axis {
            None => arg_extremum_flat(
                a,
                "argmax",
                |candidate, best| candidate && !best,
                |_| false,
            ),
            Some(axis) => arg_extremum_axis(
                a,
                axis,
                |candidate, best| candidate && !best,
                |_| false,
            ),
        }
    }
}

impl ExtremumReduce for f64 {
    fn reduce_min(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self>> {
        match nan_policy {
            NanPolicy::Propagate => {
                reduce_f64_extremum::<true>(a, axes, keepdims)
            }
            NanPolicy::Ignore => {
                reduce_f64_extremum_ignore::<true>(a, axes, keepdims)
            }
        }
    }
    fn reduce_max(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self>> {
        match nan_policy {
            NanPolicy::Propagate => {
                reduce_f64_extremum::<false>(a, axes, keepdims)
            }
            NanPolicy::Ignore => {
                reduce_f64_extremum_ignore::<false>(a, axes, keepdims)
            }
        }
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn reduce_argmin(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<i64>> {
        match nan_policy {
            NanPolicy::Propagate => match axis {
                None => {
                    arg_extremum_flat(a, "argmin", |c, b| c < b, f64::is_nan)
                }
                Some(ax) => arg_extremum_axis(a, ax, |c, b| c < b, f64::is_nan),
            },
            NanPolicy::Ignore => match axis {
                None => arg_extremum_flat_ignore(
                    a,
                    "argmin",
                    |c, b| c < b,
                    f64::is_nan,
                ),
                Some(ax) => arg_extremum_axis_ignore(
                    a,
                    ax,
                    "argmin",
                    |c, b| c < b,
                    f64::is_nan,
                ),
            },
        }
    }
    fn reduce_argmax(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<i64>> {
        match nan_policy {
            NanPolicy::Propagate => match axis {
                None => {
                    arg_extremum_flat(a, "argmax", |c, b| c > b, f64::is_nan)
                }
                Some(ax) => arg_extremum_axis(a, ax, |c, b| c > b, f64::is_nan),
            },
            NanPolicy::Ignore => match axis {
                None => arg_extremum_flat_ignore(
                    a,
                    "argmax",
                    |c, b| c > b,
                    f64::is_nan,
                ),
                Some(ax) => arg_extremum_axis_ignore(
                    a,
                    ax,
                    "argmax",
                    |c, b| c > b,
                    f64::is_nan,
                ),
            },
        }
    }
}

/// Mean with NaN propagation (or non-floating types).
///
/// Every output slot folds the same number of elements. Sums via
/// [`reduce_associative_with_plan`], then divides each slot by
/// `plan.reduction_len`.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
///
/// # Returns
///
/// An array of means with dtype `T::Acc`.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or a reduced slice is
/// empty.
fn mean_propagate<T: MeanReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T::Acc>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    // Every slot folds the same number of elements.
    let count = plan.reduction_len as f64;
    let sums = reduce_associative_with_plan(
        a,
        &plan,
        T::identity(),
        T::accumulate,
        T::combine,
    )?;
    Ok(transform_owned_c_order(sums, |value| {
        T::divide_by_count(value, count)
    }))
}

/// Mean skipping NaN elements, dividing by per-slot finite counts.
///
/// Uses [`reduce_ignore_with_counts`] so each output slot tracks how many
/// non-NaN values contributed. Slots with count zero remain `nan`.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `axes` - Axes to reduce, or `None` for all axes.
/// * `keepdims` - When true, reduced axes remain as length-1 dimensions.
/// * `nan` - Sentinel written for slots with no finite inputs.
/// * `is_nan` - Predicate marking values to skip.
///
/// # Returns
///
/// An array of means with dtype `T::Acc`.
///
/// # Errors
///
/// Returns an error when axis indices are invalid or allocation fails.
fn mean_ignore<T: MeanReduce, N: Fn(T) -> bool>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
    nan: T::Acc,
    is_nan: N,
) -> Result<Array<T::Acc>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    let (mut sums, counts) = reduce_ignore_with_counts(
        a,
        &plan,
        T::identity(),
        nan,
        T::accumulate,
        T::combine,
        is_nan,
    )?;
    let data = std::sync::Arc::make_mut(&mut sums.data);
    // Divide only slots that saw at least one finite value.
    for (value, &count) in data.iter_mut().zip(&counts) {
        if count > 0 {
            *value = T::divide_by_count(*value, count as f64);
        }
    }
    Ok(sums)
}

impl MeanReduce for bool {
    type Acc = f64;
    #[inline]
    fn identity() -> Self::Acc {
        0.0
    }
    #[inline]
    fn to_acc(self) -> Self::Acc {
        CastTo::<f64>::cast_to(self)
    }
    #[inline]
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
        acc + CastTo::<f64>::cast_to(x)
    }
    #[inline]
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left + right
    }
    #[inline]
    fn divide_by_count(acc: Self::Acc, count: f64) -> Self::Acc {
        acc / count
    }
    fn reduce_mean(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        mean_propagate(a, axes, keepdims)
    }
}

impl MeanReduce for i64 {
    type Acc = f64;
    #[inline]
    fn identity() -> Self::Acc {
        0.0
    }
    #[inline]
    fn to_acc(self) -> Self::Acc {
        self as f64
    }
    #[inline]
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
        acc + (x as f64)
    }
    #[inline]
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left + right
    }
    #[inline]
    fn divide_by_count(acc: Self::Acc, count: f64) -> Self::Acc {
        acc / count
    }
    fn reduce_mean(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        mean_propagate(a, axes, keepdims)
    }
}

impl MeanReduce for f64 {
    type Acc = f64;
    #[inline]
    fn identity() -> Self::Acc {
        0.0
    }
    #[inline]
    fn to_acc(self) -> Self::Acc {
        self
    }
    #[inline]
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
        acc + x
    }
    #[inline]
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left + right
    }
    #[inline]
    fn divide_by_count(acc: Self::Acc, count: f64) -> Self::Acc {
        acc / count
    }
    fn reduce_mean(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        match nan_policy {
            NanPolicy::Propagate => mean_propagate(a, axes, keepdims),
            NanPolicy::Ignore => {
                mean_ignore(a, axes, keepdims, f64::NAN, f64::is_nan)
            }
        }
    }
}

impl MeanReduce for Complex64 {
    type Acc = Complex64;
    #[inline]
    fn identity() -> Self::Acc {
        Complex64::new(0.0, 0.0)
    }
    #[inline]
    fn to_acc(self) -> Self::Acc {
        self
    }
    #[inline]
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc {
        acc + x
    }
    #[inline]
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left + right
    }
    #[inline]
    fn divide_by_count(acc: Self::Acc, count: f64) -> Self::Acc {
        acc / count
    }
    fn reduce_mean(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>> {
        match nan_policy {
            NanPolicy::Propagate => mean_propagate(a, axes, keepdims),
            NanPolicy::Ignore => mean_ignore(
                a,
                axes,
                keepdims,
                Complex64::new(f64::NAN, f64::NAN),
                |value| value.re.is_nan() || value.im.is_nan(),
            ),
        }
    }
}

impl VarReduce for bool {
    #[inline]
    fn to_f64(self) -> f64 {
        CastTo::<i64>::cast_to(self) as f64
    }
    fn reduce_var(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<f64>> {
        reduce_var(a, axes, keepdims, <Self as VarReduce>::to_f64)
    }
}

impl VarReduce for i64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn reduce_var(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        _nan_policy: NanPolicy,
    ) -> Result<Array<f64>> {
        reduce_var(a, axes, keepdims, <Self as VarReduce>::to_f64)
    }
}

impl VarReduce for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    fn reduce_var(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<f64>> {
        match nan_policy {
            NanPolicy::Propagate => {
                reduce_var(a, axes, keepdims, <Self as VarReduce>::to_f64)
            }
            NanPolicy::Ignore => reduce_var_ignore_f64(a, axes, keepdims),
        }
    }
}
