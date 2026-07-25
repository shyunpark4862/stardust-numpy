//! Type-specific reduction behaviour via traits.

use crate::array::Array;
use crate::dtype::{AsBool, CastTo, Complex64, Scalar};
use crate::error::{Error, Result};
use crate::reduction::kernels::{
    arg_extremum_axis, arg_extremum_axis_ignore, arg_extremum_flat,
    arg_extremum_flat_ignore, cumulate, cumulate_ignore, reduce_associative,
    reduce_associative_with_plan, reduce_bool_extremum, reduce_bool_logical,
    reduce_f64_extremum, reduce_f64_extremum_ignore, reduce_fold,
    reduce_i64_max, reduce_i64_min, reduce_ignore_with_counts, reduce_var,
    reduce_var_ignore_f64, transform_owned_c_order,
};
use crate::reduction::plan::ReducePlan;

/// Controls how numeric reductions handle NaN values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NanPolicy {
    /// Preserve the existing behavior: a NaN propagates into the result.
    Propagate,
    /// Skip NaN elements; all-NaN non-empty slices produce NaN or an error.
    Ignore,
}

/// Associative sum fold: `Acc` may differ from `Self` (e.g. bool → i64).
///
/// Used by [`crate::reduction::sum`] and [`crate::reduction::cumsum`].
pub trait SumReduce: Scalar {
    /// Accumulator / output element type.
    type Acc: Scalar;
    /// Additive identity (empty reduction).
    fn identity() -> Self::Acc;
    /// Promote one element into the accumulator type.
    fn to_acc(self) -> Self::Acc;
    /// `acc + x` in the accumulator type.
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;
    /// Merge two partial accumulators.
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc;
    /// Type-dispatched sum implementation.
    fn reduce_sum(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
    /// Type-dispatched cumulative sum implementation.
    fn reduce_cumsum(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
}

/// Associative product fold.
///
/// Used by [`crate::reduction::prod`] and [`crate::reduction::cumprod`].
pub trait ProdReduce: Scalar {
    /// Accumulator / output element type.
    type Acc: Scalar;
    /// Multiplicative identity (empty reduction).
    fn identity() -> Self::Acc;
    /// Promote one element into the accumulator type.
    fn to_acc(self) -> Self::Acc;
    /// `acc * x` in the accumulator type.
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;
    /// Merge two partial accumulators.
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc;
    /// Type-dispatched product implementation.
    fn reduce_prod(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
    /// Type-dispatched cumulative product implementation.
    fn reduce_cumprod(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
}

/// Orderable element for min / max / argmin / argmax (NaN-aware).
pub trait ExtremumReduce: Scalar + PartialOrd {
    /// Type-specific minimum kernel.
    fn reduce_min(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self>>;
    /// Type-specific maximum kernel.
    fn reduce_max(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self>>;
    /// Whether this value is NaN (only `f64` is true).
    fn is_nan(self) -> bool;
    /// Type-specific argmin kernel.
    fn reduce_argmin(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<i64>>;
    /// Type-specific argmax kernel.
    fn reduce_argmax(
        a: &Array<Self>,
        axis: Option<isize>,
        nan_policy: NanPolicy,
    ) -> Result<Array<i64>>;
}

/// Sum-then-divide protocol for [`crate::reduction::mean`].
pub trait MeanReduce: Scalar {
    /// Mean accumulator / output type.
    type Acc: Scalar;
    /// Additive identity for the mean accumulator.
    fn identity() -> Self::Acc;
    /// Cast an element into the mean accumulator.
    fn to_acc(self) -> Self::Acc;
    /// `acc + x` (after promotion).
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;
    /// Merge two partial accumulators.
    fn combine(left: Self::Acc, right: Self::Acc) -> Self::Acc;
    /// Divide accumulator by the element count.
    fn divide_by_count(acc: Self::Acc, count: f64) -> Self::Acc;
    /// Type-dispatched mean implementation.
    fn reduce_mean(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<Self::Acc>>;
}

/// Element → `f64` cast for population [`crate::reduction::var`] / [`crate::reduction::std`].
pub trait VarReduce: Scalar {
    /// Cast to the floating value used by Welford’s algorithm.
    fn to_f64(self) -> f64;
    /// Type-dispatched population variance implementation.
    fn reduce_var(
        a: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        nan_policy: NanPolicy,
    ) -> Result<Array<f64>>;
}

/// Type-dispatched logical reduction used by [`crate::any`] and [`crate::all`].
pub trait LogicalReduce: Scalar + AsBool {
    /// Logical OR reduction.
    fn reduce_any(
        array: &Array<Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<bool>>;

    /// Logical AND reduction.
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
                acc + x
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
                acc * x
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

fn mean_propagate<T: MeanReduce>(
    a: &Array<T>,
    axes: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T::Acc>> {
    let plan = ReducePlan::new(a.shape(), axes, keepdims)?;
    if plan.output_len == 0 {
        return Array::from_vec(Vec::new(), &plan.output_shape);
    }
    if plan.reduction_is_empty() {
        return Err(Error::InvalidArgument("mean of empty array".into()));
    }
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
    if plan.reduction_is_empty() {
        return Err(Error::InvalidArgument("mean of empty array".into()));
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
