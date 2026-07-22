//! Type-specific reduction behaviour via traits.

use crate::dtype::{AsBool, CastTo, Complex64, Scalar};

/// Associative sum fold: `Acc` may differ from `Self` (e.g. bool → i64).
///
/// Used by [`crate::reduce::sum`] and [`crate::reduce::cumsum`].
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
}

/// Associative product fold.
///
/// Used by [`crate::reduce::prod`] and [`crate::reduce::cumprod`].
pub trait ProdReduce: Scalar {
    /// Accumulator / output element type.
    type Acc: Scalar;
    /// Multiplicative identity (empty reduction).
    fn identity() -> Self::Acc;
    /// Promote one element into the accumulator type.
    fn to_acc(self) -> Self::Acc;
    /// `acc * x` in the accumulator type.
    fn accumulate(acc: Self::Acc, x: Self) -> Self::Acc;
}

/// Orderable element for min / max / argmin / argmax (NaN-aware).
pub trait ExtremumReduce: Scalar + PartialOrd {
    /// Whether this value is NaN (only `f64` is true).
    fn is_nan(self) -> bool;
}

/// Sum-then-divide protocol for [`crate::reduce::mean`].
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
}

/// Element → `f64` cast for population [`crate::reduce::var`] / [`crate::reduce::std`].
pub trait VarReduce: Scalar {
    /// Cast to the floating value used by Welford’s algorithm.
    fn to_f64(self) -> f64;
}

/// Marker used by [`crate::reduce::any`] / [`crate::reduce::all`].
pub trait LogicalReduce: Scalar + AsBool {}
impl<T: Scalar + AsBool> LogicalReduce for T {}

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
        }
    };
}

sum_prod_same!(i64, 0_i64, 1_i64);
sum_prod_same!(f64, 0.0, 1.0);
sum_prod_same!(
    Complex64,
    Complex64::new(0.0, 0.0),
    Complex64::new(1.0, 0.0)
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
}

impl ExtremumReduce for i64 {
    #[inline]
    fn is_nan(self) -> bool {
        false
    }
}

impl ExtremumReduce for bool {
    #[inline]
    fn is_nan(self) -> bool {
        false
    }
}

impl ExtremumReduce for f64 {
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
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
}

impl VarReduce for bool {
    #[inline]
    fn to_f64(self) -> f64 {
        CastTo::<i64>::cast_to(self) as f64
    }
}

impl VarReduce for i64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl VarReduce for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}
