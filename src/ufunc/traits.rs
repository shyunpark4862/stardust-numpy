//! Per-dtype scalar traits for element-wise arithmetic and float classify.
//!
//! Division-like ops split by promoted output semantics:
//! - [`ElemDiv`] / [`ElemTruncDiv`] / [`ElemRem`] / [`ElemPow`]: infallible
//!   (`f64`, `Complex64`, and `bool` power) — IEEE-style `inf`/`nan`, fast path.
//! - [`FallibleElemDiv`] / …: explicit [`Error`](crate::error::Error) for
//!   `bool` and `i64` (`DivideByZero`, `InvalidArgument` on integer `power`).

use num_complex::Complex;
use num_traits::Float;

use crate::dtype::{Complex64, Scalar};
use crate::error::{Error, Result};

/// Element-wise `+` with bool-as-0/1 then nonzero→bool.
pub trait ElemAdd: Scalar {
    /// `self + rhs` (bool: numeric then truthiness).
    fn elem_add(self, rhs: Self) -> Self;
}

/// Element-wise `-`.
pub trait ElemSub: Scalar {
    /// `self - rhs`.
    fn elem_sub(self, rhs: Self) -> Self;
}

/// Element-wise `*`.
pub trait ElemMul: Scalar {
    /// `self * rhs`.
    fn elem_mul(self, rhs: Self) -> Self;
}

/// Infallible element-wise `/` (`f64` / `Complex64`: IEEE `inf`/`nan`).
pub trait ElemDiv: Scalar {
    /// `self / rhs`.
    fn elem_div(self, rhs: Self) -> Self;
}

/// Fallible element-wise `/` (`bool` / `i64`: [`Error::DivideByZero`]).
pub trait FallibleElemDiv: Scalar {
    /// `self / rhs`.
    fn elem_div(self, rhs: Self) -> Result<Self>;
}

/// Infallible truncating quotient (`f64` / `Complex64`).
pub trait ElemTruncDiv: Scalar {
    /// Truncating quotient toward zero.
    fn elem_trunc_div(self, rhs: Self) -> Self;
}

/// Fallible truncating quotient (`bool` / `i64`).
pub trait FallibleElemTruncDiv: Scalar {
    /// Truncating quotient toward zero.
    fn elem_trunc_div(self, rhs: Self) -> Result<Self>;
}

/// Infallible remainder (`f64` / `Complex64`).
pub trait ElemRem: Scalar {
    /// `self % rhs` (or complex analogue).
    fn elem_rem(self, rhs: Self) -> Self;
}

/// Fallible remainder (`bool` / `i64`).
pub trait FallibleElemRem: Scalar {
    /// `self % rhs`.
    fn elem_rem(self, rhs: Self) -> Result<Self>;
}

/// Infallible power (`bool`, `f64`, `Complex64`).
pub trait ElemPow: Scalar {
    /// `self.pow(rhs)`-like.
    fn elem_pow(self, rhs: Self) -> Self;
}

/// Fallible power (`i64`: negative or oversized exponent).
pub trait FallibleElemPow: Scalar {
    /// `self.pow(rhs)`-like.
    fn elem_pow(self, rhs: Self) -> Result<Self>;
}

/// Unary negation.
pub trait ElemNeg: Scalar {
    /// `-self`.
    fn elem_neg(self) -> Self;
}

/// Absolute value; may change type (`Complex` → `f64`).
pub trait ElemAbs: Scalar {
    /// Result type of `abs`.
    type Output: Scalar;
    /// Absolute value / magnitude.
    fn elem_abs(self) -> Self::Output;
}

macro_rules! impl_arith_via_ops {
    ($t:ty) => {
        impl ElemAdd for $t {
            #[inline]
            fn elem_add(self, rhs: Self) -> Self {
                self + rhs
            }
        }
        impl ElemSub for $t {
            #[inline]
            fn elem_sub(self, rhs: Self) -> Self {
                self - rhs
            }
        }
        impl ElemMul for $t {
            #[inline]
            fn elem_mul(self, rhs: Self) -> Self {
                self * rhs
            }
        }
        impl ElemNeg for $t {
            #[inline]
            fn elem_neg(self) -> Self {
                -self
            }
        }
    };
}

impl_arith_via_ops!(i64);
impl_arith_via_ops!(f64);
impl_arith_via_ops!(Complex64);

impl ElemAdd for bool {
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        (i64::from(self) + i64::from(rhs)) != 0
    }
}
impl ElemSub for bool {
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        (i64::from(self) - i64::from(rhs)) != 0
    }
}
impl ElemMul for bool {
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        (i64::from(self) * i64::from(rhs)) != 0
    }
}
impl ElemNeg for bool {
    #[inline]
    fn elem_neg(self) -> Self {
        // -True → -1 → true; -False → 0 → false
        (-i64::from(self)) != 0
    }
}

impl FallibleElemDiv for bool {
    #[inline]
    fn elem_div(self, rhs: Self) -> Result<Self> {
        if !rhs {
            return Err(Error::DivideByZero);
        }
        Ok((i64::from(self) / i64::from(rhs)) != 0)
    }
}
impl FallibleElemTruncDiv for bool {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Result<Self> {
        FallibleElemDiv::elem_div(self, rhs)
    }
}
impl FallibleElemRem for bool {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Result<Self> {
        if !rhs {
            return Err(Error::DivideByZero);
        }
        Ok((i64::from(self) % i64::from(rhs)) != 0)
    }
}
impl ElemPow for bool {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Self {
        i64::from(self).pow(u32::from(rhs)) != 0
    }
}

impl FallibleElemDiv for i64 {
    #[inline]
    fn elem_div(self, rhs: Self) -> Result<Self> {
        if rhs == 0 {
            return Err(Error::DivideByZero);
        }
        Ok(self / rhs)
    }
}
impl FallibleElemTruncDiv for i64 {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Result<Self> {
        FallibleElemDiv::elem_div(self, rhs)
    }
}
impl FallibleElemRem for i64 {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Result<Self> {
        if rhs == 0 {
            return Err(Error::DivideByZero);
        }
        Ok(self % rhs)
    }
}
impl FallibleElemPow for i64 {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Result<Self> {
        if rhs < 0 {
            return Err(Error::InvalidArgument(
                "i64 power with negative exponent is not supported".into(),
            ));
        }
        if rhs > u32::MAX as i64 {
            return Err(Error::InvalidArgument(
                "i64 power exponent too large".into(),
            ));
        }
        Ok(self.pow(rhs as u32))
    }
}

impl ElemDiv for f64 {
    #[inline]
    fn elem_div(self, rhs: Self) -> Self {
        self / rhs
    }
}
impl ElemTruncDiv for f64 {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Self {
        (self / rhs).trunc()
    }
}
impl ElemRem for f64 {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Self {
        self % rhs
    }
}
impl ElemPow for f64 {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Self {
        self.powf(rhs)
    }
}

impl ElemDiv for Complex64 {
    #[inline]
    fn elem_div(self, rhs: Self) -> Self {
        self / rhs
    }
}
impl ElemTruncDiv for Complex64 {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Self {
        let q = self / rhs;
        Complex::new(q.re.trunc(), q.im.trunc())
    }
}
impl ElemRem for Complex64 {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Self {
        let q = self.elem_trunc_div(rhs);
        self - q * rhs
    }
}
impl ElemPow for Complex64 {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Self {
        self.powc(rhs)
    }
}

impl ElemAbs for bool {
    type Output = bool;
    #[inline]
    fn elem_abs(self) -> Self::Output {
        self
    }
}
impl ElemAbs for i64 {
    type Output = i64;
    #[inline]
    fn elem_abs(self) -> Self::Output {
        self.abs()
    }
}
impl ElemAbs for f64 {
    type Output = f64;
    #[inline]
    fn elem_abs(self) -> Self::Output {
        self.abs()
    }
}
impl ElemAbs for Complex64 {
    type Output = f64;
    #[inline]
    fn elem_abs(self) -> Self::Output {
        self.norm()
    }
}

/// Marker used by `isnan` / `isinf` / `isfinite`.
pub trait FloatClassify: Scalar {
    /// `true` if NaN.
    fn is_nan(self) -> bool;
    /// `true` if infinite.
    fn is_infinite(self) -> bool;
    /// `true` if finite.
    fn is_finite(self) -> bool;
}

impl FloatClassify for f64 {
    #[inline]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        Float::is_infinite(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }
}

impl FloatClassify for Complex64 {
    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}
