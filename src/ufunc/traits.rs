//! Per-dtype scalar traits for element-wise arithmetic and float classify.

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

/// Element-wise Rust `/` (int truncate; float IEEE).
pub trait ElemDiv: Scalar {
    /// `self / rhs`.
    fn elem_div(self, rhs: Self) -> Result<Self>;
}

/// Quotient toward zero (`trunc_divide`).
pub trait ElemTruncDiv: Scalar {
    /// Truncating quotient.
    fn elem_trunc_div(self, rhs: Self) -> Result<Self>;
}

/// Remainder (`%` semantics of the type).
pub trait ElemRem: Scalar {
    /// `self % rhs`.
    fn elem_rem(self, rhs: Self) -> Result<Self>;
}

/// Power.
pub trait ElemPow: Scalar {
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

impl ElemDiv for bool {
    #[inline]
    fn elem_div(self, rhs: Self) -> Result<Self> {
        if !rhs {
            return Err(Error::DivideByZero);
        }
        Ok((i64::from(self) / i64::from(rhs)) != 0)
    }
}
impl ElemTruncDiv for bool {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Result<Self> {
        self.elem_div(rhs)
    }
}
impl ElemRem for bool {
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
    fn elem_pow(self, rhs: Self) -> Result<Self> {
        Ok(i64::from(self).pow(u32::from(rhs)) != 0)
    }
}

impl ElemDiv for i64 {
    #[inline]
    fn elem_div(self, rhs: Self) -> Result<Self> {
        if rhs == 0 {
            return Err(Error::DivideByZero);
        }
        Ok(self / rhs)
    }
}
impl ElemTruncDiv for i64 {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Result<Self> {
        self.elem_div(rhs)
    }
}
impl ElemRem for i64 {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Result<Self> {
        if rhs == 0 {
            return Err(Error::DivideByZero);
        }
        Ok(self % rhs)
    }
}
impl ElemPow for i64 {
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
    fn elem_div(self, rhs: Self) -> Result<Self> {
        Ok(self / rhs)
    }
}
impl ElemTruncDiv for f64 {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Result<Self> {
        Ok((self / rhs).trunc())
    }
}
impl ElemRem for f64 {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Result<Self> {
        Ok(self % rhs)
    }
}
impl ElemPow for f64 {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Result<Self> {
        Ok(self.powf(rhs))
    }
}

impl ElemDiv for Complex64 {
    #[inline]
    fn elem_div(self, rhs: Self) -> Result<Self> {
        if rhs.re == 0.0 && rhs.im == 0.0 {
            if self.re == 0.0 && self.im == 0.0 {
                return Ok(Complex::new(f64::NAN, f64::NAN));
            }
            let inf = f64::INFINITY;
            return Ok(Complex::new(
                if self.re == 0.0 {
                    0.0
                } else {
                    self.re.signum() * inf
                },
                if self.im == 0.0 {
                    0.0
                } else {
                    self.im.signum() * inf
                },
            ));
        }
        Ok(self / rhs)
    }
}
impl ElemTruncDiv for Complex64 {
    #[inline]
    fn elem_trunc_div(self, rhs: Self) -> Result<Self> {
        let q = self.elem_div(rhs)?;
        Ok(Complex::new(q.re.trunc(), q.im.trunc()))
    }
}
impl ElemRem for Complex64 {
    #[inline]
    fn elem_rem(self, rhs: Self) -> Result<Self> {
        if rhs.re == 0.0 && rhs.im == 0.0 {
            return Ok(Complex::new(f64::NAN, f64::NAN));
        }
        let q = self.elem_trunc_div(rhs)?;
        Ok(self - q * rhs)
    }
}
impl ElemPow for Complex64 {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Result<Self> {
        Ok(self.powc(rhs))
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
