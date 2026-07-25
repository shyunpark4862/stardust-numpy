//! Per-dtype scalar traits backing element-wise ufuncs.
//!
//! Each trait defines one scalar operation applied inside map kernels after
//! dtype promotion. Division-like ops split into infallible IEEE paths (floats)
//! and fallible paths (integers/bools) that surface explicit errors instead of
//! `inf`/`nan`. Kernels call these methods once per coalesced run element.

use num_complex::Complex;
use num_traits::Float;

use crate::dtype::{Complex64, Scalar};
use crate::error::{Error, Result};

/// Element-wise addition; bool operands use 0/1 arithmetic then truthiness.
pub trait ElemAdd: Scalar {
    /// Add two scalars element-wise (`self + rhs`).
    ///
    /// Invoked inside broadcast-aware map kernels after operands are promoted
    /// to a common dtype. Integer paths use wrapping arithmetic; bool paths
    /// promote to `i64`, add, then coerce back to truthiness.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand operand already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// Sum of `self` and `rhs` in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_add(self, rhs: Self) -> Self;
}

/// Element-wise subtraction.
pub trait ElemSub: Scalar {
    /// Subtract two scalars element-wise (`self - rhs`).
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand operand already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// Difference `self - rhs` in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_sub(self, rhs: Self) -> Self;
}

/// Element-wise multiplication.
pub trait ElemMul: Scalar {
    /// Multiply two scalars element-wise (`self * rhs`).
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand operand already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// Product of `self` and `rhs` in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_mul(self, rhs: Self) -> Self;
}

/// Infallible element-wise division (`f64`/`Complex64`: IEEE `inf`/`nan`).
pub trait ElemDiv: Scalar {
    /// Divide two scalars element-wise (`self / rhs`).
    ///
    /// Floating and complex paths follow IEEE rules; divide-by-zero yields
    /// infinities or NaNs rather than errors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Divisor already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// Quotient of `self` and `rhs`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_div(self, rhs: Self) -> Self;
}

/// Fallible element-wise division (`bool`/`i64`: [`Error::DivideByZero`]).
pub trait FallibleElemDiv: Scalar {
    /// Divide two scalars element-wise with explicit error on invalid cases.
    ///
    /// Used by [`super::kernels::try_map_binary`] when promotion yields an
    /// integer or bool dtype that must not silently produce `inf`/`nan`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Divisor already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// `Ok(quotient)` when division is defined for these scalars.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DivideByZero`] when `rhs` is zero, or
    /// [`Error::InvalidArgument`] on integer overflow.
    fn elem_div(self, rhs: Self) -> Result<Self>;
}

/// Infallible truncating division toward zero.
pub trait ElemTruncDiv: Scalar {
    /// Quotient truncated toward zero (`trunc(self / rhs)`).
    ///
    /// For complexes, truncates real and imaginary parts separately after
    /// division.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Divisor already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// Truncated quotient in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_trunc_div(self, rhs: Self) -> Self;
}

/// Fallible truncating division toward zero.
pub trait FallibleElemTruncDiv: Scalar {
    /// Truncating division with explicit error on invalid integer/bool cases.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Divisor already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// `Ok(truncated_quotient)` when division is defined.
    ///
    /// # Errors
    ///
    /// Same failure modes as [`FallibleElemDiv::elem_div`].
    fn elem_trunc_div(self, rhs: Self) -> Result<Self>;
}

/// Infallible remainder.
pub trait ElemRem: Scalar {
    /// Remainder after division (`self % rhs` or complex analogue).
    ///
    /// Complex remainder uses truncated division: `self - trunc(self/rhs)*rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Divisor already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// Remainder in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_rem(self, rhs: Self) -> Self;
}

/// Fallible remainder.
pub trait FallibleElemRem: Scalar {
    /// Remainder with explicit error on divide-by-zero or overflow.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Divisor already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// `Ok(remainder)` when the operation is defined.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DivideByZero`] or [`Error::InvalidArgument`] on
    /// integer paths.
    fn elem_rem(self, rhs: Self) -> Result<Self>;
}

/// Infallible power (`bool`, `f64`, `Complex64`).
pub trait ElemPow: Scalar {
    /// Raise `self` to the power `rhs`.
    ///
    /// Floats use `powf`; complexes use `powc`; bools use `i64::pow` with a
    /// bool exponent interpreted as 0/1.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Exponent already coerced to `Self`.
    ///
    /// # Returns
    ///
    /// `self` raised to `rhs` in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_pow(self, rhs: Self) -> Self;
}

/// Fallible power (`i64`: rejects negative or oversized exponents).
pub trait FallibleElemPow: Scalar {
    /// Integer power with validated exponent range.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Exponent; must satisfy `0 <= rhs <= u32::MAX` for `i64`.
    ///
    /// # Returns
    ///
    /// `Ok(self.wrapping_pow(rhs as u32))` when the exponent is admissible.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `rhs` is negative or exceeds
    /// `u32::MAX`.
    fn elem_pow(self, rhs: Self) -> Result<Self>;
}

/// Unary negation.
pub trait ElemNeg: Scalar {
    /// Negate `self` element-wise (`-self`).
    ///
    /// Bool negation follows NumPy: `-True → -1 → True`, `-False → 0 → False`.
    ///
    /// # Arguments
    ///
    /// None beyond `self`.
    ///
    /// # Returns
    ///
    /// Negated value in the scalar domain of `Self`.
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
    fn elem_neg(self) -> Self;
}

/// Absolute value; may change dtype (`Complex64` → `f64`).
pub trait ElemAbs: Scalar {
    /// Output dtype of the absolute-value operation.
    ///
    /// Differs from `Self` for complex input (`Complex64` → `f64` magnitude).
    type Output: Scalar;

    /// Absolute value or complex magnitude.
    ///
    /// # Arguments
    ///
    /// None beyond `self`.
    ///
    /// # Returns
    ///
    /// `|self|` for reals; complex norm for [`Complex64`].
    ///
    /// # Errors
    ///
    /// This method does not fail for supported dtypes.
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

impl_arith_via_ops!(f64);
impl_arith_via_ops!(Complex64);

impl ElemAdd for i64 {
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}
impl ElemSub for i64 {
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}
impl ElemMul for i64 {
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}
impl ElemNeg for i64 {
    #[inline]
    fn elem_neg(self) -> Self {
        self.wrapping_neg()
    }
}

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
        // NumPy: -True → -1 → True; -False → 0 → False.
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
        self.checked_div(rhs).ok_or_else(|| {
            Error::InvalidArgument("integer division overflow".into())
        })
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
        self.checked_rem(rhs).ok_or_else(|| {
            Error::InvalidArgument("integer remainder overflow".into())
        })
    }
}
impl FallibleElemPow for i64 {
    #[inline]
    fn elem_pow(self, rhs: Self) -> Result<Self> {
        if rhs < 0 || rhs > u32::MAX as i64 {
            return Err(Error::InvalidArgument(
                "i64 power exponent must be in 0..=u32::MAX".into(),
            ));
        }
        Ok(self.wrapping_pow(rhs as u32))
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
        self.wrapping_abs()
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

/// Float classification hooks used by `isnan`, `isinf`, and `isfinite`.
pub trait FloatClassify: Scalar {
    /// Whether the value is NaN.
    ///
    /// For complexes, true if either component is NaN.
    ///
    /// # Arguments
    ///
    /// None beyond `self`.
    ///
    /// # Returns
    ///
    /// `true` when the value (or either complex component) is NaN.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    fn is_nan(self) -> bool;

    /// Whether the value is infinite.
    ///
    /// For complexes, true if either component is infinite.
    ///
    /// # Arguments
    ///
    /// None beyond `self`.
    ///
    /// # Returns
    ///
    /// `true` when the value (or either complex component) is infinite.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    fn is_infinite(self) -> bool;

    /// Whether the value is finite (neither NaN nor infinite).
    ///
    /// For complexes, both components must be finite.
    ///
    /// # Arguments
    ///
    /// None beyond `self`.
    ///
    /// # Returns
    ///
    /// `true` when the value is neither NaN nor infinite.
    ///
    /// # Errors
    ///
    /// This method does not fail.
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
