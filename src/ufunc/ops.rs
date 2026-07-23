//! Public element-wise free functions (`add`, `divide`, …).
//!
//! Cross-type ops use [`Promote`] + [`CastTo`]. No `std::ops` overloads
//! (broadcast errors stay in [`Result`]).
//!
//! After promotion, division-like ops dispatch by output type:
//! - `f64` / `Complex64`: infallible IEEE semantics, [`map_binary`] fast path.
//! - `bool` / `i64`: explicit element errors via [`try_map_binary`].

use crate::array::Array;
use crate::dtype::{AsBool, CastTo, Complex64, Promote, Scalar};
use crate::error::Result;
use crate::ufunc::kernels::{map_binary, map_unary, try_map_binary};
use crate::ufunc::traits::{
    ElemAbs, ElemAdd, ElemDiv, ElemMul, ElemNeg, ElemPow, ElemRem, ElemSub,
    ElemTruncDiv, FallibleElemDiv, FallibleElemPow, FallibleElemRem,
    FallibleElemTruncDiv, FloatClassify,
};

fn promote_binary<L, R, Out, F>(
    a: &Array<L>,
    b: &Array<R>,
    mut f: F,
) -> Result<Array<Out>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: Scalar,
    Out: Scalar,
    F: FnMut(L::Output, L::Output) -> Out,
{
    map_binary(a, b, |x, y| f(x.cast_to(), y.cast_to()))
}

fn try_promote_binary<L, R, Out, F>(
    a: &Array<L>,
    b: &Array<R>,
    mut f: F,
) -> Result<Array<Out>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: Scalar,
    Out: Scalar,
    F: FnMut(L::Output, L::Output) -> Result<Out>,
{
    try_map_binary(a, b, |x, y| f(x.cast_to(), y.cast_to()))
}

/// Routes division-like ufuncs to infallible or fallible kernels by promoted output.
pub trait DivDispatch: Scalar {
    /// Element-wise division after operands are promoted to `Self`.
    fn divide_promoted<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

/// Routes truncating division by promoted output type.
pub trait TruncDivDispatch: Scalar {
    /// Element-wise truncating division after promotion.
    fn trunc_divide_promoted<L, R>(
        a: &Array<L>,
        b: &Array<R>,
    ) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

/// Routes remainder by promoted output type.
pub trait RemDispatch: Scalar {
    /// Element-wise remainder after promotion.
    fn remainder_promoted<L, R>(
        a: &Array<L>,
        b: &Array<R>,
    ) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

/// Routes power by promoted output type.
pub trait PowDispatch: Scalar {
    /// Element-wise power after promotion.
    fn power_promoted<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

macro_rules! impl_infallible_div_dispatch {
    ($t:ty) => {
        impl DivDispatch for $t {
            fn divide_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                promote_binary(a, b, ElemDiv::elem_div)
            }
        }
        impl TruncDivDispatch for $t {
            fn trunc_divide_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                promote_binary(a, b, ElemTruncDiv::elem_trunc_div)
            }
        }
        impl RemDispatch for $t {
            fn remainder_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                promote_binary(a, b, ElemRem::elem_rem)
            }
        }
    };
}

macro_rules! impl_infallible_pow_dispatch {
    ($t:ty) => {
        impl PowDispatch for $t {
            fn power_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                promote_binary(a, b, ElemPow::elem_pow)
            }
        }
    };
}

macro_rules! impl_fallible_div_dispatch {
    ($t:ty) => {
        impl DivDispatch for $t {
            fn divide_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                try_promote_binary(a, b, FallibleElemDiv::elem_div)
            }
        }
        impl TruncDivDispatch for $t {
            fn trunc_divide_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                try_promote_binary(a, b, FallibleElemTruncDiv::elem_trunc_div)
            }
        }
        impl RemDispatch for $t {
            fn remainder_promoted<L, R>(
                a: &Array<L>,
                b: &Array<R>,
            ) -> Result<Array<Self>>
            where
                L: Promote<R, Output = Self> + CastTo<Self>,
                R: CastTo<Self>,
            {
                try_promote_binary(a, b, FallibleElemRem::elem_rem)
            }
        }
    };
}

impl_infallible_div_dispatch!(f64);
impl_infallible_div_dispatch!(Complex64);
impl_infallible_pow_dispatch!(bool);
impl_infallible_pow_dispatch!(f64);
impl_infallible_pow_dispatch!(Complex64);
impl_fallible_div_dispatch!(bool);
impl_fallible_div_dispatch!(i64);

impl PowDispatch for i64 {
    fn power_promoted<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<i64>>
    where
        L: Promote<R, Output = i64> + CastTo<i64>,
        R: CastTo<i64>,
    {
        try_promote_binary(a, b, FallibleElemPow::elem_pow)
    }
}

/// Element-wise addition with promotion.
pub fn add<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: ElemAdd,
{
    promote_binary(a, b, ElemAdd::elem_add)
}

/// Element-wise subtraction with promotion.
pub fn subtract<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: ElemSub,
{
    promote_binary(a, b, ElemSub::elem_sub)
}

/// Element-wise multiplication with promotion.
pub fn multiply<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: ElemMul,
{
    promote_binary(a, b, ElemMul::elem_mul)
}

/// Element-wise division after promotion.
///
/// `i64` truncates toward zero; `f64` / `Complex64` follow IEEE (`inf`/`nan`).
/// `bool` / `i64` division by zero returns
/// [`Error::DivideByZero`](crate::error::Error::DivideByZero).
pub fn divide<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: DivDispatch,
{
    L::Output::divide_promoted(a, b)
}

/// Truncating quotient (toward zero) after promotion.
///
/// For `i64` this matches [`divide`]. For `f64` returns `(a/b).trunc()`.
pub fn trunc_divide<L, R>(
    a: &Array<L>,
    b: &Array<R>,
) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: TruncDivDispatch,
{
    L::Output::trunc_divide_promoted(a, b)
}

/// Element-wise remainder after promotion.
pub fn remainder<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: RemDispatch,
{
    L::Output::remainder_promoted(a, b)
}

/// Element-wise power after promotion.
///
/// `f64` / `Complex64`: IEEE-style (`nan` for invalid real powers).
/// `i64`: negative or oversized exponent →
/// [`Error::InvalidArgument`](crate::error::Error::InvalidArgument).
pub fn power<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PowDispatch,
{
    L::Output::power_promoted(a, b)
}

/// Element-wise negation.
pub fn negative<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: ElemNeg,
{
    map_unary(a, ElemNeg::elem_neg)
}

/// Absolute value / magnitude (`Complex` → `f64`).
pub fn absolute<T>(a: &Array<T>) -> Result<Array<T::Output>>
where
    T: ElemAbs,
{
    map_unary(a, ElemAbs::elem_abs)
}

/// Element-wise `==` (result `bool`).
pub fn equal<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialEq,
{
    promote_binary(a, b, |x, y| x == y)
}

/// Element-wise `!=` (result `bool`).
pub fn not_equal<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialEq,
{
    promote_binary(a, b, |x, y| x != y)
}

macro_rules! ord_op {
    ($name:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
        where
            L: Promote<R> + CastTo<L::Output>,
            R: CastTo<L::Output>,
            L::Output: PartialOrd,
        {
            promote_binary(a, b, |x, y| x.$method(&y))
        }
    };
}

ord_op!(less, lt, "Element-wise `<` (result `bool`).");
ord_op!(less_equal, le, "Element-wise `<=` (result `bool`).");
ord_op!(greater, gt, "Element-wise `>` (result `bool`).");
ord_op!(greater_equal, ge, "Element-wise `>=` (result `bool`).");

/// Element-wise logical AND (truthiness via [`AsBool`]).
pub fn logical_and<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: AsBool,
    R: AsBool,
{
    map_binary(a, b, |x, y| x.as_bool() & y.as_bool())
}

/// Element-wise logical OR.
pub fn logical_or<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: AsBool,
    R: AsBool,
{
    map_binary(a, b, |x, y| x.as_bool() | y.as_bool())
}

/// Element-wise logical NOT.
pub fn logical_not<T: AsBool>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, |x| !x.as_bool())
}

/// Element-wise `isnan` (`f64` / `Complex64`).
pub fn isnan<T: FloatClassify>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, FloatClassify::is_nan)
}

/// Element-wise `isinf`.
pub fn isinf<T: FloatClassify>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, FloatClassify::is_infinite)
}

/// Element-wise `isfinite`.
pub fn isfinite<T: FloatClassify>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, FloatClassify::is_finite)
}

/// Complex conjugate.
pub fn conj(a: &Array<Complex64>) -> Result<Array<Complex64>> {
    map_unary(a, |z| z.conj())
}

/// Real part of each complex element.
pub fn real(a: &Array<Complex64>) -> Result<Array<f64>> {
    map_unary(a, |z| z.re)
}

/// Imaginary part.
pub fn imag(a: &Array<Complex64>) -> Result<Array<f64>> {
    map_unary(a, |z| z.im)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use num_complex::Complex;

    #[test]
    fn add_promote_i64_f64() {
        let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
        let b = Array::from_slice(&[0.5, 1.5], &[2]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.get(&[0]).unwrap(), 1.5);
        assert_eq!(c.get(&[1]).unwrap(), 3.5);
    }

    #[test]
    fn add_bool_bool() {
        let a = Array::from_slice(&[true, true, false], &[3]).unwrap();
        let b = Array::from_slice(&[true, false, false], &[3]).unwrap();
        let c = add(&a, &b).unwrap();
        assert!(c.get(&[0]).unwrap()); // 1+1 → true
        assert!(c.get(&[1]).unwrap()); // 1+0 → true
        assert!(!c.get(&[2]).unwrap());
    }

    #[test]
    fn divide_i64_truncates() {
        let a = Array::from_slice(&[7_i64, -7], &[2]).unwrap();
        let b = Array::from_slice(&[2_i64, 2], &[2]).unwrap();
        let c = divide(&a, &b).unwrap();
        assert_eq!(c.get(&[0]).unwrap(), 3);
        assert_eq!(c.get(&[1]).unwrap(), -3);
    }

    #[test]
    fn divide_i64_by_zero() {
        let a = Array::from_slice(&[1_i64], &[1]).unwrap();
        let b = Array::from_slice(&[0_i64], &[1]).unwrap();
        assert_eq!(divide(&a, &b).unwrap_err(), Error::DivideByZero);
    }

    #[test]
    fn divide_f64_by_zero_is_infinite() {
        let a = Array::from_slice(&[1.0_f64, -2.0, 0.0], &[3]).unwrap();
        let b = Array::from_slice(&[0.0_f64, 0.0, 0.0], &[3]).unwrap();
        let c = divide(&a, &b).unwrap();
        assert!(c.get(&[0]).unwrap().is_infinite());
        assert!(c.get(&[0]).unwrap().is_sign_positive());
        assert!(c.get(&[1]).unwrap().is_infinite());
        assert!(c.get(&[1]).unwrap().is_sign_negative());
        assert!(c.get(&[2]).unwrap().is_nan());
    }

    #[test]
    fn divide_i64_f64_promotes_to_f64_infinite() {
        let a = Array::from_slice(&[1_i64], &[1]).unwrap();
        let b = Array::from_slice(&[0.0_f64], &[1]).unwrap();
        let c = divide(&a, &b).unwrap();
        assert!(c.get(&[0]).unwrap().is_infinite());
    }

    #[test]
    fn power_i64_negative_exponent_errors() {
        let a = Array::from_slice(&[2_i64], &[1]).unwrap();
        let b = Array::from_slice(&[-1_i64], &[1]).unwrap();
        assert!(matches!(
            power(&a, &b).unwrap_err(),
            Error::InvalidArgument(_)
        ));
    }

    #[test]
    fn power_f64_invalid_is_nan() {
        let a = Array::from_slice(&[-1.0_f64], &[1]).unwrap();
        let b = Array::from_slice(&[0.5_f64], &[1]).unwrap();
        let c = power(&a, &b).unwrap();
        assert!(c.get(&[0]).unwrap().is_nan());
    }

    #[test]
    fn trunc_divide_f64() {
        let a = Array::from_slice(&[-7.0], &[1]).unwrap();
        let b = Array::from_slice(&[2.0], &[1]).unwrap();
        let c = trunc_divide(&a, &b).unwrap();
        assert_eq!(c.get(&[0]).unwrap(), -3.0);
        let d = divide(&a, &b).unwrap();
        assert_eq!(d.get(&[0]).unwrap(), -3.5);
    }

    #[test]
    fn compare_and_logical() {
        let a = Array::from_slice(&[1_i64, 2, 0], &[3]).unwrap();
        let b = Array::from_slice(&[2_i64, 2, 0], &[3]).unwrap();
        let eq = equal(&a, &b).unwrap();
        assert!(!eq.get(&[0]).unwrap());
        assert!(eq.get(&[1]).unwrap());

        let land = logical_and(&a, &b).unwrap();
        assert!(land.get(&[0]).unwrap());
        assert!(!land.get(&[2]).unwrap());
    }

    #[test]
    fn absolute_complex() {
        let a = Array::from_slice(&[Complex::new(3.0, 4.0)], &[1]).unwrap();
        let b = absolute(&a).unwrap();
        assert_eq!(b.get(&[0]).unwrap(), 5.0);
    }

    #[test]
    fn isnan_f64() {
        let a = Array::from_slice(&[1.0, f64::NAN], &[2]).unwrap();
        let m = isnan(&a).unwrap();
        assert!(!m.get(&[0]).unwrap());
        assert!(m.get(&[1]).unwrap());
    }
}
