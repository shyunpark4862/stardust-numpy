//! Public element-wise ufunc entry points (`add`, `divide`, and friends).
//!
//! Cross-type operations promote operands to a common dtype, then dispatch
//! through scalar traits. Division-like ops choose infallible IEEE kernels for
//! floats and fallible error-returning kernels for integers and bools.

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

/// Internal dispatch for element-wise division after dtype promotion.
///
/// Float and complex types use infallible IEEE kernels. Integer and bool
/// types use fallible kernels that signal divide-by-zero.
pub trait DivDispatch: Scalar {
    /// Element-wise `/` after operands are promoted to `Self`.
    ///
    /// # Arguments
    ///
    /// * `a` - Dividend array (already typed at the call site).
    /// * `b` - Divisor array.
    ///
    /// # Returns
    ///
    /// A new array with dtype `Self` and the broadcast shape of `a` and `b`.
    ///
    /// # Errors
    ///
    /// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
    ///   shapes.
    /// * [`Error::DivideByZero`](crate::error::Error::DivideByZero) — integer
    ///   or bool division by zero.
    fn divide_promoted<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

/// Internal dispatch for truncating division after dtype promotion.
pub trait TruncDivDispatch: Scalar {
    /// Element-wise truncating division after promotion.
    ///
    /// # Arguments
    ///
    /// * `a` - Dividend array.
    /// * `b` - Divisor array.
    ///
    /// # Returns
    ///
    /// A new array with dtype `Self` and the broadcast shape of `a` and `b`.
    ///
    /// # Errors
    ///
    /// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
    ///   shapes.
    /// * [`Error::DivideByZero`](crate::error::Error::DivideByZero) — integer
    ///   or bool division by zero.
    fn trunc_divide_promoted<L, R>(
        a: &Array<L>,
        b: &Array<R>,
    ) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

/// Internal dispatch for element-wise remainder after dtype promotion.
pub trait RemDispatch: Scalar {
    /// Element-wise `%` after promotion.
    ///
    /// # Arguments
    ///
    /// * `a` - Dividend array.
    /// * `b` - Divisor array.
    ///
    /// # Returns
    ///
    /// A new array with dtype `Self` and the broadcast shape of `a` and `b`.
    ///
    /// # Errors
    ///
    /// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
    ///   shapes.
    /// * [`Error::DivideByZero`](crate::error::Error::DivideByZero) — integer
    ///   or bool remainder by zero.
    fn remainder_promoted<L, R>(
        a: &Array<L>,
        b: &Array<R>,
    ) -> Result<Array<Self>>
    where
        L: Promote<R, Output = Self> + CastTo<Self>,
        R: CastTo<Self>,
        Self: Sized;
}

/// Internal dispatch for element-wise power after dtype promotion.
pub trait PowDispatch: Scalar {
    /// Element-wise power after promotion.
    ///
    /// # Arguments
    ///
    /// * `a` - Base array.
    /// * `b` - Exponent array.
    ///
    /// # Returns
    ///
    /// A new array with dtype `Self` and the broadcast shape of `a` and `b`.
    ///
    /// # Errors
    ///
    /// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
    ///   shapes.
    /// * [`Error::InvalidArgument`](crate::error::Error::InvalidArgument) —
    ///   invalid integer exponent (see [`power`]).
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

/// Element-wise addition (`+`) of two arrays.
///
/// Operand dtypes are unified via the [`Promote`] trait: both inputs are cast
/// to `L::Output` before [`ElemAdd`] is applied. Shapes are broadcast to a
/// common result shape under NumPy rules.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{add, Array};
///
/// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[10_i64], &[1]).unwrap();
/// let sum = add(&a, &b).unwrap();
/// assert_eq!(sum.get(&[1, 1]).unwrap(), 14);
/// ```
pub fn add<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: ElemAdd,
{
    promote_binary(a, b, ElemAdd::elem_add)
}

/// Element-wise subtraction (`-`) of two arrays.
///
/// Operand dtypes are unified via the [`Promote`] trait: both inputs are cast
/// to `L::Output` before [`ElemSub`] is applied. Shapes are broadcast to a
/// common result shape under NumPy rules.
///
/// # Arguments
///
/// * `a` - Minuend array.
/// * `b` - Subtrahend array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{subtract, Array};
///
/// let a = Array::from_slice(&[10_i64, 20, 30, 40], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// let diff = subtract(&a, &b).unwrap();
/// assert_eq!(diff.get(&[0, 1]).unwrap(), 18);
/// ```
pub fn subtract<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: ElemSub,
{
    promote_binary(a, b, ElemSub::elem_sub)
}

/// Element-wise multiplication (`*`) of two arrays.
///
/// Operand dtypes are unified via the [`Promote`] trait: both inputs are cast
/// to `L::Output` before [`ElemMul`] is applied. Shapes are broadcast to a
/// common result shape under NumPy rules.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{multiply, Array};
///
/// let a = Array::from_slice(&[2_i64, 3, 4, 5], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[10_i64], &[1]).unwrap();
/// let prod = multiply(&a, &b).unwrap();
/// assert_eq!(prod.get(&[1, 0]).unwrap(), 40);
/// ```
pub fn multiply<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: ElemMul,
{
    promote_binary(a, b, ElemMul::elem_mul)
}

/// Element-wise true division (`/`) of two arrays.
///
/// Operand dtypes are unified via the [`Promote`] trait before
/// [`DivDispatch`] selects the kernel. Shapes are broadcast under NumPy
/// rules. Floating and complex division follows IEEE semantics (`inf` /
/// `nan` on divide-by-zero). Integer and bool division by zero returns
/// [`Error::DivideByZero`](crate::error::Error::DivideByZero).
///
/// # Arguments
///
/// * `a` - Dividend array.
/// * `b` - Divisor array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
///   shapes.
/// * [`Error::DivideByZero`](crate::error::Error::DivideByZero) — integer or
///   bool division by zero.
///
/// # Examples
///
/// ```rust
/// use sdnp::{divide, Array};
///
/// let a = Array::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[2.0_f64], &[1]).unwrap();
/// let quot = divide(&a, &b).unwrap();
/// assert_eq!(quot.get(&[0, 0]).unwrap(), 0.5);
/// ```
pub fn divide<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: DivDispatch,
{
    L::Output::divide_promoted(a, b)
}

/// Element-wise truncating division toward zero.
///
/// Operand dtypes are unified via the [`Promote`] trait before
/// [`TruncDivDispatch`] runs. Shapes are broadcast under NumPy rules. For
/// `i64`, this matches [`divide`]. For `f64`, each quotient is truncated
/// toward zero (`(a / b).trunc()`). Integer and bool division by zero
/// returns [`Error::DivideByZero`](crate::error::Error::DivideByZero).
///
/// # Arguments
///
/// * `a` - Dividend array.
/// * `b` - Divisor array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
///   shapes.
/// * [`Error::DivideByZero`](crate::error::Error::DivideByZero) — integer or
///   bool division by zero.
///
/// # Examples
///
/// ```rust
/// use sdnp::{trunc_divide, Array};
///
/// let a = Array::from_slice(&[7.0_f64, -7.0], &[2]).unwrap();
/// let b = Array::from_slice(&[2.0_f64, 2.0], &[2]).unwrap();
/// let q = trunc_divide(&a, &b).unwrap();
/// assert_eq!(q.get(&[0]).unwrap(), 3.0);
/// assert_eq!(q.get(&[1]).unwrap(), -3.0);
/// ```
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

/// Element-wise remainder (`%`) of two arrays.
///
/// Operand dtypes are unified via the [`Promote`] trait before
/// [`RemDispatch`] runs. Shapes are broadcast under NumPy rules. Floating
/// remainder follows IEEE rules. Integer and bool remainder by zero returns
/// [`Error::DivideByZero`](crate::error::Error::DivideByZero).
///
/// # Arguments
///
/// * `a` - Dividend array.
/// * `b` - Divisor array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
///   shapes.
/// * [`Error::DivideByZero`](crate::error::Error::DivideByZero) — integer or
///   bool remainder by zero.
///
/// # Examples
///
/// ```rust
/// use sdnp::{remainder, Array};
///
/// let a = Array::from_slice(&[7_i64, 8, 9, 10], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[3_i64], &[1]).unwrap();
/// let rem = remainder(&a, &b).unwrap();
/// assert_eq!(rem.get(&[0, 0]).unwrap(), 1);
/// ```
pub fn remainder<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: RemDispatch,
{
    L::Output::remainder_promoted(a, b)
}

/// Element-wise exponentiation (`**`) of two arrays.
///
/// Operand dtypes are unified via the [`Promote`] trait before
/// [`PowDispatch`] runs. Shapes are broadcast under NumPy rules. Floating
/// exponents follow IEEE-style rules. Integer exponents must lie in
/// `0..=u32::MAX`; negative integer exponents return
/// [`Error::InvalidArgument`](crate::error::Error::InvalidArgument).
///
/// # Arguments
///
/// * `a` - Base array.
/// * `b` - Exponent array.
///
/// # Returns
///
/// A new owned array with dtype `L::Output` and the broadcast shape.
///
/// # Errors
///
/// * [`Error::Broadcast`](crate::error::Error::Broadcast) — incompatible
///   shapes.
/// * [`Error::InvalidArgument`](crate::error::Error::InvalidArgument) —
///   invalid integer exponent.
///
/// # Examples
///
/// ```rust
/// use sdnp::{power, Array};
///
/// let bases = Array::from_slice(&[2_i64, 3, 4, 5], &[2, 2]).unwrap();
/// let exps = Array::from_slice(&[2_i64], &[1]).unwrap();
/// let out = power(&bases, &exps).unwrap();
/// assert_eq!(out.get(&[0, 0]).unwrap(), 4);
/// ```
pub fn power<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<L::Output>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PowDispatch,
{
    L::Output::power_promoted(a, b)
}

/// Element-wise unary negation (`-`) of an array.
///
/// Applies [`ElemNeg`] to every element. The output shape and dtype match the
/// input.
///
/// # Arguments
///
/// * `a` - Input array.
///
/// # Returns
///
/// A new owned array with the same shape and dtype as `a`.
///
/// # Errors
///
/// Never fails for supported scalar types.
///
/// # Examples
///
/// ```rust
/// use sdnp::{negative, Array};
///
/// let a = Array::from_slice(&[1_i64, -2, 3], &[3]).unwrap();
/// let b = negative(&a).unwrap();
/// assert_eq!(b.get(&[1]).unwrap(), 2);
/// ```
pub fn negative<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: ElemNeg,
{
    map_unary(a, ElemNeg::elem_neg)
}

/// Element-wise absolute value or complex magnitude.
///
/// Real types return `|x|` via [`ElemAbs`]. [`Complex64`] elements yield an
/// `f64` magnitude array. The output shape matches the input.
///
/// # Arguments
///
/// * `a` - Input array.
///
/// # Returns
///
/// A new owned array with dtype `T::Output` and the same shape as `a`.
///
/// # Errors
///
/// Never fails for supported scalar types.
///
/// # Examples
///
/// ```rust
/// use sdnp::{absolute, Array};
///
/// let a = Array::from_slice(&[-3_i64, 4, -5], &[3]).unwrap();
/// let b = absolute(&a).unwrap();
/// assert_eq!(b.get(&[0]).unwrap(), 3);
/// ```
pub fn absolute<T>(a: &Array<T>) -> Result<Array<T::Output>>
where
    T: ElemAbs,
{
    map_unary(a, ElemAbs::elem_abs)
}

/// Element-wise equality (`==`) comparison.
///
/// Operand dtypes are unified via the [`Promote`] trait before comparison.
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`, not the promoted arithmetic type.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{equal, Array};
///
/// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[1_i64, 0, 3, 0], &[2, 2]).unwrap();
/// let mask = equal(&a, &b).unwrap();
/// assert!(mask.get(&[0, 0]).unwrap());
/// assert!(!mask.get(&[0, 1]).unwrap());
/// ```
pub fn equal<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialEq,
{
    promote_binary(a, b, |x, y| x == y)
}

/// Element-wise inequality (`!=`) comparison.
///
/// Operand dtypes are unified via the [`Promote`] trait before comparison.
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`, not the promoted arithmetic type.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{not_equal, Array};
///
/// let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
/// let b = Array::from_slice(&[1_i64, 0], &[2]).unwrap();
/// let mask = not_equal(&a, &b).unwrap();
/// assert!(!mask.get(&[0]).unwrap());
/// assert!(mask.get(&[1]).unwrap());
/// ```
pub fn not_equal<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialEq,
{
    promote_binary(a, b, |x, y| x != y)
}

/// Element-wise less-than (`<`) comparison.
///
/// Operand dtypes are unified via the [`Promote`] trait before comparison.
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{less, Array};
///
/// let a = Array::from_slice(&[1_i64, 3], &[2]).unwrap();
/// let b = Array::from_slice(&[2_i64, 2], &[2]).unwrap();
/// let mask = less(&a, &b).unwrap();
/// assert!(mask.get(&[0]).unwrap());
/// assert!(!mask.get(&[1]).unwrap());
/// ```
pub fn less<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialOrd,
{
    promote_binary(a, b, |x, y| x.lt(&y))
}

/// Element-wise less-than-or-equal (`<=`) comparison.
///
/// Operand dtypes are unified via the [`Promote`] trait before comparison.
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{less_equal, Array};
///
/// let a = Array::from_slice(&[2_i64, 3], &[2]).unwrap();
/// let b = Array::from_slice(&[2_i64, 2], &[2]).unwrap();
/// let mask = less_equal(&a, &b).unwrap();
/// assert!(mask.get(&[0]).unwrap());
/// assert!(!mask.get(&[1]).unwrap());
/// ```
pub fn less_equal<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialOrd,
{
    promote_binary(a, b, |x, y| x.le(&y))
}

/// Element-wise greater-than (`>`) comparison.
///
/// Operand dtypes are unified via the [`Promote`] trait before comparison.
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{greater, Array};
///
/// let a = Array::from_slice(&[3_i64, 1], &[2]).unwrap();
/// let b = Array::from_slice(&[2_i64, 2], &[2]).unwrap();
/// let mask = greater(&a, &b).unwrap();
/// assert!(mask.get(&[0]).unwrap());
/// assert!(!mask.get(&[1]).unwrap());
/// ```
pub fn greater<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialOrd,
{
    promote_binary(a, b, |x, y| x.gt(&y))
}

/// Element-wise greater-than-or-equal (`>=`) comparison.
///
/// Operand dtypes are unified via the [`Promote`] trait before comparison.
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{greater_equal, Array};
///
/// let a = Array::from_slice(&[2_i64, 1], &[2]).unwrap();
/// let b = Array::from_slice(&[2_i64, 2], &[2]).unwrap();
/// let mask = greater_equal(&a, &b).unwrap();
/// assert!(mask.get(&[0]).unwrap());
/// assert!(!mask.get(&[1]).unwrap());
/// ```
pub fn greater_equal<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: Promote<R> + CastTo<L::Output>,
    R: CastTo<L::Output>,
    L::Output: PartialOrd,
{
    promote_binary(a, b, |x, y| x.ge(&y))
}

/// Element-wise logical AND using NumPy-style truthiness.
///
/// Each element is coerced with [`AsBool`] (zero is false, non-zero is true).
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{logical_and, Array};
///
/// let a = Array::from_slice(&[1_i64, 0, 1, 0], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[1_i64, 1, 0, 0], &[2, 2]).unwrap();
/// let out = logical_and(&a, &b).unwrap();
/// assert!(out.get(&[0, 0]).unwrap());
/// assert!(!out.get(&[0, 1]).unwrap());
/// ```
pub fn logical_and<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: AsBool,
    R: AsBool,
{
    map_binary(a, b, |x, y| x.as_bool() & y.as_bool())
}

/// Element-wise logical OR using NumPy-style truthiness.
///
/// Each element is coerced with [`AsBool`] (zero is false, non-zero is true).
/// Shapes are broadcast under NumPy rules. The result dtype is always
/// `bool`.
///
/// # Arguments
///
/// * `a` - Left-hand operand array.
/// * `b` - Right-hand operand array.
///
/// # Returns
///
/// A new `Array<bool>` with the broadcast shape.
///
/// # Errors
///
/// Returns [`Error::Broadcast`](crate::error::Error::Broadcast) when shapes
/// cannot be aligned.
///
/// # Examples
///
/// ```rust
/// use sdnp::{logical_or, Array};
///
/// let a = Array::from_slice(&[1_i64, 0, 0, 0], &[2, 2]).unwrap();
/// let b = Array::from_slice(&[0_i64, 1, 0, 0], &[2, 2]).unwrap();
/// let out = logical_or(&a, &b).unwrap();
/// assert!(out.get(&[0, 0]).unwrap());
/// assert!(out.get(&[0, 1]).unwrap());
/// ```
pub fn logical_or<L, R>(a: &Array<L>, b: &Array<R>) -> Result<Array<bool>>
where
    L: AsBool,
    R: AsBool,
{
    map_binary(a, b, |x, y| x.as_bool() | y.as_bool())
}

/// Element-wise logical NOT using NumPy-style truthiness.
///
/// Each element is coerced with [`AsBool`]. The output shape matches the
/// input and the result dtype is always `bool`.
///
/// # Arguments
///
/// * `a` - Input array.
///
/// # Returns
///
/// A new `Array<bool>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails for types implementing [`AsBool`].
///
/// # Examples
///
/// ```rust
/// use sdnp::{logical_not, Array};
///
/// let a = Array::from_slice(&[1_i64, 0, 2, 0], &[2, 2]).unwrap();
/// let out = logical_not(&a).unwrap();
/// assert!(!out.get(&[0, 0]).unwrap());
/// assert!(out.get(&[0, 1]).unwrap());
/// ```
pub fn logical_not<T: AsBool>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, |x| !x.as_bool())
}

/// Element-wise `isnan` test for floating and complex types.
///
/// # Arguments
///
/// * `a` - Input array of floats or complex values.
///
/// # Returns
///
/// A new `Array<bool>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails for supported floating types.
///
/// # Examples
///
/// ```rust
/// use sdnp::{isnan, Array};
///
/// let a = Array::from_slice(&[1.0_f64, f64::NAN], &[2]).unwrap();
/// let mask = isnan(&a).unwrap();
/// assert!(!mask.get(&[0]).unwrap());
/// assert!(mask.get(&[1]).unwrap());
/// ```
pub fn isnan<T: FloatClassify>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, FloatClassify::is_nan)
}

/// Element-wise `isinf` test for floating and complex types.
///
/// # Arguments
///
/// * `a` - Input array of floats or complex values.
///
/// # Returns
///
/// A new `Array<bool>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails for supported floating types.
///
/// # Examples
///
/// ```rust
/// use sdnp::{isinf, Array};
///
/// let a = Array::from_slice(&[1.0_f64, f64::INFINITY], &[2]).unwrap();
/// let mask = isinf(&a).unwrap();
/// assert!(!mask.get(&[0]).unwrap());
/// assert!(mask.get(&[1]).unwrap());
/// ```
pub fn isinf<T: FloatClassify>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, FloatClassify::is_infinite)
}

/// Element-wise `isfinite` test for floating and complex types.
///
/// # Arguments
///
/// * `a` - Input array of floats or complex values.
///
/// # Returns
///
/// A new `Array<bool>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails for supported floating types.
///
/// # Examples
///
/// ```rust
/// use sdnp::{isfinite, Array};
///
/// let a = Array::from_slice(&[1.0_f64, f64::INFINITY], &[2]).unwrap();
/// let mask = isfinite(&a).unwrap();
/// assert!(mask.get(&[0]).unwrap());
/// assert!(!mask.get(&[1]).unwrap());
/// ```
pub fn isfinite<T: FloatClassify>(a: &Array<T>) -> Result<Array<bool>> {
    map_unary(a, FloatClassify::is_finite)
}

/// Complex conjugate of each element.
///
/// # Arguments
///
/// * `a` - Input complex array.
///
/// # Returns
///
/// A new `Array<Complex64>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails.
///
/// # Examples
///
/// ```rust
/// use sdnp::{conj, Array, Complex64};
///
/// let z = Complex64::new(1.0, 2.0);
/// let a = Array::from_slice(&[z], &[1]).unwrap();
/// let c = conj(&a).unwrap();
/// assert_eq!(c.get(&[0]).unwrap().im, -2.0);
/// ```
pub fn conj(a: &Array<Complex64>) -> Result<Array<Complex64>> {
    map_unary(a, |z| z.conj())
}

/// Real part of each complex element.
///
/// # Arguments
///
/// * `a` - Input complex array.
///
/// # Returns
///
/// A new `Array<f64>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails.
///
/// # Examples
///
/// ```rust
/// use sdnp::{real, Array, Complex64};
///
/// let z = Complex64::new(3.0, 4.0);
/// let a = Array::from_slice(&[z], &[1]).unwrap();
/// let r = real(&a).unwrap();
/// assert_eq!(r.get(&[0]).unwrap(), 3.0);
/// ```
pub fn real(a: &Array<Complex64>) -> Result<Array<f64>> {
    map_unary(a, |z| z.re)
}

/// Imaginary part of each complex element.
///
/// # Arguments
///
/// * `a` - Input complex array.
///
/// # Returns
///
/// A new `Array<f64>` with the same shape as `a`.
///
/// # Errors
///
/// Never fails.
///
/// # Examples
///
/// ```rust
/// use sdnp::{imag, Array, Complex64};
///
/// let z = Complex64::new(3.0, 4.0);
/// let a = Array::from_slice(&[z], &[1]).unwrap();
/// let im = imag(&a).unwrap();
/// assert_eq!(im.get(&[0]).unwrap(), 4.0);
/// ```
pub fn imag(a: &Array<Complex64>) -> Result<Array<f64>> {
    map_unary(a, |z| z.im)
}
