//! Compile-time numeric promotion and widening casts between scalar types.
//!
//! [`Promote`] records the result type when two scalars meet in a binary
//! ufunc. [`CastTo`] performs the actual value conversion along the
//! promotion ladder. Narrowing is intentionally **not** modeled here; see
//! [`ArrayCast`](super::ArrayCast) for full dtype conversion.

use num_complex::Complex;

use super::{Complex64, Scalar};

/// Type-level promotion: combining `Self` with `Rhs` yields
/// [`Output`](Promote::Output).
///
/// Used by binary ufuncs to pick a common result dtype before any element
/// is read. The ladder is fixed at compile time via macro-generated impls.
pub trait Promote<Rhs: Scalar>: Scalar {
    /// Scalar type produced after promoting both operands.
    type Output: Scalar;
}

macro_rules! promote_impl {
    ($left:ty, $right:ty, $out:ty) => {
        impl Promote<$right> for $left {
            type Output = $out;
        }
    };
}

// Identity: same-type pairs promote to themselves.
promote_impl!(bool, bool, bool);
promote_impl!(i64, i64, i64);
promote_impl!(f64, f64, f64);
promote_impl!(Complex64, Complex64, Complex64);

// bool widens toward numeric types.
promote_impl!(bool, i64, i64);
promote_impl!(i64, bool, i64);
promote_impl!(bool, f64, f64);
promote_impl!(f64, bool, f64);
promote_impl!(bool, Complex64, Complex64);
promote_impl!(Complex64, bool, Complex64);

// Integer and float meet at f64; both widen to complex.
promote_impl!(i64, f64, f64);
promote_impl!(f64, i64, f64);
promote_impl!(i64, Complex64, Complex64);
promote_impl!(Complex64, i64, Complex64);

// Real and complex meet at Complex64.
promote_impl!(f64, Complex64, Complex64);
promote_impl!(Complex64, f64, Complex64);

/// Widening cast from `self` into promoted type `T`.
///
/// Only conversions along the promotion ladder are provided. Narrowing
/// (e.g. `f64` → `i64`) is handled by [`ArrayCast`](super::ArrayCast).
pub trait CastTo<T: Scalar>: Scalar {
    /// Convert this value into the target promoted type.
    ///
    /// # Arguments
    ///
    /// None — only `self` is converted.
    ///
    /// # Returns
    ///
    /// The widened scalar in type `T`.
    ///
    /// # Errors
    ///
    /// Never fails; unsupported conversions are rejected at compile time.
    fn cast_to(self) -> T;
}

impl<T: Scalar> CastTo<T> for T {
    #[inline]
    fn cast_to(self) -> T {
        self
    }
}

impl CastTo<i64> for bool {
    #[inline]
    fn cast_to(self) -> i64 {
        i64::from(self)
    }
}

impl CastTo<f64> for bool {
    #[inline]
    fn cast_to(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
}

impl CastTo<Complex64> for bool {
    #[inline]
    fn cast_to(self) -> Complex64 {
        Complex::new(self.cast_to(), 0.0)
    }
}

impl CastTo<f64> for i64 {
    #[inline]
    fn cast_to(self) -> f64 {
        self as f64
    }
}

impl CastTo<Complex64> for i64 {
    #[inline]
    fn cast_to(self) -> Complex64 {
        Complex::new(self as f64, 0.0)
    }
}

impl CastTo<Complex64> for f64 {
    #[inline]
    fn cast_to(self) -> Complex64 {
        Complex::new(self, 0.0)
    }
}
