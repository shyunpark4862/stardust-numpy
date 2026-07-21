//! Scalar element types and numeric promotion.
//!
//! Promotion hierarchy (NumPy-like, fixed-width):
//! `bool < i64 < f64 < Complex<f64>`.

use num_complex::Complex;

/// Complex element type used by this crate.
pub type Complex64 = Complex<f64>;

/// Marker for allowed `Array` element types.
pub trait Scalar: Copy + Send + Sync + 'static {}

impl Scalar for bool {}
impl Scalar for i64 {}
impl Scalar for f64 {}
impl Scalar for Complex64 {}

/// Type-level promotion: `Self` combined with `Rhs` yields [`Promote::Output`].
pub trait Promote<Rhs: Scalar>: Scalar {
    /// Resulting scalar type after promotion.
    type Output: Scalar;
}

macro_rules! promote_impl {
    ($left:ty, $right:ty, $out:ty) => {
        impl Promote<$right> for $left {
            type Output = $out;
        }
    };
}

// Same-type
promote_impl!(bool, bool, bool);
promote_impl!(i64, i64, i64);
promote_impl!(f64, f64, f64);
promote_impl!(Complex64, Complex64, Complex64);

// bool promotes toward wider numeric types
promote_impl!(bool, i64, i64);
promote_impl!(i64, bool, i64);
promote_impl!(bool, f64, f64);
promote_impl!(f64, bool, f64);
promote_impl!(bool, Complex64, Complex64);
promote_impl!(Complex64, bool, Complex64);

// i64 ↔ f64 / complex
promote_impl!(i64, f64, f64);
promote_impl!(f64, i64, f64);
promote_impl!(i64, Complex64, Complex64);
promote_impl!(Complex64, i64, Complex64);

// f64 ↔ complex
promote_impl!(f64, Complex64, Complex64);
promote_impl!(Complex64, f64, Complex64);

/// Cast a scalar into a wider (or same) promoted type.
pub trait CastTo<T: Scalar>: Scalar {
    /// Convert `self` into `T`.
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

/// Truthiness for logical ufuncs (`logical_and`, …).
///
/// Distinct from [`CastTo`]: this answers “is this value true in a
/// boolean context?” (non-zero), not numeric promotion.
pub trait AsBool: Scalar {
    /// Convert to a boolean predicate value.
    fn as_bool(self) -> bool;
}

impl AsBool for bool {
    #[inline]
    fn as_bool(self) -> bool {
        self
    }
}

impl AsBool for i64 {
    #[inline]
    fn as_bool(self) -> bool {
        self != 0
    }
}

impl AsBool for f64 {
    #[inline]
    fn as_bool(self) -> bool {
        self != 0.0
    }
}

impl AsBool for Complex64 {
    #[inline]
    fn as_bool(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promote_hierarchy() {
        fn out<L: Promote<R>, R: Scalar>() -> &'static str {
            std::any::type_name::<L::Output>()
        }

        assert!(out::<bool, i64>().contains("i64"));
        assert!(out::<i64, f64>().contains("f64"));
        assert!(out::<f64, Complex64>().contains("Complex"));
    }

    #[test]
    fn cast_bool_to_i64() {
        assert_eq!(CastTo::<i64>::cast_to(true), 1_i64);
        assert_eq!(CastTo::<i64>::cast_to(false), 0_i64);
    }

    #[test]
    fn as_bool_truthiness() {
        assert!(AsBool::as_bool(2_i64));
        assert!(!AsBool::as_bool(0_i64));
        assert!(AsBool::as_bool(-1.0));
        assert!(!AsBool::as_bool(Complex::new(0.0, 0.0)));
        assert!(AsBool::as_bool(Complex::new(0.0, 1.0)));
    }
}
