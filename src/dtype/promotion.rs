use num_complex::Complex;

use super::{Complex64, Scalar};

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
