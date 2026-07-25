//! Scalar element types and numeric promotion.
//!
//! Promotion hierarchy (NumPy-like, fixed-width):
//! `bool < i64 < f64 < Complex<f64>`.

mod cast;
mod promotion;
mod scalar;

pub use cast::ArrayCast;
pub use promotion::{CastTo, Promote};
pub use scalar::{AsBool, Complex64, Scalar};

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

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

    #[test]
    fn explicit_array_casts_cover_narrowing_and_complex_values() {
        assert_eq!(ArrayCast::<i64>::array_cast(3.9_f64), 3);
        assert_eq!(ArrayCast::<i64>::array_cast(f64::NAN), 0);
        assert_eq!(ArrayCast::<i64>::array_cast(f64::INFINITY), i64::MAX);
        assert_eq!(ArrayCast::<f64>::array_cast(Complex64::new(2.5, 9.0)), 2.5);
        assert_eq!(ArrayCast::<i64>::array_cast(Complex64::new(-2.5, 9.0)), -2);
        assert!(ArrayCast::<bool>::array_cast(Complex64::new(0.0, 1.0)));
    }
}
