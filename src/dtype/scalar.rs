//! Marker traits and type aliases for supported array element types.
//!
//! Only four scalar kinds are supported in this educational build: boolean,
//! 64-bit integer, 64-bit float, and complex float. The [`Scalar`] marker
//! gates generic impls; [`AsBool`] models truthiness for logical ufuncs.

use num_complex::Complex;

/// Complex element type (`Complex<f64>`) used throughout the crate.
pub type Complex64 = Complex<f64>;

/// Marker trait for types that may be stored in an [`Array`](crate::Array).
///
/// Implemented only for `bool`, `i64`, `f64`, and [`Complex64`]. Generic
/// array code requires `T: Scalar` so unsupported element types are rejected
/// at compile time.
pub trait Scalar: Copy + Send + Sync + 'static {}

impl Scalar for bool {}
impl Scalar for i64 {}
impl Scalar for f64 {}
impl Scalar for Complex64 {}

/// Truthiness conversion for logical ufuncs such as `logical_and`.
///
/// Distinct from [`CastTo`](super::CastTo): this answers whether a value
/// is truthy in a boolean context (non-zero), not which type to promote to.
pub trait AsBool: Scalar {
    /// Return the boolean predicate for this scalar value.
    ///
    /// # Arguments
    ///
    /// None — only `self` is inspected.
    ///
    /// # Returns
    ///
    /// `true` when the value is truthy in NumPy-style logical operations.
    ///
    /// # Errors
    ///
    /// Never fails.
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
