use num_complex::Complex;

/// Complex element type used by this crate.
pub type Complex64 = Complex<f64>;

/// Marker for allowed `Array` element types.
pub trait Scalar: Copy + Send + Sync + 'static {}

impl Scalar for bool {}
impl Scalar for i64 {}
impl Scalar for f64 {}
impl Scalar for Complex64 {}

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
