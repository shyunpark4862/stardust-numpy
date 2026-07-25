use super::{AsBool, Complex64, Scalar};

/// Explicit scalar conversion used by [`Array::astype`](crate::Array::astype).
///
/// Unlike [`CastTo`], which only models promotion, this trait covers every
/// conversion among the four supported scalar types. Narrowing follows Rust
/// `as` semantics. Converting a complex value to a real or integer value uses
/// its real component.
pub trait ArrayCast<T: Scalar>: Scalar {
    /// Convert one array element to the requested scalar type.
    fn array_cast(self) -> T;
}

macro_rules! array_cast_as {
    ($from:ty => $to:ty) => {
        impl ArrayCast<$to> for $from {
            #[inline]
            fn array_cast(self) -> $to {
                self as $to
            }
        }
    };
}

macro_rules! array_cast_identity {
    ($type:ty) => {
        impl ArrayCast<$type> for $type {
            #[inline]
            fn array_cast(self) -> $type {
                self
            }
        }
    };
}

array_cast_identity!(bool);
array_cast_identity!(i64);
array_cast_identity!(f64);
array_cast_identity!(Complex64);

impl ArrayCast<bool> for i64 {
    #[inline]
    fn array_cast(self) -> bool {
        self.as_bool()
    }
}

impl ArrayCast<bool> for f64 {
    #[inline]
    fn array_cast(self) -> bool {
        self.as_bool()
    }
}

impl ArrayCast<bool> for Complex64 {
    #[inline]
    fn array_cast(self) -> bool {
        self.as_bool()
    }
}

array_cast_as!(bool => i64);
array_cast_as!(i64 => f64);
array_cast_as!(f64 => i64);

impl ArrayCast<f64> for bool {
    #[inline]
    fn array_cast(self) -> f64 {
        i64::from(self) as f64
    }
}

impl ArrayCast<i64> for Complex64 {
    #[inline]
    fn array_cast(self) -> i64 {
        self.re as i64
    }
}

impl ArrayCast<f64> for Complex64 {
    #[inline]
    fn array_cast(self) -> f64 {
        self.re
    }
}

impl ArrayCast<Complex64> for bool {
    #[inline]
    fn array_cast(self) -> Complex64 {
        Complex64::new(i64::from(self) as f64, 0.0)
    }
}

impl ArrayCast<Complex64> for i64 {
    #[inline]
    fn array_cast(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl ArrayCast<Complex64> for f64 {
    #[inline]
    fn array_cast(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}
