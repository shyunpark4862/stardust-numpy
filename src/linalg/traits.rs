//! Scalar protocols used by contraction kernels.

use crate::dtype::{Complex64, Scalar};

/// Arithmetic required after both contraction operands have been promoted.
pub trait ContractElement: Scalar {
    /// Additive identity for an empty contraction.
    fn zero() -> Self;

    /// Combine two independently accumulated partial sums.
    fn add(accumulator: Self, value: Self) -> Self;

    /// Accumulate `left * right` into `accumulator`.
    fn multiply_add(accumulator: Self, left: Self, right: Self) -> Self;

    /// Complex conjugate; a no-op for real element types.
    fn conjugate(self) -> Self;
}

macro_rules! impl_real_contract {
    ($type:ty, $zero:expr) => {
        impl ContractElement for $type {
            #[inline]
            fn zero() -> Self {
                $zero
            }

            #[inline]
            fn add(accumulator: Self, value: Self) -> Self {
                accumulator + value
            }

            #[inline]
            fn multiply_add(
                accumulator: Self,
                left: Self,
                right: Self,
            ) -> Self {
                accumulator + left * right
            }

            #[inline]
            fn conjugate(self) -> Self {
                self
            }
        }
    };
}

impl_real_contract!(i64, 0_i64);
impl_real_contract!(f64, 0.0_f64);

impl ContractElement for bool {
    #[inline]
    fn zero() -> Self {
        false
    }

    #[inline]
    fn add(accumulator: Self, value: Self) -> Self {
        accumulator || value
    }

    #[inline]
    fn multiply_add(accumulator: Self, left: Self, right: Self) -> Self {
        accumulator || (left && right)
    }

    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

impl ContractElement for Complex64 {
    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn add(accumulator: Self, value: Self) -> Self {
        accumulator + value
    }

    #[inline]
    fn multiply_add(accumulator: Self, left: Self, right: Self) -> Self {
        accumulator + left * right
    }

    #[inline]
    fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }
}
