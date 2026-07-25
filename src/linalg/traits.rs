//! Element arithmetic used by contraction and dot-product kernels.
//!
//! After dtype promotion, both operands share an output type that must
//! support zero, addition, fused multiply-add, and complex conjugation.
//! Real, integer, boolean, and complex types each map these operations to
//! their natural semantics.

use crate::dtype::{Complex64, Scalar};

/// Promoted scalar algebra required by matrix and vector contractions.
///
/// After dtype promotion, both operands of [`crate::dot`], [`crate::matmul`],
/// [`crate::vdot`], and [`crate::outer`] share a type implementing this
/// trait. Real and integer types use ordinary arithmetic; complex types
/// conjugate in [`Self::conjugate`] for inner products.
///
/// # Examples
///
/// ```
/// use sdnp::ContractElement;
/// use sdnp::Complex64;
///
/// let z = Complex64::new(1.0, 2.0);
/// assert_eq!(z.conjugate(), Complex64::new(1.0, -2.0));
/// assert_eq!(f64::conjugate(3.0), 3.0);
/// ```
pub trait ContractElement: Scalar {
    /// Additive identity for an empty inner product.
    ///
    /// # Arguments
    ///
    /// None — this is a type-level constant query.
    ///
    /// # Returns
    ///
    /// The neutral element for [`Self::add`] (e.g. `0`, `0.0`, `false`).
    fn zero() -> Self;

    /// Add two partial accumulators from independent contraction chains.
    ///
    /// # Arguments
    ///
    /// * `accumulator` — running partial sum
    /// * `value` — value to combine
    ///
    /// # Returns
    ///
    /// Combined accumulator (wrapping for integers, logical OR for bool).
    fn add(accumulator: Self, value: Self) -> Self;

    /// Fused `accumulator + left * right`.
    ///
    /// # Arguments
    ///
    /// * `accumulator` — running partial sum
    /// * `left`, `right` — factors along the contraction axis
    ///
    /// # Returns
    ///
    /// Updated accumulator after one multiply-add step.
    fn multiply_add(accumulator: Self, left: Self, right: Self) -> Self;

    /// Complex conjugate; identity for real and integer types.
    ///
    /// Used by [`crate::vdot`] when conjugating the left operand.
    ///
    /// # Arguments
    ///
    /// * `self` - Scalar factor from the left operand of an inner product.
    ///
    /// # Returns
    ///
    /// Conjugated value (unchanged for real/integer/bool types).
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

impl_real_contract!(f64, 0.0_f64);

impl ContractElement for i64 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn add(accumulator: Self, value: Self) -> Self {
        accumulator.wrapping_add(value)
    }

    #[inline]
    fn multiply_add(accumulator: Self, left: Self, right: Self) -> Self {
        accumulator.wrapping_add(left.wrapping_mul(right))
    }

    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

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
