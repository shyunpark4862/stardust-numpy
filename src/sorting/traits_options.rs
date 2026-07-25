//! Sorting and unique-value type traits plus optional unique outputs.
//!
//! [`SortElement`] and [`UniqueElement`] are sealed traits that encode NumPy-
//! compatible comparison rules per dtype. [`UniqueOptions`] selects which
//! auxiliary arrays `unique_with` returns alongside sorted unique values.

use std::cmp::Ordering;

use crate::dtype::{Complex64, Scalar};
use crate::Array;

mod sealed {
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}

/// Element types supported by [`sort`] and [`argsort`].
///
/// Sealed to `bool`, `i64`, and `f64`. Complex arrays do not support general
/// sorting in this crate.
pub trait SortElement: Scalar + sealed::Sealed {
    /// Compare two values using sort ordering (NaNs last for `f64`).
    ///
    /// # Arguments
    ///
    /// * `other` — value to compare against `self`
    ///
    /// # Returns
    ///
    /// [`Ordering`] suitable for stable sort kernels.
    ///
    /// # Errors
    ///
    /// Never fails.
    fn sort_cmp(&self, other: &Self) -> Ordering;
}

impl SortElement for bool {
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl SortElement for i64 {
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl SortElement for f64 {
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (false, false) => {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
            // NumPy: non-NaN values precede NaNs.
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (true, true) => Ordering::Equal,
        }
    }
}

mod unique_sealed {
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
    impl Sealed for crate::dtype::Complex64 {}
}

/// Element types supported by [`unique`] and [`unique_with`].
///
/// Sealed to `bool`, `i64`, `f64`, and [`Complex64`].
pub trait UniqueElement: Scalar + unique_sealed::Sealed {
    /// Compare values for ordering in sorted unique output.
    ///
    /// # Arguments
    ///
    /// * `other` — value to compare against `self`
    ///
    /// # Returns
    ///
    /// [`Ordering`] for stable unique sorting; NaNs sort last for floats.
    ///
    /// # Errors
    ///
    /// Never fails.
    fn unique_cmp(&self, other: &Self) -> Ordering;

    /// Whether two values belong to the same unique group.
    ///
    /// # Arguments
    ///
    /// * `other` — value to compare against `self`
    ///
    /// # Returns
    ///
    /// `true` when both values collapse to one unique entry (all NaNs
    /// match for floating types).
    ///
    /// # Errors
    ///
    /// Never fails.
    fn unique_eq(&self, other: &Self) -> bool;
}

impl UniqueElement for bool {
    #[inline]
    fn unique_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for i64 {
    #[inline]
    fn unique_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for f64 {
    #[inline]
    fn unique_cmp(&self, other: &Self) -> Ordering {
        self.sort_cmp(other)
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        // All NaNs collapse to one unique group, like NumPy.
        self == other || (self.is_nan() && other.is_nan())
    }
}

impl UniqueElement for Complex64 {
    fn unique_cmp(&self, other: &Self) -> Ordering {
        match (complex_is_nan(*self), complex_is_nan(*other)) {
            (false, false) => self
                .re
                .sort_cmp(&other.re)
                .then_with(|| self.im.sort_cmp(&other.im)),
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (true, true) => Ordering::Equal,
        }
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other || (complex_is_nan(*self) && complex_is_nan(*other))
    }
}

/// Return whether either component of a complex value is NaN.
///
/// Used by [`UniqueElement`] for complex ordering and equality, mirroring
/// NumPy's treatment of NaN complex entries as a single group.
///
/// # Arguments
///
/// * `value` — complex scalar to inspect
///
/// # Returns
///
/// `true` when `re` or `im` is NaN.
///
/// # Errors
///
/// Never fails.
#[inline]
fn complex_is_nan(value: Complex64) -> bool {
    value.re.is_nan() || value.im.is_nan()
}

/// Flags selecting optional outputs from [`unique_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniqueOptions {
    /// Include each unique value's first flat C-order input index.
    pub return_index: bool,
    /// Include a map from each flat input element to its unique group.
    pub return_inverse: bool,
    /// Include occurrence counts for each unique value.
    pub return_counts: bool,
}

/// Result bundle returned by [`unique_with`].
#[derive(Clone, Debug)]
pub struct UniqueResult<T: Scalar> {
    /// Sorted unique values (1-D).
    pub values: Array<T>,
    /// First flat input indices, when requested.
    pub indices: Option<Array<i64>>,
    /// Unique group index for every flat input element, when requested.
    pub inverse_indices: Option<Array<i64>>,
    /// Occurrence count per unique value, when requested.
    pub counts: Option<Array<i64>>,
}
