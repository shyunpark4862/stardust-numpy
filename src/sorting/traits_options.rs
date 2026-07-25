use std::cmp::Ordering;

use crate::dtype::{Complex64, Scalar};
use crate::Array;

mod sealed {
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}

/// Element types accepted by [`sort`] and [`argsort`].
///
/// This trait is sealed; the supported types are `bool`, `i64`, and `f64`.
/// In particular, complex arrays intentionally do not implement general
/// sorting.
pub trait SortElement: Scalar + sealed::Sealed {
    /// Compare two values using the ordering required by sorting operations.
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

/// Element types accepted by [`unique`] and [`unique_with`].
///
/// This trait is sealed; the supported types are `bool`, `i64`, `f64`, and
/// [`Complex64`].
pub trait UniqueElement: Scalar + unique_sealed::Sealed {
    /// Compare values for their position in sorted unique output.
    fn unique_cmp(&self, other: &Self) -> Ordering;

    /// Return whether values belong to the same unique-value group.
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

#[inline]
fn complex_is_nan(value: Complex64) -> bool {
    value.re.is_nan() || value.im.is_nan()
}

/// Optional outputs requested from [`unique_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniqueOptions {
    /// Include each unique value's first flat input index.
    pub return_index: bool,
    /// Include a map from each flat input element to its unique value.
    pub return_inverse: bool,
    /// Include the number of occurrences of each unique value.
    pub return_counts: bool,
}

/// Result of [`unique_with`].
#[derive(Clone, Debug)]
pub struct UniqueResult<T: Scalar> {
    /// Sorted unique values.
    pub values: Array<T>,
    /// First C-order input indices, when requested.
    pub indices: Option<Array<i64>>,
    /// Unique-value index for every C-order input element, when requested.
    pub inverse_indices: Option<Array<i64>>,
    /// Occurrence count for every unique value, when requested.
    pub counts: Option<Array<i64>>,
}
