//! Normalized indexing specifications (`IndexSpec`).
//!
//! Python/`__getitem__` is expected to parse user objects into these variants.
//! Ellipsis is supported for Rust ergonomics; trailing missing axes are also
//! padded with full slices during preparation.

use crate::array::Array;

/// One entry in a normalized index expression.
#[derive(Clone, Debug)]
pub enum IndexSpec {
    /// Integer index along one source axis (may be negative until resolved).
    Index(i64),
    /// Python-style slice; omitted bounds use axis defaults; `step` defaults to 1.
    Slice {
        /// Start (inclusive); `None` means default for the step direction.
        start: Option<i64>,
        /// Stop (exclusive); `None` means default for the step direction.
        stop: Option<i64>,
        /// Step; `None` means `1`. Zero is an error.
        step: Option<i64>,
    },
    /// Insert a length-1 axis (NumPy `newaxis` / `None`).
    NewAxis,
    /// Fill remaining source axes with full slices (at most one per index).
    Ellipsis,
    /// Boolean array mask; consumes `ndim` consecutive source axes.
    BoolArray(Array<bool>),
    /// Integer fancy index for one source axis.
    IntegerArray(Array<i64>),
}

impl IndexSpec {
    /// Full axis (`:`).
    #[inline]
    pub fn full() -> Self {
        Self::Slice {
            start: None,
            stop: None,
            step: Some(1),
        }
    }

    /// Integer index.
    #[inline]
    pub fn index(i: i64) -> Self {
        Self::Index(i)
    }

    /// Slice with optional bounds/step.
    #[inline]
    pub fn slice(
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Self {
        Self::Slice { start, stop, step }
    }
}
