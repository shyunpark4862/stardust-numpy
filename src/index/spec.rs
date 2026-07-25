//! Parsed index expressions before they touch an array shape.
//!
//! Each [`IndexSpec`] describes one slot in a NumPy-style index tuple. Higher
//! layers expand ellipsis, boolean masks, and missing trailing axes before
//! gather/scatter run. Think of this as the Rust-side mirror of what Python
//! passes to `__getitem__` / `__setitem__`.

use crate::array::Array;

/// One component of a normalized index expression.
///
/// Build tuples of [`IndexSpec`] values and pass them to [`gather`],
/// [`scatter`], or [`scatter_array`]. Ellipsis and boolean masks are expanded
/// and boolean masks are expanded during preparation; integer indices may be
/// negative until they are resolved against axis length.
///
/// # Examples
///
/// ```rust
/// use sdnp::IndexSpec;
///
/// // Row slice: index 0, all columns.
/// let row = [IndexSpec::index(0), IndexSpec::full()];
/// assert!(matches!(row[0], IndexSpec::Index(0)));
/// ```
#[derive(Clone, Debug)]
pub enum IndexSpec {
    /// Integer index along one source axis (negative until resolved).
    Index(i64),
    /// Python-style slice; omitted bounds use axis defaults.
    Slice {
        /// Start (inclusive); `None` picks the default for the step sign.
        start: Option<i64>,
        /// Stop (exclusive); `None` picks the default for the step sign.
        stop: Option<i64>,
        /// Step; `None` means `1`. Zero is rejected later.
        step: Option<i64>,
    },
    /// Insert a length-1 axis, like NumPy's `np.newaxis` or `None`.
    NewAxis,
    /// Fill remaining source axes with full slices (at most one per index).
    Ellipsis,
    /// Boolean mask; consumes `ndim` consecutive source axes.
    BoolArray(Array<bool>),
    /// Integer fancy index for one source axis (may be negative until
    /// resolved).
    IntegerArray(Array<i64>),
}

impl IndexSpec {
    /// Full-axis slice (`:`), equivalent to `Slice { start: None, ... }`.
    ///
    /// # Returns
    ///
    /// An [`IndexSpec::Slice`] with `step: Some(1)` and open bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::IndexSpec;
    ///
    /// let spec = IndexSpec::full();
    /// assert!(matches!(
    ///     spec,
    ///     IndexSpec::Slice {
    ///         step: Some(1),
    ///         ..
    ///     }
    /// ));
    /// ```
    #[inline]
    pub fn full() -> Self {
        Self::Slice {
            start: None,
            stop: None,
            step: Some(1),
        }
    }

    /// Integer index along one axis.
    ///
    /// Negative values count from the end of the axis and are resolved during
    /// index preparation.
    ///
    /// # Arguments
    ///
    /// * `i` - Element index along one source axis.
    ///
    /// # Returns
    ///
    /// An [`IndexSpec::Index`] variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::IndexSpec;
    ///
    /// assert!(matches!(IndexSpec::index(-1), IndexSpec::Index(-1)));
    /// ```
    #[inline]
    pub fn index(i: i64) -> Self {
        Self::Index(i)
    }

    /// Slice with optional start, stop, and step.
    ///
    /// Omitted bounds follow NumPy defaults once the step sign is known. A
    /// step of zero is rejected during preparation.
    ///
    /// # Arguments
    ///
    /// * `start` - Inclusive start index, or `None` for the axis default.
    /// * `stop` - Exclusive stop index, or `None` for the axis default.
    /// * `step` - Stride between selected elements; `None` means `1`.
    ///
    /// # Returns
    ///
    /// An [`IndexSpec::Slice`] variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp::IndexSpec;
    ///
    /// let every_other = IndexSpec::slice(Some(0), None, Some(2));
    /// assert!(matches!(
    ///     every_other,
    ///     IndexSpec::Slice {
    ///         start: Some(0),
    ///         step: Some(2),
    ///         ..
    ///     }
    /// ));
    /// ```
    #[inline]
    pub fn slice(
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Self {
        Self::Slice { start, stop, step }
    }
}
