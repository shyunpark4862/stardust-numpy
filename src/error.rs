//! Error types for the `sdnp` core.
//!
//! The core owns semantic validation shared by every frontend, including axis,
//! shape, indexing, broadcast, and layout rules. Bindings may validate
//! frontend-specific policy, but callers can rely on core operations returning
//! structured errors instead of requiring prevalidation.

use thiserror::Error;

/// Shorthand for `Result<T, [`Error`]>` used throughout the crate.
///
/// Most fallible array operations return this alias rather than spelling
/// the full `std::result::Result` type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised during semantic validation and array execution.
///
/// Frontends translate these variants into their native exception types.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// The backing buffer length does not equal the shape's element count.
    ///
    /// Raised when constructing an array from a slice or vector whose length
    /// is not the product of the supplied shape dimensions.
    #[error("buffer length {buffer_len} does not match shape size {size}")]
    BufferSizeMismatch {
        /// Actual number of elements in the supplied buffer.
        buffer_len: usize,
        /// Expected count: product of all shape dimensions.
        size: usize,
    },

    /// `shape` and `strides` vectors have different lengths.
    ///
    /// Every axis must have a matching stride entry; this catches rank
    /// mismatches before layout bounds are computed.
    #[error(
        "shape rank {shape_ndim} does not match strides rank {strides_ndim}"
    )]
    ShapeStridesMismatch {
        /// Number of entries in the shape vector.
        shape_ndim: usize,
        /// Number of entries in the strides vector.
        strides_ndim: usize,
    },

    /// Two or more shapes cannot be aligned under NumPy broadcast rules.
    ///
    /// Operand shapes were pairwise checked and at least one axis could not
    /// be reconciled to a common length (excluding length-1 broadcast).
    #[error("cannot broadcast shapes {shapes:?}")]
    Broadcast {
        /// The incompatible shapes that were supplied.
        shapes: Vec<Vec<usize>>,
    },

    /// A variadic operation received no arrays.
    #[error("{op} requires at least one array")]
    EmptyOperands {
        /// Public operation name used in the diagnostic.
        op: &'static str,
    },

    /// An operand rank is outside an operation's supported contract.
    #[error("{op} requires {expected}; got rank {actual}")]
    InvalidRank {
        /// Public operation name used in the diagnostic.
        op: &'static str,
        /// Human-readable supported rank contract.
        expected: &'static str,
        /// Actual operand rank.
        actual: usize,
    },

    /// One operand in a variadic operation has a different rank.
    #[error(
        "all arrays must have the same rank in {op}; array 0 has rank \
         {expected}, array {index} has rank {actual}"
    )]
    RankMismatch {
        /// Public operation name used in the diagnostic.
        op: &'static str,
        /// Rank of the first operand.
        expected: usize,
        /// Index of the incompatible operand.
        index: usize,
        /// Rank of the incompatible operand.
        actual: usize,
    },

    /// One operand has an incompatible shape.
    #[error(
        "{op} shape mismatch: array 0 has shape {expected:?}, array {index} \
         has shape {actual:?}"
    )]
    ShapeMismatch {
        /// Public operation name used in the diagnostic.
        op: &'static str,
        /// Shape of the first operand.
        expected: Vec<usize>,
        /// Index of the incompatible operand.
        index: usize,
        /// Shape of the incompatible operand.
        actual: Vec<usize>,
    },

    /// Matrix contraction dimensions differ.
    #[error("matmul inner dimensions differ: {left} != {right}")]
    ContractionMismatch {
        /// Left operand's contraction length.
        left: usize,
        /// Right operand's contraction length.
        right: usize,
    },

    /// Matrix batch prefixes cannot be broadcast together.
    #[error(
        "matmul batch dimensions are not broadcast-compatible: {left:?} and \
         {right:?}"
    )]
    BatchBroadcastMismatch {
        /// Left batch shape.
        left: Vec<usize>,
        /// Right batch shape.
        right: Vec<usize>,
    },

    /// Flattened operands to `vdot` have different element counts.
    #[error("vdot requires equal flattened sizes: {left} != {right}")]
    FlattenedSizeMismatch {
        /// Left flattened size.
        left: usize,
        /// Right flattened size.
        right: usize,
    },

    /// A multi-index referred to a coordinate outside an axis length.
    ///
    /// Raised by [`Array::get`](crate::Array::get) and [`Array::set`] when
    /// any index is negative in effect (≥ axis length in unsigned form).
    #[error("index {index} out of bounds for axis of length {axis_len}")]
    IndexOutOfBounds {
        /// The out-of-range index that was requested.
        index: i64,
        /// Length of the axis that was indexed.
        axis_len: usize,
    },

    /// An axis index falls outside the valid range for an array rank.
    #[error("axis {axis} is out of bounds for array of dimension {ndim}")]
    AxisOutOfBounds {
        /// The signed axis supplied by the caller.
        axis: isize,
        /// Rank against which the axis was resolved.
        ndim: usize,
    },

    /// An axis list resolves to the same axis more than once.
    #[error("axes must not contain duplicates")]
    DuplicateAxes,

    /// An axis sequence is not a complete permutation of the array axes.
    #[error("axes must be a permutation")]
    NotPermutation,

    /// Two axes that define a plane resolve to the same dimension.
    #[error("axes must be different")]
    AxesMustDiffer,

    /// Squeeze was requested for an axis whose length is not one.
    #[error("cannot squeeze axis {axis} with length {axis_len}")]
    CannotSqueezeAxis {
        /// Resolved axis that was requested for removal.
        axis: usize,
        /// Current length of that axis.
        axis_len: usize,
    },

    /// A reduction requiring at least one value received an empty slice.
    #[error("{op} of empty array / empty axis")]
    EmptyReduction {
        /// Public operation name used in the diagnostic.
        op: &'static str,
    },

    /// A NaN-ignoring arg reduction found no finite value in a slice.
    #[error("{op} of all-NaN slice")]
    AllNanSlice {
        /// Public operation name used in the diagnostic.
        op: &'static str,
    },

    /// An index expression is structurally invalid for the source shape.
    ///
    /// Covers duplicate ellipses, too many consumed axes, unsupported index
    /// kinds on specialized paths, and boolean masks whose shape does not
    /// match the axes they consume.
    #[error("{0}")]
    InvalidIndex(String),

    /// A write was attempted on a read-only view (e.g. a broadcast view).
    ///
    /// Broadcast arrays set [`Array::is_writable`](crate::Array::is_writable)
    /// to `false`; mutating methods reject them before copy-on-write runs.
    #[error("array is read-only")]
    ReadOnly,

    /// Integer division by zero was requested.
    ///
    /// Raised by reduction kernels that divide by an axis length or count
    /// that evaluates to zero at runtime.
    #[error("division by zero")]
    DivideByZero,

    /// Catch-all for rare internal paths; prefer `debug_assert` for misuse.
    ///
    /// Covers shape/stride overflow, invalid reshape specs, layout bounds
    /// violations, and allocation limit failures. The message is intended
    /// for debugging rather than end-user display.
    #[error("{0}")]
    InvalidArgument(String),
}
