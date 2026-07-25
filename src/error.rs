//! Error types for the `sdnp` core.
//!
//! Validation is split by layer: the Python binding (`sdnp-py`) handles user
//! input and policy; this crate enforces memory and layout invariants only.
//! Most caller misuse is caught by `debug_assert` in debug builds; the
//! variants here cover recoverable runtime failures such as broadcast
//! conflicts and out-of-bounds indexing.

use thiserror::Error;

/// Shorthand for `Result<T, [`Error`]>` used throughout the crate.
///
/// Most fallible array operations return this alias rather than spelling
/// the full `std::result::Result` type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised during array construction, layout validation, and
/// element-wise operations.
///
/// The Python binding layer (`sdnp-py`) adds its own validation; this enum
/// covers recoverable runtime failures in the Rust core.
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
