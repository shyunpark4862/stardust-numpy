//! Error types for `sdnp`.

use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised by array construction and operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// Shape product does not match buffer length.
    #[error("buffer length {buffer_len} does not match shape size {size}")]
    BufferSizeMismatch {
        /// Actual buffer length.
        buffer_len: usize,
        /// Product of shape dimensions.
        size: usize,
    },

    /// `shape` and `strides` have different ranks.
    #[error(
        "shape rank {shape_ndim} does not match strides rank {strides_ndim}"
    )]
    ShapeStridesMismatch {
        /// Number of shape axes.
        shape_ndim: usize,
        /// Number of stride axes.
        strides_ndim: usize,
    },

    /// Broadcast failure (shapes cannot be aligned).
    #[error("cannot broadcast shapes {shapes:?}")]
    Broadcast {
        /// Shapes that could not be broadcast together.
        shapes: Vec<Vec<usize>>,
    },

    /// Index out of bounds.
    #[error("index {index} out of bounds for axis of length {axis_len}")]
    IndexOutOfBounds {
        /// Requested index.
        index: isize,
        /// Axis length.
        axis_len: usize,
    },

    /// Attempted write to a read-only array (e.g. a broadcast view).
    #[error("array is read-only")]
    ReadOnly,

    /// Integer division by zero.
    #[error("division by zero")]
    DivideByZero,

    /// Generic invalid argument.
    #[error("{0}")]
    InvalidArgument(String),
}
