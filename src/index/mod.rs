//! NumPy-style array indexing: read, write, and index normalization.
//!
//! User-facing index expressions are represented as [`IndexSpec`] values,
//! expanded into a prepared form, then executed as gather (read) or scatter
//! (write) operations. Basic indexing returns views when possible; fancy and
//! boolean indexing always materialize new arrays.
//!
//! # Pipeline
//!
//! 1. [`IndexSpec`] captures the raw tuple (integers, slices, ellipsis,
//!    `newaxis`, boolean masks, fancy arrays).
//! 2. [`prepare::prepare_index`] expands ellipsis, lowers boolean masks to
//!    integer coordinate arrays, resolves bounds, and broadcasts fancy
//!    operands.
//! 3. [`ops::gather`] and scatter helpers execute the prepared plan as a
//!    view (basic) or element-wise copy (fancy).

mod bounds;
mod ops;
mod prepare;
mod spec;

pub(crate) use bounds::advance_multi_index;
pub use ops::{gather, scatter, scatter_array};
pub use spec::IndexSpec;
