//! Indexing: [`IndexSpec`], [`gather`], [`scatter`], [`scatter_array`].
//!
//! User-facing Python objects are normalized into [`IndexSpec`] (Phase 8).
//! Ellipsis and trailing full-slice padding are handled here for Rust tests.

mod bounds;
mod ops;
mod prepare;
mod spec;

pub(crate) use bounds::advance_multi_index;
pub use ops::{gather, scatter, scatter_array};
pub use spec::IndexSpec;
