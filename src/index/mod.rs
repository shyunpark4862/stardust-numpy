//! Indexing: [`IndexSpec`], [`gather`], [`scatter`], [`scatter_array`].
//!
//! Index expressions are normalized into [`IndexSpec`]. Ellipsis and trailing
//! full-slice padding are expanded during index preparation.

mod bounds;
mod ops;
mod prepare;
mod spec;

pub(crate) use bounds::advance_multi_index;
pub use ops::{gather, scatter, scatter_array};
pub use spec::IndexSpec;
