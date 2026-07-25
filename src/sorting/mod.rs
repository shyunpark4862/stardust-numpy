//! Stable sorting and unique-value extraction.
//!
//! Sorting follows NumPy-like axis semantics with stable ordering. Floating-
//! point NaNs sort after all non-NaN values. Unique operations flatten input
//! in C order before deduplication, matching `np.unique` behavior.
//!
//! See [`SortElement`] and [`UniqueElement`] for per-dtype comparison rules.

mod sort;
mod traits_options;
mod unique;

pub use sort::{argsort, sort};
pub use traits_options::{
    SortElement, UniqueElement, UniqueOptions, UniqueResult,
};
pub use unique::{unique, unique_with};
