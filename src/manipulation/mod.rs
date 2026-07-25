//! Join arrays along existing axes and insert new stack axes.
//!
//! [`concatenate`] grows one existing dimension; [`stack`] inserts a fresh
//! axis then concatenates. Convenience wrappers [`vstack`] and [`hstack`]
//! match common NumPy row/column stacking patterns.
//!
//! All outputs are newly allocated C-contiguous arrays; operands may be
//! strided views and are materialized slab-by-slab during copying.

mod concatenate;
mod stack;

pub use concatenate::concatenate;
pub use stack::{hstack, stack, vstack};
