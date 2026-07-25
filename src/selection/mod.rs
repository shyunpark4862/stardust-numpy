//! Element selection, masking, and bounding operations.
//!
//! Implements NumPy-style `np.where`, `np.nonzero`, and `np.clip`. All
//! functions broadcast inputs as needed and return newly allocated
//! C-contiguous results unless noted otherwise.
//!
//! Boolean masks follow [`AsBool`](crate::dtype::AsBool) truthiness rules
//! during broadcast alignment.

mod ops;

pub use ops::{clip, nonzero, where_};
