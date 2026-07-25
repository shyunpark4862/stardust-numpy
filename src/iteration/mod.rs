//! NumPy-style read-only iteration over shapes and arrays.
//!
//! Provides flat value iteration, coordinate enumeration, axis-0 slicing, and
//! multi-array lockstep walks. All iterators follow logical C-order, matching
//! how NumPy flattens arrays for `flat`, `ndenumerate`, and `nditer`.
//!
//! **Stride note:** Contiguous arrays use direct slice iteration; strided
//! layouts delegate to [`StrideIter`](crate::traversal::StrideIter) so
//! iteration cost tracks memory layout rather than rank alone.

mod array_iter;
mod multi_operand;
mod shape_index;

pub use array_iter::{ndenumerate, Axis0Iter, FlatIter, NdEnumerate};
pub use multi_operand::{nditer, NdIter};
pub use shape_index::{ndindex, NdIndex};
