//! NumPy-style read-only iteration over shapes and arrays.

mod array_iter;
mod multi_operand;
mod shape_index;

pub use array_iter::{ndenumerate, Axis0Iter, FlatIter, NdEnumerate};
pub use multi_operand::{nditer, NdIter};
pub use shape_index::{ndindex, NdIndex};
