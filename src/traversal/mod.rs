//! Shared strided-layout and run traversal primitives.

mod layout;
mod run;
mod stride_iter;

pub(crate) use layout::CoalescedLayout;
pub(crate) use run::{
    collect_binary, collect_ternary, collect_unary, extend_unary,
    try_collect_binary, RunKind, RunPlan,
};
pub(crate) use stride_iter::{StrideCursor, StrideIter};
