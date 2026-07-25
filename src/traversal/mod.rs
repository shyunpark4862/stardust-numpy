//! Shared strided-layout traversal for kernels across the crate.
//!
//! These primitives walk array memory in C-order without materializing
//! broadcast shapes. [`CoalescedLayout`] merges adjacent axes into linear
//! runs when every operand's strides allow it; [`RunPlan`] then drives ufuncs,
//! reductions, indexing, and iterators over an outer run grid plus inner
//! fixed-stride segments.

mod layout;
mod run;
mod stride_iter;

pub(crate) use layout::CoalescedLayout;
pub(crate) use run::{
    collect_binary, collect_ternary, collect_unary, extend_unary,
    try_collect_binary, RunKind, RunPlan,
};
pub(crate) use stride_iter::{StrideCursor, StrideIter};
