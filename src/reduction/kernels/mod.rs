//! Stride-aware reduction / cumulative kernels.
//!
//! Dispatch (same philosophy as ufunc / indexing):
//! 1. suffix reduced axes + C-contiguous slice → contiguous chunk fold
//! 2. otherwise → general strided walk, itself split into an outer
//!    traversal and a coalesced reduced-axis inner run (see
//!    [`ReducedAxisRuns`]). A single reduced axis coalesces to exactly one run,
//!    so it needs no separate dispatch arm or duplicated kernel.
//!
//! Contiguity is decided once at the dispatch site via
//! [`Array::as_c_contiguous_slice`]; fast-path kernels never fall back.

use std::sync::Arc;

use crate::array::Array;
use crate::axis::normalize_axis;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::reduction::plan::{
    AxisTraversalPlan, ReducePlan, TraversalSchedule,
};
use crate::shape::c_order_strides;
use crate::traversal::{RunPlan, StrideCursor};

enum ReductionPath<'a, T> {
    SuffixContiguous(&'a [T]),
    PrefixContiguous(&'a [T]),
    GeneralStrided,
}

#[inline]
fn reduction_path<'a, T: Scalar>(
    a: &'a Array<T>,
    plan: &ReducePlan,
) -> ReductionPath<'a, T> {
    let contiguous = a.as_c_contiguous_slice();
    match plan.traversal_schedule(a.ndim(), contiguous.is_some()) {
        TraversalSchedule::SuffixContiguous => {
            ReductionPath::SuffixContiguous(contiguous.unwrap())
        }
        TraversalSchedule::PrefixContiguous {
            reduced_len,
            output_len,
        } => {
            debug_assert_eq!(reduced_len, plan.reduction_len);
            debug_assert_eq!(output_len, plan.output_len);
            ReductionPath::PrefixContiguous(contiguous.unwrap())
        }
        TraversalSchedule::GeneralStrided => ReductionPath::GeneralStrided,
    }
}

mod arg;
mod cumulative;
mod extremum;
mod fold;
mod reduced_axis_runs;
mod variance;

pub(crate) use arg::{
    arg_extremum_axis, arg_extremum_axis_ignore, arg_extremum_flat,
    arg_extremum_flat_ignore,
};
pub(crate) use cumulative::{cumulate, cumulate_ignore};
pub(crate) use extremum::{
    reduce_bool_extremum, reduce_bool_logical, reduce_f64_extremum,
    reduce_f64_extremum_ignore, reduce_i64_max, reduce_i64_min,
};
pub(crate) use fold::{
    reduce_associative, reduce_associative_with_plan, reduce_fold,
    reduce_ignore_with_counts,
};
pub(super) use reduced_axis_runs::ReducedAxisRuns;
pub(crate) use variance::{
    reduce_var, reduce_var_ignore_f64, transform_owned_c_order,
};
