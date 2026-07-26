//! Stride-aware reduction and cumulative execution kernels.
//!
//! Dispatch mirrors ufunc and indexing: suffix reduced axes on a C-contiguous
//! buffer use chunk folds; prefix reductions scan rows in parallel; all other
//! layouts walk outer runs plus coalesced reduced-axis inner runs via
//! [`ReducedAxisRuns`]. Contiguity is decided once at the dispatch site.

use std::sync::Arc;

use crate::array::Array;
use crate::axis::resolve_axis;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::reduction::plan::{
    AxisTraversalPlan, ReducePlan, TraversalSchedule,
};
use crate::shape::{c_order_strides_unchecked, checked_allocation_len};
use crate::traversal::{RunPlan, StrideCursor};

enum ReductionPath<'a, T> {
    SuffixContiguous(&'a [T]),
    PrefixContiguous(&'a [T]),
    GeneralStrided,
}

/// Classify input layout into a reduction traversal path.
///
/// Inspects C-contiguity once and maps [`ReducePlan::traversal_schedule`] to
/// a concrete slice borrow or the general-strided fallback.
///
/// # Arguments
///
/// * `a` - Input array (may or may not be C-contiguous).
/// * `plan` - Precomputed axis geometry for this reduction.
///
/// # Returns
///
/// [`ReductionPath::SuffixContiguous`] or [`ReductionPath::PrefixContiguous`]
/// when the buffer is contiguous and the plan allows chunk/row scans;
/// otherwise [`ReductionPath::GeneralStrided`].
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
        TraversalSchedule::PrefixContiguous { .. } => {
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
