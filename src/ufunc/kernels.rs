//! Stride-aware element-wise map kernels for ufuncs.
//!
//! When operands are C-contiguous, kernels zip flat slices directly. Otherwise
//! layouts are coalesced into [`RunPlan`](crate::traversal::RunPlan) and walked
//! as fixed-stride runs, which handles broadcasting (`stride == 0`) efficiently.

use crate::array::Array;
use crate::broadcast::broadcast_shape;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::{checked_allocation_len, size_of_shape_unchecked};
use crate::traversal::{
    collect_binary, collect_unary, try_collect_binary, RunPlan,
};

/// Apply `f` to every element of one array, producing a C-contiguous result.
///
/// Fast path: if the input is C-contiguous, maps over a flat slice with no
/// stride logic. Slow path: builds a unary [`RunPlan`] and uses coalesced run
/// iteration ([`collect_unary`]) so non-contiguous and broadcast views still
/// avoid per-element index unraveling.
///
/// # Arguments
///
/// * `array` - Source array; dtype `A` must implement [`Scalar`].
/// * `f` - Unary transform applied to each element (may change dtype to `Out`).
///
/// # Returns
///
/// A new C-contiguous [`Array<Out>`] with the same shape as `array`.
///
/// # Errors
///
/// Returns an allocation error from [`checked_allocation_len`] when the
/// output would exceed safe limits.
pub(crate) fn map_unary<A, Out, F>(array: &Array<A>, f: F) -> Result<Array<Out>>
where
    A: Scalar,
    Out: Scalar,
    F: FnMut(A) -> Out,
{
    checked_allocation_len::<Out>(array.size())?;
    if let Some(values) = array.as_c_contiguous_slice() {
        let output = values.iter().copied().map(f).collect();
        return Array::from_vec(output, array.shape());
    }

    let plan = RunPlan::new(array.shape(), [array.strides()]);
    let output = collect_unary(&plan, &array.data, array.offset(), f);
    Array::from_vec(output, array.shape())
}

/// Broadcast two operands, then apply infallible `f` element-wise.
///
/// Aligns shapes via [`align_binary`] (broadcasting views when needed), then
/// either zips contiguous slices or walks a binary [`RunPlan`] with [`RunKind`]
/// specialization inside [`collect_binary`]. Dtype promotion is the caller's
/// responsibility: `f` receives already-coerced `A` and `B` scalars.
///
/// # Arguments
///
/// * `left` - Left operand array.
/// * `right` - Right operand array (may differ in shape; broadcast rules apply).
/// * `f` - Infallible binary transform producing `Out` per element.
///
/// # Returns
///
/// A C-contiguous [`Array<Out>`] with the broadcast result shape.
///
/// # Errors
///
/// Returns broadcast or allocation errors from [`align_binary`] or
/// [`checked_allocation_len`].
pub(super) fn map_binary<A, B, Out, F>(
    left: &Array<A>,
    right: &Array<B>,
    mut f: F,
) -> Result<Array<Out>>
where
    A: Scalar,
    B: Scalar,
    Out: Scalar,
    F: FnMut(A, B) -> Out,
{
    let aligned = align_binary(left, right)?;
    let left = aligned.left.as_ref().unwrap_or(left);
    let right = aligned.right.as_ref().unwrap_or(right);
    let shape = &aligned.shape;
    checked_allocation_len::<Out>(size_of_shape_unchecked(shape))?;

    if let (Some(left_values), Some(right_values)) =
        (left.as_c_contiguous_slice(), right.as_c_contiguous_slice())
    {
        let output = left_values
            .iter()
            .copied()
            .zip(right_values.iter().copied())
            .map(|(left, right)| f(left, right))
            .collect();
        return Array::from_vec(output, shape);
    }

    let plan = RunPlan::new(shape, [left.strides(), right.strides()]);
    let output = collect_binary(
        &plan,
        &left.data,
        &right.data,
        [left.offset(), right.offset()],
        f,
    );
    Array::from_vec(output, shape)
}

/// Broadcast two operands, then apply fallible `f` element-wise.
///
/// Same alignment and coalesced [`RunPlan`] strategy as [`map_binary`], but
/// uses [`try_collect_binary`] so integer division and similar ops can abort
/// mid-traversal with a typed error.
///
/// # Arguments
///
/// * `left` - Left operand array.
/// * `right` - Right operand array.
/// * `f` - Fallible binary transform (`Result<Out>` per element).
///
/// # Returns
///
/// A C-contiguous [`Array<Out>`] on success.
///
/// # Errors
///
/// Propagates broadcast/allocation errors, or the first error returned by `f`.
pub(super) fn try_map_binary<A, B, Out, F>(
    left: &Array<A>,
    right: &Array<B>,
    mut f: F,
) -> Result<Array<Out>>
where
    A: Scalar,
    B: Scalar,
    Out: Scalar,
    F: FnMut(A, B) -> Result<Out>,
{
    let aligned = align_binary(left, right)?;
    let left = aligned.left.as_ref().unwrap_or(left);
    let right = aligned.right.as_ref().unwrap_or(right);
    let shape = &aligned.shape;
    checked_allocation_len::<Out>(size_of_shape_unchecked(shape))?;

    if let (Some(left_values), Some(right_values)) =
        (left.as_c_contiguous_slice(), right.as_c_contiguous_slice())
    {
        let mut output = Vec::with_capacity(size_of_shape_unchecked(shape));
        for (&left, &right) in left_values.iter().zip(right_values) {
            output.push(f(left, right)?);
        }
        return Array::from_vec(output, shape);
    }
    let plan = RunPlan::new(shape, [left.strides(), right.strides()]);
    let output = try_collect_binary(
        &plan,
        &left.data,
        &right.data,
        [left.offset(), right.offset()],
        f,
    )?;
    Array::from_vec(output, shape)
}

/// Broadcast views produced before a binary map loop.
struct AlignedBinary<A: Scalar, B: Scalar> {
    left: Option<Array<A>>,
    right: Option<Array<B>>,
    shape: Vec<usize>,
}

/// Align two arrays to a common broadcast shape for binary ufuncs.
///
/// When shapes already match, returns the original arrays unchanged (`None`
/// placeholders). Otherwise computes the broadcast shape and materializes
/// broadcast views only for operands that need them. Stride coalescing in the
/// subsequent [`RunPlan`] then sees broadcast strides (`0`) on expanded axes.
///
/// # Arguments
///
/// * `left` - First operand.
/// * `right` - Second operand.
///
/// # Returns
///
/// An [`AlignedBinary`] holding optional broadcast views and the result shape.
///
/// # Errors
///
/// Returns an error from [`broadcast_shape`] or [`Array::broadcast_to`] when
/// operands are not broadcast-compatible.
fn align_binary<A: Scalar, B: Scalar>(
    left: &Array<A>,
    right: &Array<B>,
) -> Result<AlignedBinary<A, B>> {
    if left.shape() == right.shape() {
        return Ok(AlignedBinary {
            left: None,
            right: None,
            shape: left.shape().to_vec(),
        });
    }
    let shape = broadcast_shape(left.shape(), right.shape())?;
    let left = (left.shape() != shape.as_slice())
        .then(|| left.broadcast_to(&shape))
        .transpose()?;
    let right = (right.shape() != shape.as_slice())
        .then(|| right.broadcast_to(&shape))
        .transpose()?;
    Ok(AlignedBinary { left, right, shape })
}
