//! Conditional selection and coordinate-finding on arrays.
//!
//! [`where_`] broadcasts a boolean condition and two value arrays, then picks
//! element-wise. [`nonzero`] reports C-order coordinates of true elements.
//! [`clip`] clamps values to optional scalar bounds.

use crate::array::Array;
use crate::broadcast::broadcast_shapes;
use crate::dtype::{AsBool, CastTo, Promote, Scalar};
use crate::error::Result;
use crate::shape::{checked_allocation_len, size_of_shape_unchecked};
use crate::traversal::{collect_ternary, RunPlan, StrideIter};
use crate::ufunc::kernels::map_unary;

/// Select from `x` or `y` according to boolean `condition`.
///
/// Like NumPy's `np.where`: for each output position, the value comes from
/// `x` when `condition` is true and from `y` otherwise. All three inputs
/// are broadcast together to a common shape using standard NumPy rules.
/// Selected values are cast to the promoted dtype of `x` and `y`.
///
/// **Broadcasting:** `condition`, `x`, and `y` must be mutually
/// broadcastable. Scalar inputs broadcast across the full output grid.
///
/// # Arguments
///
/// * `condition` - Boolean mask (may be broadcast).
/// * `x` - Values chosen where `condition` is true.
/// * `y` - Values chosen where `condition` is false.
///
/// # Returns
///
/// A new [`Array`] with the broadcast output shape and promoted element type.
///
/// # Errors
///
/// * [`Error::Broadcast`](crate::Error::Broadcast) - The three inputs cannot
///   be aligned to a common shape.
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   exceeds platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{where_, Array};
///
/// let cond = Array::from_slice(&[true, false, true], &[3]).unwrap();
/// let x = Array::from_slice(&[10_i64, 20, 30], &[3]).unwrap();
/// let y = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
/// assert_eq!(where_(&cond, &x, &y).unwrap().to_vec(), vec![10, 2, 30]);
/// ```
pub fn where_<X, Y>(
    condition: &Array<bool>,
    x: &Array<X>,
    y: &Array<Y>,
) -> Result<Array<<X as Promote<Y>>::Output>>
where
    X: Promote<Y> + CastTo<<X as Promote<Y>>::Output>,
    Y: Scalar + CastTo<<X as Promote<Y>>::Output>,
    <X as Promote<Y>>::Output: Scalar,
{
    let shape = broadcast_shapes(&[condition.shape(), x.shape(), y.shape()])?;
    let condition = condition.broadcast_to(&shape)?;
    let x = x.broadcast_to(&shape)?;
    let y = y.broadcast_to(&shape)?;
    checked_allocation_len::<<X as Promote<Y>>::Output>(
        size_of_shape_unchecked(&shape),
    )?;

    if let (Some(conditions), Some(xs), Some(ys)) = (
        condition.as_c_contiguous_slice(),
        x.as_c_contiguous_slice(),
        y.as_c_contiguous_slice(),
    ) {
        let out = conditions
            .iter()
            .copied()
            .zip(xs.iter().copied())
            .zip(ys.iter().copied())
            .map(
                |((choose_x, x), y)| {
                    if choose_x {
                        x.cast_to()
                    } else {
                        y.cast_to()
                    }
                },
            )
            .collect();
        return Array::from_vec(out, &shape);
    }

    let plan =
        RunPlan::new(&shape, [condition.strides(), x.strides(), y.strides()]);
    let out = collect_ternary(
        &plan,
        condition.as_buffer(),
        x.as_buffer(),
        y.as_buffer(),
        [condition.offset(), x.offset(), y.offset()],
        |choose_x, x, y| {
            if choose_x {
                x.cast_to()
            } else {
                y.cast_to()
            }
        },
    );
    Array::from_vec(out, &shape)
}

/// Return C-order coordinates of elements that evaluate to true.
///
/// Like NumPy's `np.nonzero`: returns one 1-D `i64` array per input axis.
/// Each coordinate array has length equal to the number of true elements.
/// Empty inputs yield empty coordinate arrays; 0-D inputs yield no
/// coordinates (empty vectors for each axis).
///
/// # Arguments
///
/// * `a` - Input array; elements are tested with [`AsBool`].
///
/// # Returns
///
/// A vector of `ndim` one-dimensional [`Array<i64>`] coordinate arrays.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   exceeds platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{nonzero, Array};
///
/// let a = Array::from_slice(&[0_i64, 1, 0, 2], &[4]).unwrap();
/// let coords = nonzero(&a).unwrap();
/// assert_eq!(coords[0].to_vec(), vec![1, 3]);
/// ```
pub fn nonzero<T: AsBool>(a: &Array<T>) -> Result<Vec<Array<i64>>> {
    checked_allocation_len::<i64>(a.size())?;
    let mut coordinates = vec![Vec::new(); a.ndim()];

    StrideIter::new(a.shape(), a.strides(), a.offset()).for_each(
        |buffer_offset, indices| {
            if a.as_buffer()[buffer_offset].as_bool() {
                for (axis, &index) in indices.iter().enumerate() {
                    coordinates[axis].push(index as i64);
                }
            }
        },
    );

    let count = coordinates.first().map_or(0, Vec::len);
    coordinates
        .into_iter()
        .map(|axis| Array::from_vec(axis, &[count]))
        .collect()
}

/// Clamp every element between optional scalar bounds.
///
/// Like NumPy's `np.clip`: the lower bound is applied before the upper bound,
/// so `min > max` still follows NumPy's two-pass semantics. `None` leaves
/// that side unbounded.
///
/// # Arguments
///
/// * `a` - Input array.
/// * `min` - Optional lower bound applied first.
/// * `max` - Optional upper bound applied second.
///
/// # Returns
///
/// A new [`Array`] with the same shape as `a`.
///
/// # Errors
///
/// Returns `Ok` for all valid inputs; does not allocate beyond the unary
/// map path (no additional error variants beyond internal invariants).
///
/// # Examples
///
/// ```rust
/// use sdnp::{clip, Array};
///
/// let a = Array::from_slice(&[0_i64, 5, 10], &[3]).unwrap();
/// assert_eq!(clip(&a, Some(2), Some(8)).unwrap().to_vec(), vec![2, 5, 8]);
/// ```
pub fn clip<T>(a: &Array<T>, min: Option<T>, max: Option<T>) -> Result<Array<T>>
where
    T: Scalar + PartialOrd,
{
    map_unary(a, |mut value| {
        if let Some(lower) = min {
            if value < lower {
                value = lower;
            }
        }
        if let Some(upper) = max {
            if value > upper {
                value = upper;
            }
        }
        value
    })
}
