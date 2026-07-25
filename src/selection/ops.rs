use crate::array::Array;
use crate::broadcast::broadcast_shapes;
use crate::dtype::{AsBool, CastTo, Promote, Scalar};
use crate::error::Result;
use crate::traversal::{collect_ternary, RunPlan, StrideIter};
use crate::ufunc::kernels::map_unary;

/// Select elements from `x` or `y` according to a boolean `condition`.
///
/// All three arrays are broadcast to a common shape. Elements selected from
/// `x` and `y` are cast to their promoted dtype, and the returned array is a
/// newly allocated C-contiguous array.
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

/// Return the C-order coordinates of elements that are logically true.
///
/// The result contains one one-dimensional `i64` array per input axis. Empty
/// inputs produce empty coordinate arrays. A zero-dimensional input has no
/// coordinate axes, so it produces an empty vector.
pub fn nonzero<T: AsBool>(a: &Array<T>) -> Result<Vec<Array<i64>>> {
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

/// Clamp every element of `a` between optional scalar bounds.
///
/// The lower bound is applied before the upper bound, matching NumPy's
/// behavior when both are supplied (including when `min > max`). `None`
/// leaves that side unbounded. The returned array is always a new
/// C-contiguous allocation.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn where_promotes_and_broadcasts_three_arrays() {
        let condition = Array::from_slice(&[true, false], &[2, 1]).unwrap();
        let x = Array::from_slice(&[1_i64, 2, 3], &[1, 3]).unwrap();
        let y = Array::from_slice(&[0.5_f64], &[]).unwrap();

        let selected = where_(&condition, &x, &y).unwrap();

        assert_eq!(selected.shape(), &[2, 3]);
        assert_eq!(selected.to_vec(), vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5]);
        assert!(selected.is_c_contiguous());
    }

    #[test]
    fn where_handles_noncontiguous_inputs() {
        let base = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let x = base.transpose();
        let condition = Array::from_slice(
            &[true, false, false, true, true, false],
            &[3, 2],
        )
        .unwrap();
        let y = Array::from_slice(&[9_i64], &[]).unwrap();

        let selected = where_(&condition, &x, &y).unwrap();

        assert_eq!(selected.to_vec(), vec![1, 9, 9, 5, 3, 9]);
    }

    #[test]
    fn nonzero_reports_transposed_coordinates_in_c_order() {
        let base = Array::from_slice(&[0_i64, 1, 2, 0, 3, 0], &[2, 3]).unwrap();
        let transposed = base.transpose();

        let indices = nonzero(&transposed).unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].to_vec(), vec![1, 1, 2]);
        assert_eq!(indices[1].to_vec(), vec![0, 1, 0]);
    }

    #[test]
    fn clip_returns_a_contiguous_copy() {
        let base =
            Array::from_slice(&[1_i64, 5, -2, 8, 3, 4], &[2, 3]).unwrap();
        let transposed = base.transpose();

        let clipped = clip(&transposed, Some(2), Some(5)).unwrap();

        assert_eq!(clipped.shape(), &[3, 2]);
        assert_eq!(clipped.to_vec(), vec![2, 5, 5, 3, 2, 4]);
        assert!(clipped.is_c_contiguous());
    }
}
