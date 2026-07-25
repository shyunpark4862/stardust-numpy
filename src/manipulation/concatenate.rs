use crate::array::Array;
use crate::axis::normalize_axis;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::checked_size_of_shape;
use crate::traversal::{extend_unary, RunPlan};

/// Join arrays along an existing `axis`.
pub fn concatenate<T: Scalar>(
    arrays: &[&Array<T>],
    axis: isize,
) -> Result<Array<T>> {
    debug_assert!(
        !arrays.is_empty(),
        "concatenate requires at least one array"
    );
    let first = arrays[0];
    let axis = normalize_axis(axis, first.ndim());

    let mut output_shape = first.shape().to_vec();
    let mut axis_len = 0usize;
    for array in arrays {
        axis_len = axis_len
            .checked_add(array.shape()[axis])
            .expect("concatenated axis length overflows usize");
    }
    output_shape[axis] = axis_len;

    let capacity = checked_size_of_shape(&output_shape)?;
    let leading_count = checked_size_of_shape(&first.shape()[..axis])?;
    let mut output = Vec::with_capacity(capacity);

    for leading_index in 0..leading_count {
        for array in arrays {
            append_axis_slab(&mut output, array, axis, leading_index)?;
        }
    }
    debug_assert_eq!(output.len(), capacity);
    Array::from_vec(output, &output_shape)
}

fn append_axis_slab<T: Scalar>(
    output: &mut Vec<T>,
    array: &Array<T>,
    axis: usize,
    leading_index: usize,
) -> Result<()> {
    let trailing_len = checked_size_of_shape(&array.shape()[axis + 1..])?;
    let slab_len = array.shape()[axis]
        .checked_mul(trailing_len)
        .expect("array slab size overflows usize");
    if slab_len == 0 {
        return Ok(());
    }

    if let Some(slice) = array.as_c_contiguous_slice() {
        let start = leading_index
            .checked_mul(slab_len)
            .expect("array offset overflows usize");
        output.extend_from_slice(&slice[start..start + slab_len]);
        return Ok(());
    }

    let leading_shape = &array.shape()[..axis];
    let leading_strides = &array.strides()[..axis];
    let mut remainder = leading_index;
    let mut base = array.offset() as isize;
    for dimension in (0..axis).rev() {
        let coordinate = remainder % leading_shape[dimension];
        remainder /= leading_shape[dimension];
        base += coordinate as isize * leading_strides[dimension];
    }
    debug_assert!(base >= 0);
    let slab_shape = &array.shape()[axis..];
    let slab_strides = &array.strides()[axis..];
    let plan = RunPlan::new(slab_shape, [slab_strides]);
    extend_unary(&plan, &array.data, base as usize, output, |value| value);
    Ok(())
}
