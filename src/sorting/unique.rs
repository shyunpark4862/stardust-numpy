use crate::error::Result;
use crate::Array;

use super::{UniqueElement, UniqueOptions, UniqueResult};

/// Return sorted unique values from the C-order flattened input.
///
/// All floating-point NaNs form one group. For complex values, any value with
/// a NaN component is a complex-NaN and all such values form one group.
pub fn unique<T: UniqueElement>(a: &Array<T>) -> Result<Array<T>> {
    Ok(unique_with(a, UniqueOptions::default())?.values)
}

/// Return sorted unique values and requested metadata.
///
/// The values, first indices, inverse map, and counts are all one-dimensional.
/// First indices and inverse indices refer to the input flattened in C order.
pub fn unique_with<T: UniqueElement>(
    a: &Array<T>,
    options: UniqueOptions,
) -> Result<UniqueResult<T>> {
    let data = a.to_vec();
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&left, &right| data[left].unique_cmp(&data[right]));

    let mut values = Vec::new();
    let mut first_indices = options
        .return_index
        .then(|| Vec::with_capacity(order.len()));
    let mut inverse = options.return_inverse.then(|| vec![0_i64; data.len()]);
    let mut counts = options
        .return_counts
        .then(|| Vec::with_capacity(order.len()));

    for input_index in order {
        let is_new = values
            .last()
            .map_or(true, |last| !data[input_index].unique_eq(last));
        if is_new {
            values.push(data[input_index]);
            if let Some(indices) = &mut first_indices {
                indices.push(input_index as i64);
            }
            if let Some(counts) = &mut counts {
                counts.push(0_i64);
            }
        } else if let Some(indices) = &mut first_indices {
            let last = indices.len() - 1;
            indices[last] = indices[last].min(input_index as i64);
        }

        let group = values.len() - 1;
        if let Some(inverse) = &mut inverse {
            inverse[input_index] = group as i64;
        }
        if let Some(counts) = &mut counts {
            counts[group] += 1;
        }
    }

    let value_len = values.len();
    Ok(UniqueResult {
        values: Array::from_vec(values, &[value_len])?,
        indices: optional_array(first_indices)?,
        inverse_indices: optional_array(inverse)?,
        counts: optional_array(counts)?,
    })
}

fn optional_array(data: Option<Vec<i64>>) -> Result<Option<Array<i64>>> {
    data.map(|data| {
        let len = data.len();
        Array::from_vec(data, &[len])
    })
    .transpose()
}
