//! Extract sorted unique values from flattened array data.
//!
//! Input is always flattened in C order before deduplication, matching
//! NumPy's `np.unique`. NaN values (real or complex) form a single group.

use crate::error::Result;
use crate::shape::checked_allocation_len;
use crate::Array;

use super::{UniqueElement, UniqueOptions, UniqueResult};

/// Return sorted unique values from the C-order flattened input.
///
/// Like NumPy's `np.unique` with default options. The input is flattened in
/// C order, deduplicated, and returned sorted. All NaN values (real or
/// complex) collapse into a single group; complex values with any NaN
/// component form one complex-NaN group.
///
/// Equivalent to calling [`unique_with`] with [`UniqueOptions::default`]
/// and taking only the `values` field.
///
/// # Arguments
///
/// * `a` - Input array of any shape.
///
/// # Returns
///
/// A 1-D sorted [`Array`] of unique values.
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
/// use sdnp::{unique, Array};
///
/// let a = Array::from_slice(&[3_i64, 1, 2, 1, 3], &[5]).unwrap();
/// assert_eq!(unique(&a).unwrap().to_vec(), vec![1, 2, 3]);
/// ```
pub fn unique<T: UniqueElement>(a: &Array<T>) -> Result<Array<T>> {
    Ok(unique_with(a, UniqueOptions::default())?.values)
}

/// Return sorted unique values plus optional metadata arrays.
///
/// Like NumPy's `np.unique` with optional flags. The input is always
/// flattened in C order before deduplication. All returned arrays are 1-D.
///
/// **Options** ([`UniqueOptions`]):
/// * `return_index` — first flat C-order index of each unique value.
/// * `return_inverse` — for every flat input element, the index of its
///   unique group in the sorted unique output.
/// * `return_counts` — occurrence count for each unique value.
///
/// When `return_index` is enabled and duplicate values appear, the smallest
/// flat index among ties is kept. NaN grouping follows [`unique`].
///
/// # Arguments
///
/// * `a` - Input array of any shape.
/// * `options` - Flags selecting optional output arrays.
///
/// # Returns
///
/// A [`UniqueResult`] containing sorted unique `values` and any requested
/// metadata arrays (`indices`, `inverse_indices`, `counts`).
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
/// use sdnp::{unique_with, Array, UniqueOptions};
///
/// let a = Array::from_slice(&[2_i64, 1, 2], &[3]).unwrap();
/// let result = unique_with(
///     &a,
///     UniqueOptions {
///         return_inverse: true,
///         ..UniqueOptions::default()
///     },
/// )
/// .unwrap();
/// assert_eq!(result.values.to_vec(), vec![1, 2]);
/// assert_eq!(result.inverse_indices.unwrap().to_vec(), vec![1, 0, 1]);
/// ```
pub fn unique_with<T: UniqueElement>(
    a: &Array<T>,
    options: UniqueOptions,
) -> Result<UniqueResult<T>> {
    checked_allocation_len::<usize>(a.size())?;
    if options.return_index || options.return_inverse || options.return_counts {
        checked_allocation_len::<i64>(a.size())?;
    }
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
            // Keep the smallest flat index as the group's first occurrence.
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
