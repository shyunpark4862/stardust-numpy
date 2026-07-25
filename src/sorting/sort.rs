//! Stable sort and argsort with NumPy-like axis handling.
//!
//! With `axis = None`, arrays flatten in C order before sorting. With a
//! specific axis, each 1-D slice along that axis is sorted independently
//! while the overall shape is preserved.

use crate::axis::normalize_axis;
use crate::error::Result;
use crate::shape::checked_allocation_len;
use crate::Array;

use super::SortElement;

/// Return a stably sorted copy of `a`.
///
/// Like NumPy's `np.sort`. Sorting is stable: equal elements retain their
/// relative order. Floating-point NaNs sort after all non-NaN values.
///
/// **Axis semantics:**
/// * `Some(axis)` — each 1-D slice along `axis` is sorted independently;
///   the output shape matches the input. Negative axes count from the end.
/// * `None` — the array is flattened in C order, sorted as one sequence,
///   and returned as a 1-D array of length `a.size()`.
///
/// # Arguments
///
/// * `a` - Input array (`bool`, `i64`, or `f64`).
/// * `axis` - Axis to sort along, or `None` to sort the flattened array.
///
/// # Returns
///
/// A new sorted [`Array`]. Shape matches `a` when `axis` is `Some`; shape
/// `[size]` when `axis` is `None`.
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
/// use sdnp::{sort, Array};
///
/// let a = Array::from_slice(&[3_i64, 1, 2], &[3]).unwrap();
/// assert_eq!(sort(&a, None).unwrap().to_vec(), vec![1, 2, 3]);
/// ```
pub fn sort<T: SortElement>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<T>> {
    let mut data = a.to_vec();
    let shape = match axis {
        Some(axis) => {
            let axis = normalize_axis(axis, a.ndim());
            sort_values_along_axis(&mut data, a.shape(), axis);
            a.shape().to_vec()
        }
        None => {
            data.sort_by(T::sort_cmp);
            vec![data.len()]
        }
    };
    Array::from_vec(data, &shape)
}

/// Return stable indices that would sort `a`.
///
/// Like NumPy's `np.argsort`. Indices refer to positions along the sorted
/// axis (when `axis` is `Some`) or to C-order flat positions (when `axis`
/// is `None`).
///
/// **Axis semantics:**
/// * `Some(axis)` — for each slice along `axis`, returns the permutation
///   that would sort that slice. Output shape matches `a`. Indices are local
///   to the axis (0 … axis_len − 1).
/// * `None` — returns indices into the C-order flattened input as a 1-D
///   array of length `a.size()`.
///
/// # Arguments
///
/// * `a` - Input array (`bool`, `i64`, or `f64`).
/// * `axis` - Axis to argsort along, or `None` for the flattened array.
///
/// # Returns
///
/// A new [`Array<i64>`] of sort indices.
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
/// use sdnp::{argsort, Array};
///
/// let a = Array::from_slice(&[30_i64, 10, 20], &[3]).unwrap();
/// assert_eq!(argsort(&a, None).unwrap().to_vec(), vec![1, 2, 0]);
/// ```
pub fn argsort<T: SortElement>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<i64>> {
    checked_allocation_len::<usize>(a.size())?;
    checked_allocation_len::<i64>(a.size())?;
    let data = a.to_vec();
    let (indices, shape) = match axis {
        Some(axis) => {
            let axis = normalize_axis(axis, a.ndim());
            (
                argsort_along_axis(&data, a.shape(), axis),
                a.shape().to_vec(),
            )
        }
        None => {
            let mut indices: Vec<usize> = (0..data.len()).collect();
            indices.sort_by(|&left, &right| data[left].sort_cmp(&data[right]));
            (indices, vec![data.len()])
        }
    };
    let indices = indices.into_iter().map(|i| i as i64).collect();
    Array::from_vec(indices, &shape)
}

fn sort_values_along_axis<T: SortElement>(
    data: &mut [T],
    shape: &[usize],
    axis: usize,
) {
    if data.is_empty() {
        return;
    }
    let axis_len = shape[axis];
    if axis_len < 2 {
        return;
    }
    let inner: usize = shape[axis + 1..].iter().product();
    if inner == 1 {
        // Contiguous axis slices: sort each row in one pass.
        for chunk in data.chunks_exact_mut(axis_len) {
            chunk.sort_by(T::sort_cmp);
        }
        return;
    }
    // General case: gather each 1-D slice along axis, sort, scatter back.
    let outer: usize = shape[..axis].iter().product();
    let mut slice = Vec::with_capacity(axis_len);
    for outer_index in 0..outer {
        let base = outer_index * axis_len * inner;
        for inner_index in 0..inner {
            slice.clear();
            for axis_index in 0..axis_len {
                slice.push(data[base + axis_index * inner + inner_index]);
            }
            slice.sort_by(T::sort_cmp);
            for (axis_index, &value) in slice.iter().enumerate() {
                data[base + axis_index * inner + inner_index] = value;
            }
        }
    }
}

fn argsort_along_axis<T: SortElement>(
    data: &[T],
    shape: &[usize],
    axis: usize,
) -> Vec<usize> {
    if data.is_empty() {
        return Vec::new();
    }
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut result = vec![0; data.len()];
    let mut indices = Vec::with_capacity(axis_len);
    for outer_index in 0..outer {
        let base = outer_index * axis_len * inner;
        for inner_index in 0..inner {
            indices.clear();
            indices.extend(0..axis_len);
            indices.sort_by(|&left, &right| {
                data[base + left * inner + inner_index]
                    .sort_cmp(&data[base + right * inner + inner_index])
            });
            for (position, &axis_index) in indices.iter().enumerate() {
                result[base + position * inner + inner_index] = axis_index;
            }
        }
    }
    result
}
