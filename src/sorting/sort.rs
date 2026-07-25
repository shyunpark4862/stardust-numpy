use crate::axis::normalize_axis;
use crate::error::Result;
use crate::Array;

use super::SortElement;

/// Return a stably sorted copy of `a`.
///
/// With `Some(axis)`, each slice along that axis is sorted independently and
/// the original shape is retained. Negative axes count from the end. With
/// `None`, the input is flattened in C order and a one-dimensional result is
/// returned.
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
/// With `Some(axis)`, indices are local to that axis and the result has the
/// same shape as `a`. Negative axes count from the end. With `None`, indices
/// refer to the C-order flattened input and the result is one-dimensional.
pub fn argsort<T: SortElement>(
    a: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<i64>> {
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
    let axis_len = shape[axis];
    if axis_len < 2 {
        return;
    }
    let inner: usize = shape[axis + 1..].iter().product();
    if inner == 1 {
        for chunk in data.chunks_exact_mut(axis_len) {
            chunk.sort_by(T::sort_cmp);
        }
        return;
    }
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
