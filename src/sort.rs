//! Sorting and unique-value operations.
//!
//! Sorting is stable and uses NumPy-like axis semantics. Floating-point NaNs
//! compare after all non-NaN values. Unique-value operations always flatten
//! their input in C order before collecting and sorting values.

use std::cmp::Ordering;

use crate::dtype::{Complex64, Scalar};
use crate::error::{Error, Result};
use crate::Array;

mod sealed {
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}

/// Element types accepted by [`sort`], [`sort_in_place`], and [`argsort`].
///
/// This trait is sealed; the supported types are `bool`, `i64`, and `f64`.
/// In particular, complex arrays intentionally do not implement general
/// sorting.
pub trait SortElement: Scalar + sealed::Sealed {
    /// Compare two values using the ordering required by sorting operations.
    fn sort_cmp(&self, other: &Self) -> Ordering;
}

impl SortElement for bool {
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl SortElement for i64 {
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl SortElement for f64 {
    #[inline]
    fn sort_cmp(&self, other: &Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (false, false) => {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (true, true) => Ordering::Equal,
        }
    }
}

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
            let axis = normalize_axis(axis, a.ndim())?;
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

/// Sort `a` in place from the caller's perspective.
///
/// The replacement array is C-contiguous and writable, which also preserves
/// copy-on-write isolation from arrays sharing the input's original buffer.
/// With `None`, all elements are sorted together while `a` keeps its shape.
pub fn sort_in_place<T: SortElement>(
    a: &mut Array<T>,
    axis: Option<isize>,
) -> Result<()> {
    if !a.is_writable() {
        return Err(Error::ReadOnly);
    }
    let shape = a.shape().to_vec();
    let mut sorted = sort(a, axis)?;
    if axis.is_none() {
        sorted = Array::from_vec(sorted.to_vec(), &shape)?;
    }
    *a = sorted;
    Ok(())
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
            let axis = normalize_axis(axis, a.ndim())?;
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

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    if ndim == 0 {
        return Err(Error::InvalidArgument(
            "axis is invalid for a 0-D array".into(),
        ));
    }
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(Error::InvalidArgument(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
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

mod unique_sealed {
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
    impl Sealed for crate::dtype::Complex64 {}
}

/// Element types accepted by [`unique`] and [`unique_with`].
///
/// This trait is sealed; the supported types are `bool`, `i64`, `f64`, and
/// [`Complex64`].
pub trait UniqueElement: Scalar + unique_sealed::Sealed {
    /// Compare values for their position in sorted unique output.
    fn unique_cmp(&self, other: &Self) -> Ordering;

    /// Return whether values belong to the same unique-value group.
    fn unique_eq(&self, other: &Self) -> bool;
}

impl UniqueElement for bool {
    #[inline]
    fn unique_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for i64 {
    #[inline]
    fn unique_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for f64 {
    #[inline]
    fn unique_cmp(&self, other: &Self) -> Ordering {
        self.sort_cmp(other)
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other || (self.is_nan() && other.is_nan())
    }
}

impl UniqueElement for Complex64 {
    fn unique_cmp(&self, other: &Self) -> Ordering {
        match (complex_is_nan(*self), complex_is_nan(*other)) {
            (false, false) => self
                .re
                .sort_cmp(&other.re)
                .then_with(|| self.im.sort_cmp(&other.im)),
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (true, true) => Ordering::Equal,
        }
    }

    #[inline]
    fn unique_eq(&self, other: &Self) -> bool {
        self == other || (complex_is_nan(*self) && complex_is_nan(*other))
    }
}

#[inline]
fn complex_is_nan(value: Complex64) -> bool {
    value.re.is_nan() || value.im.is_nan()
}

/// Optional outputs requested from [`unique_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniqueOptions {
    /// Include each unique value's first flat input index.
    pub return_index: bool,
    /// Include a map from each flat input element to its unique value.
    pub return_inverse: bool,
    /// Include the number of occurrences of each unique value.
    pub return_counts: bool,
}

/// Result of [`unique_with`].
#[derive(Clone, Debug)]
pub struct UniqueResult<T: Scalar> {
    /// Sorted unique values.
    pub values: Array<T>,
    /// First C-order input indices, when requested.
    pub indices: Option<Array<i64>>,
    /// Unique-value index for every C-order input element, when requested.
    pub inverse_indices: Option<Array<i64>>,
    /// Occurrence count for every unique value, when requested.
    pub counts: Option<Array<i64>>,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_axes_and_flatten() {
        let a = Array::from_slice(&[3_i64, 1, 2, 6, 4, 5], &[2, 3]).unwrap();
        assert_eq!(sort(&a, Some(-1)).unwrap().to_vec(), [1, 2, 3, 4, 5, 6]);
        assert_eq!(sort(&a, Some(0)).unwrap().to_vec(), [3, 1, 2, 6, 4, 5]);
        let flat = sort(&a, None).unwrap();
        assert_eq!(flat.shape(), &[6]);
        assert_eq!(flat.to_vec(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn argsort_is_stable_and_nan_last() {
        let a = Array::from_slice(&[2.0, f64::NAN, 1.0, 2.0], &[4]).unwrap();
        assert_eq!(argsort(&a, None).unwrap().to_vec(), [2, 0, 3, 1]);
    }

    #[test]
    fn unique_metadata_merges_nan() {
        let a = Array::from_slice(&[2.0, f64::NAN, 1.0, 2.0, f64::NAN], &[5])
            .unwrap();
        let result = unique_with(
            &a,
            UniqueOptions {
                return_index: true,
                return_inverse: true,
                return_counts: true,
            },
        )
        .unwrap();
        let values = result.values.to_vec();
        assert_eq!(&values[..2], &[1.0, 2.0]);
        assert!(values[2].is_nan());
        assert_eq!(result.indices.unwrap().to_vec(), [2, 0, 1]);
        assert_eq!(result.inverse_indices.unwrap().to_vec(), [1, 2, 0, 1, 2]);
        assert_eq!(result.counts.unwrap().to_vec(), [1, 2, 2]);
    }

    #[test]
    fn in_place_replacement_breaks_sharing_and_keeps_shape() {
        let mut a = Array::from_slice(&[3_i64, 1, 2, 0], &[2, 2]).unwrap();
        let shared = a.clone();
        sort_in_place(&mut a, None).unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.to_vec(), [0, 1, 2, 3]);
        assert_eq!(shared.to_vec(), [3, 1, 2, 0]);
        assert!(!a.shares_buffer_with(&shared));
    }
}
