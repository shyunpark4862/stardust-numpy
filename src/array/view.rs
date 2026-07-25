//! View operations: transpose, reshape, and cheap aliases.

use std::sync::Arc;

use crate::array::Array;
use crate::axis::{normalize_axis_list, normalize_insert_axis};
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::{c_order_strides, size_of_shape};

/// Insert a size-one axis without copying the backing buffer.
pub(crate) fn insert_axis_view<T: Scalar>(
    array: &Array<T>,
    axis: isize,
) -> Result<Array<T>> {
    let axis = normalize_insert_axis(axis, array.ndim());
    let mut shape = array.shape().to_vec();
    let mut strides = array.strides().to_vec();
    shape.insert(axis, 1);
    strides.insert(axis, 0);
    Array::from_shared_parts(
        Arc::clone(&array.data),
        shape,
        strides,
        array.offset(),
        array.is_writable(),
    )
}

impl<T: Scalar> Array<T> {
    /// Return a view with axes reversed (matrix transpose for 2-D).
    ///
    /// Shares the backing buffer. For 0-D and 1-D, returns a view with the
    /// same layout.
    pub fn transpose(&self) -> Array<T> {
        if self.ndim() <= 1 {
            return self.view();
        }
        let shape: Vec<usize> = self.shape.iter().rev().copied().collect();
        let strides: Vec<isize> = self.strides.iter().rev().copied().collect();
        Self::from_shared_parts(
            Arc::clone(&self.data),
            shape,
            strides,
            self.offset,
            self.writable,
        )
        .expect("transpose preserves shape/strides rank")
    }

    /// Alias for [`Array::transpose`].
    #[inline]
    pub fn t(&self) -> Array<T> {
        self.transpose()
    }

    /// Return a view with axes permuted by `axes`.
    ///
    /// `axes` must be a permutation of `0..ndim`; negative axes count from
    /// the end.
    pub fn permute_axes(&self, axes: &[isize]) -> Result<Array<T>> {
        debug_assert_eq!(axes.len(), self.ndim());
        let axes = normalize_axis_list(axes, self.ndim());
        let shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let strides: Vec<isize> =
            axes.iter().map(|&a| self.strides[a]).collect();
        Self::from_shared_parts(
            Arc::clone(&self.data),
            shape,
            strides,
            self.offset,
            self.writable,
        )
    }

    /// Reshape the array.
    ///
    /// One dimension may be `-1` to infer size. Returns a view when this
    /// array is C-contiguous; otherwise returns a contiguous copy with the
    /// new shape.
    pub fn reshape(&self, shape: &[isize]) -> Result<Array<T>> {
        let new_shape = resolve_reshape(shape, self.size());
        if self.is_c_contiguous() {
            let strides = c_order_strides(&new_shape);
            Self::from_shared_parts(
                Arc::clone(&self.data),
                new_shape,
                strides,
                self.offset,
                self.writable,
            )
        } else {
            let data = self.to_vec_c_order();
            Array::from_vec(data, &new_shape)
        }
    }

    /// Remove length-one axes as a zero-copy view.
    ///
    /// `None` removes every length-one axis. `Some(axes)` removes only the
    /// requested axes; negative axes count from the end. Requested axes must
    /// be unique and have length one. Removing every axis yields a valid 0-D
    /// array.
    pub fn squeeze(&self, axes: Option<&[isize]>) -> Result<Array<T>> {
        let mut remove = vec![false; self.ndim()];
        match axes {
            None => {
                for (axis, &length) in self.shape.iter().enumerate() {
                    remove[axis] = length == 1;
                }
            }
            Some(axes) => {
                for axis in normalize_axis_list(axes, self.ndim()) {
                    debug_assert_eq!(self.shape[axis], 1);
                    remove[axis] = true;
                }
            }
        }

        let shape = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(axis, &length)| (!remove[axis]).then_some(length))
            .collect();
        let strides = self
            .strides
            .iter()
            .enumerate()
            .filter_map(|(axis, &stride)| (!remove[axis]).then_some(stride))
            .collect();
        Self::from_shared_parts(
            Arc::clone(&self.data),
            shape,
            strides,
            self.offset,
            self.writable,
        )
    }

    /// Cheap view alias: same buffer, shape, strides, and offset.
    pub fn view(&self) -> Array<T> {
        Self::from_shared_parts(
            Arc::clone(&self.data),
            self.shape.clone(),
            self.strides.clone(),
            self.offset,
            self.writable,
        )
        .expect("view preserves shape/strides rank")
    }
}

fn resolve_reshape(shape: &[isize], size: usize) -> Vec<usize> {
    let mut inferred = None;
    let mut known = 1_usize;
    let mut out = Vec::with_capacity(shape.len());

    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            debug_assert!(
                inferred.is_none(),
                "only one reshape dimension may be -1"
            );
            inferred = Some(i);
            out.push(0);
        } else {
            debug_assert!(d >= 0, "invalid reshape dimension {d}");
            let d = d as usize;
            known *= d;
            out.push(d);
        }
    }

    if let Some(idx) = inferred {
        debug_assert!(
            known != 0,
            "cannot infer reshape dimension when another is 0"
        );
        debug_assert_eq!(
            size % known,
            0,
            "cannot reshape array of size {size} into shape {shape:?}"
        );
        out[idx] = size / known;
    } else {
        debug_assert_eq!(
            size_of_shape(&out),
            size,
            "cannot reshape array of size {size} into shape {shape:?}"
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_shares_buffer() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let b = a.transpose();
        assert!(a.shares_buffer_with(&b));
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.strides(), &[1, 3]);
        assert_eq!(b.get(&[0, 1]).unwrap(), 4); // was a[1,0]
        assert_eq!(b.get(&[2, 1]).unwrap(), 6); // was a[1,2]
    }

    #[test]
    fn reshape_view_when_contiguous() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let b = a.reshape(&[3, 2]).unwrap();
        assert!(a.shares_buffer_with(&b));
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(&[1, 0]).unwrap(), 3);
    }

    #[test]
    fn reshape_infers_minus_one() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let b = a.reshape(&[-1, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn reshape_copies_when_not_contiguous() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let t = a.transpose();
        assert!(!t.is_c_contiguous());
        let b = t.reshape(&[6]).unwrap();
        assert!(!t.shares_buffer_with(&b));
        assert_eq!(b.shape(), &[6]);
        assert_eq!(b.get(&[0]).unwrap(), 1);
        assert_eq!(b.get(&[1]).unwrap(), 4);
    }

    #[test]
    fn permute_axes() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let b = a.permute_axes(&[1, 0]).unwrap();
        assert_eq!(b.shape(), a.transpose().shape());
        assert_eq!(b.get(&[2, 0]).unwrap(), a.get(&[0, 2]).unwrap());

        let negative = a.permute_axes(&[-1, -2]).unwrap();
        assert_eq!(negative.shape(), b.shape());
    }

    #[test]
    fn squeeze_all_and_selected_axes_share_storage() {
        let a = Array::from_slice(&[1_i64, 2, 3], &[1, 3, 1]).unwrap();
        let all = a.squeeze(None).unwrap();
        assert_eq!(all.shape(), &[3]);
        assert_eq!(all.to_vec(), [1, 2, 3]);
        assert!(all.shares_buffer_with(&a));

        let selected = a.squeeze(Some(&[0])).unwrap();
        assert_eq!(selected.shape(), &[3, 1]);
        assert!(selected.shares_buffer_with(&a));

        let negative = a.squeeze(Some(&[-1])).unwrap();
        assert_eq!(negative.shape(), &[1, 3]);
    }

    #[test]
    fn squeeze_allows_zero_dimensional_results() {
        let a = Array::from_slice(&[7_i64], &[1]).unwrap();
        let scalar = a.squeeze(None).unwrap();
        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.item().unwrap(), 7);

        let zero_dimensional = Array::from_slice(&[9_i64], &[]).unwrap();
        assert!(zero_dimensional.squeeze(None).unwrap().shape().is_empty());
    }

    #[test]
    fn squeeze_preserves_empty_broadcast_and_copy_on_write_semantics() {
        let empty = Array::from_slice(&[] as &[i64], &[0, 1]).unwrap();
        assert_eq!(empty.squeeze(None).unwrap().shape(), &[0]);

        let base = Array::from_slice(&[1_i64, 2, 3], &[1, 3, 1]).unwrap();
        let broadcast = base.broadcast_to(&[2, 3, 1]).unwrap();
        let squeezed = broadcast.squeeze(None).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3]);
        assert!(!squeezed.is_writable());
        assert!(squeezed.shares_buffer_with(&base));

        let mut writable = base.squeeze(None).unwrap();
        writable.set(&[0], 9).unwrap();
        assert_eq!(writable.get(&[0]).unwrap(), 9);
        assert_eq!(base.get(&[0, 0, 0]).unwrap(), 1);
        assert!(!writable.shares_buffer_with(&base));
    }
}
