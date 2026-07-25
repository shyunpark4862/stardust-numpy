//! View operations: transpose, reshape, and cheap aliases.

use std::sync::Arc;

use crate::array::Array;
use crate::axis::{normalize_axis_list, normalize_insert_axis};
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::shape::{c_order_strides, size_of_shape};

/// Insert a size-one axis without copying the backing buffer.
pub(crate) fn insert_axis_view<T: Scalar>(
    array: &Array<T>,
    axis: isize,
) -> Result<Array<T>> {
    let axis = normalize_insert_axis(axis, array.ndim())?;
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
        if axes.len() != self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "axes length {} does not match ndim {}",
                axes.len(),
                self.ndim()
            )));
        }
        let axes =
            normalize_axis_list(axes, self.ndim()).map_err(
                |error| match error {
                    Error::InvalidArgument(_) => Error::InvalidArgument(
                        "axes must be a permutation of 0..ndim-1".into(),
                    ),
                    other => other,
                },
            )?;
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
        let new_shape = resolve_reshape(shape, self.size())?;
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

fn resolve_reshape(shape: &[isize], size: usize) -> Result<Vec<usize>> {
    let mut inferred = None;
    let mut known = 1_usize;
    let mut out = Vec::with_capacity(shape.len());

    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if inferred.is_some() {
                return Err(Error::InvalidArgument(
                    "only one reshape dimension may be -1".into(),
                ));
            }
            inferred = Some(i);
            out.push(0);
        } else if d < 0 {
            return Err(Error::InvalidArgument(format!(
                "invalid reshape dimension {d}"
            )));
        } else {
            let d = d as usize;
            known *= d;
            out.push(d);
        }
    }

    if let Some(idx) = inferred {
        if known == 0 {
            return Err(Error::InvalidArgument(
                "cannot infer reshape dimension when another is 0".into(),
            ));
        }
        if size % known != 0 {
            return Err(Error::InvalidArgument(format!(
                "cannot reshape array of size {size} into shape {shape:?}"
            )));
        }
        out[idx] = size / known;
    } else if size_of_shape(&out) != size {
        return Err(Error::InvalidArgument(format!(
            "cannot reshape array of size {size} into shape {shape:?}"
        )));
    }

    Ok(out)
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
        assert!(a.permute_axes(&[0, -2]).is_err());
    }
}
