//! Array joining and dimension-insertion operations.

mod concatenate;
mod stack;

pub use concatenate::concatenate;
pub use stack::{hstack, stack, vstack};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{insert_axis_view, Array};
    use std::sync::Arc;

    #[test]
    fn concatenate_noncontiguous_on_middle_axis() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let transposed = a.transpose();
        let joined = concatenate(&[&transposed, &transposed], 1).unwrap();
        assert_eq!(joined.shape(), &[3, 4]);
        assert_eq!(joined.to_vec(), vec![1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6]);
        assert!(joined.is_c_contiguous());
    }

    #[test]
    fn concatenate_strided_views_with_nonzero_offset() {
        let view = Array::from_shared_parts(
            Arc::new(vec![0_i64, 1, 2, 3, 4, 5, 6, 7]),
            vec![2, 2],
            vec![3, -1],
            4,
            true,
        )
        .unwrap();
        let joined = concatenate(&[&view, &view], 0).unwrap();

        assert_eq!(joined.shape(), &[4, 2]);
        assert_eq!(joined.to_vec(), vec![4, 3, 7, 6, 4, 3, 7, 6]);
    }

    #[test]
    fn stack_supports_scalars_and_negative_axis() {
        let a = Array::from_slice(&[1_i64], &[]).unwrap();
        let b = Array::from_slice(&[2_i64], &[]).unwrap();
        let joined = stack(&[&a, &b], -1).unwrap();
        assert_eq!(joined.shape(), &[2]);
        assert_eq!(joined.to_vec(), vec![1, 2]);
    }

    #[test]
    fn inserted_singleton_axis_keeps_contiguous_storage() {
        let a = Array::from_slice(&[1_i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let expanded = insert_axis_view(&a, 0).unwrap();

        assert_eq!(expanded.shape(), &[1, 2, 3]);
        assert_eq!(expanded.strides(), &[0, 3, 1]);
        assert!(expanded.is_c_contiguous());
        assert_eq!(
            expanded.as_c_contiguous_slice(),
            Some(&[1, 2, 3, 4, 5, 6][..])
        );
    }

    #[test]
    fn vertical_and_horizontal_promotion() {
        let a = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
        let b = Array::from_slice(&[3_i64, 4], &[2]).unwrap();
        let vertical = vstack(&[&a, &b]).unwrap();
        assert_eq!(vertical.shape(), &[2, 2]);
        assert_eq!(vertical.to_vec(), vec![1, 2, 3, 4]);

        let horizontal = hstack(&[&a, &b]).unwrap();
        assert_eq!(horizontal.shape(), &[4]);
        assert_eq!(horizontal.to_vec(), vec![1, 2, 3, 4]);
    }
}
