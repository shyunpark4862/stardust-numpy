//! Sorting and unique-value operations.
//!
//! Sorting is stable and uses NumPy-like axis semantics. Floating-point NaNs
//! compare after all non-NaN values. Unique-value operations always flatten
//! their input in C order before collecting and sorting values.

mod sort;
mod traits_options;
mod unique;

pub use sort::{argsort, sort};
pub use traits_options::{
    SortElement, UniqueElement, UniqueOptions, UniqueResult,
};
pub use unique::{unique, unique_with};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

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
}
