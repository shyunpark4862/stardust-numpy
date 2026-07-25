//! Array creation helpers (`zeros`, `ones`, `full`, `arange`, `eye`, …).

mod factories;
mod grids;
mod ranges;
mod triangular;

pub use factories::{full, ones, zeros};
pub use grids::{meshgrid, MeshgridIndexing};
pub use ranges::{arange, arange_stop, geomspace, linspace, logspace};
pub use triangular::{diag, eye, eye_with, tri, tri_with, tril, triu};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::dtype::Complex64;
    use approx::assert_relative_eq;
    use num_complex::Complex;

    #[test]
    fn full_zeros_ones() {
        let z = zeros::<i64>(&[2, 3]).unwrap();
        assert_eq!(z.shape(), &[2, 3]);
        assert_eq!(z.get(&[1, 2]).unwrap(), 0);

        let o = ones::<f64>(&[2]).unwrap();
        assert_eq!(o.get(&[0]).unwrap(), 1.0);

        let f = full(&[2, 2], true).unwrap();
        assert!(f.get(&[0, 1]).unwrap());
    }

    #[test]
    fn zeros_complex() {
        let z = zeros::<Complex64>(&[2]).unwrap();
        assert_eq!(z.get(&[0]).unwrap(), Complex::new(0.0, 0.0));
    }

    #[test]
    fn arange_basic() {
        let a = arange_stop(3).unwrap();
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.get(&[0]).unwrap(), 0);
        assert_eq!(a.get(&[2]).unwrap(), 2);

        let b = arange(1, 5, 2).unwrap();
        assert_eq!(b.get(&[0]).unwrap(), 1);
        assert_eq!(b.get(&[1]).unwrap(), 3);
        assert_eq!(b.size(), 2);

        let c = arange(5, 1, -2).unwrap();
        assert_eq!(c.get(&[0]).unwrap(), 5);
        assert_eq!(c.get(&[1]).unwrap(), 3);
    }

    #[test]
    fn spaces_endpoints_and_small_counts() {
        assert_eq!(linspace(0.0, 1.0, 0, true).unwrap().shape(), &[0]);
        assert_eq!(linspace(2.0, 8.0, 1, false).unwrap().to_vec(), vec![2.0]);
        assert_eq!(
            linspace(0.0, 1.0, 5, true).unwrap().to_vec(),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );
        assert_eq!(
            linspace(0.0, 1.0, 4, false).unwrap().to_vec(),
            vec![0.0, 0.25, 0.5, 0.75]
        );
    }

    #[test]
    fn logarithmic_spaces() {
        assert_eq!(
            logspace(0.0, 2.0, 3, true, 10.0).unwrap().to_vec(),
            vec![1.0, 10.0, 100.0]
        );
        let positive = geomspace(1.0, 16.0, 5, true).unwrap().to_vec();
        let negative = geomspace(-1.0, -16.0, 5, true).unwrap().to_vec();
        for (actual, expected) in
            positive.iter().zip([1.0, 2.0, 4.0, 8.0, 16.0])
        {
            assert_relative_eq!(actual, &expected);
        }
        for (actual, expected) in
            negative.iter().zip([-1.0, -2.0, -4.0, -8.0, -16.0])
        {
            assert_relative_eq!(actual, &expected);
        }
    }

    #[test]
    fn meshgrid_indexing_and_views() {
        let x = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
        let y = Array::from_slice(&[10_i64, 20], &[2]).unwrap();

        let xy = meshgrid(&[&x, &y], MeshgridIndexing::Xy).unwrap();
        assert_eq!(xy[0].shape(), &[2, 3]);
        assert_eq!(xy[1].shape(), &[2, 3]);
        assert_eq!(xy[0].get(&[1, 2]).unwrap(), 3);
        assert_eq!(xy[1].get(&[1, 2]).unwrap(), 20);
        assert!(xy[0].shares_buffer_with(&x));
        assert!(!xy[0].is_writable());

        let ij = meshgrid(&[&x, &y], MeshgridIndexing::Ij).unwrap();
        assert_eq!(ij[0].shape(), &[3, 2]);
        assert_eq!(ij[1].shape(), &[3, 2]);
        assert_eq!(ij[0].get(&[2, 1]).unwrap(), 3);
        assert_eq!(ij[1].get(&[2, 1]).unwrap(), 20);
    }

    #[test]
    fn meshgrid_empty_single() {
        let none: Vec<&Array<i64>> = Vec::new();
        assert!(meshgrid(&none, MeshgridIndexing::Xy).unwrap().is_empty());

        let x = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
        let one = meshgrid(&[&x], MeshgridIndexing::Xy).unwrap();
        assert_eq!(one[0].shape(), &[2]);
    }

    #[test]
    fn eye_square() {
        let e = eye::<i64>(3).unwrap();
        assert_eq!(e.shape(), &[3, 3]);
        assert_eq!(e.get(&[0, 0]).unwrap(), 1);
        assert_eq!(e.get(&[0, 1]).unwrap(), 0);
        assert_eq!(e.get(&[2, 2]).unwrap(), 1);
    }

    #[test]
    fn eye_with_offset() {
        let e = eye_with::<i64>(3, 3, 1).unwrap();
        assert_eq!(e.get(&[0, 1]).unwrap(), 1);
        assert_eq!(e.get(&[0, 0]).unwrap(), 0);
        assert_eq!(e.get(&[1, 2]).unwrap(), 1);
    }
}
