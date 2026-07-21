//! Array creation helpers (`zeros`, `ones`, `full`, `arange`, `eye`, …).

use num_traits::{One, Zero};

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::shape::size_of_shape;

/// Return a C-contiguous array filled with `value`.
pub fn full<T: Scalar>(shape: &[usize], value: T) -> Result<Array<T>> {
    let size = size_of_shape(shape);
    Array::from_vec(vec![value; size], shape)
}

/// Return a C-contiguous array of zeros.
pub fn zeros<T: Scalar + Zero>(shape: &[usize]) -> Result<Array<T>> {
    full(shape, T::zero())
}

/// Return a C-contiguous array of ones.
pub fn ones<T: Scalar + One>(shape: &[usize]) -> Result<Array<T>> {
    full(shape, T::one())
}

/// Evenly spaced `i64` values in the half-open interval `[start, stop)`.
///
/// `step` must be non-zero. Floating-point spacing (`linspace` / `logspace` /
/// `geomspace`) and triangle helpers (`tri` / `tril` / `triu`) are deferred
/// (see `plan.md` Phase 5–6).
pub fn arange(start: i64, stop: i64, step: i64) -> Result<Array<i64>> {
    if step == 0 {
        return Err(Error::InvalidArgument(
            "arange step must not be zero".into(),
        ));
    }

    let mut values = Vec::new();
    let mut current = start;
    if step > 0 {
        while current < stop {
            values.push(current);
            current += step;
        }
    } else {
        while current > stop {
            values.push(current);
            current += step;
        }
    }

    let len = values.len();
    Array::from_vec(values, &[len])
}

/// `arange(0, stop, 1)` — NumPy's single-argument `np.arange(n)`.
pub fn arange_stop(stop: i64) -> Result<Array<i64>> {
    arange(0, stop, 1)
}

/// 2-D identity-like matrix with ones on diagonal `k` (default main diagonal).
pub fn eye<T: Scalar + Zero + One>(n: usize) -> Result<Array<T>> {
    eye_with(n, n, 0)
}

/// Like [`eye`], with optional column count `m` and diagonal offset `k`.
///
/// `k > 0` is above the main diagonal; `k < 0` is below.
pub fn eye_with<T: Scalar + Zero + One>(
    n: usize,
    m: usize,
    k: isize,
) -> Result<Array<T>> {
    let mut data = vec![T::zero(); n * m];
    for i in 0..n {
        let j = i as isize + k;
        if j >= 0 && (j as usize) < m {
            data[i * m + j as usize] = T::one();
        }
    }
    Array::from_vec(data, &[n, m])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Complex64;
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
    fn arange_step_zero_err() {
        assert!(arange(0, 3, 0).is_err());
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
