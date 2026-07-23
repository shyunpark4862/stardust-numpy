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
/// `step` must be non-zero. Triangle helpers (`tri` / `tril` / `triu`) are
/// deferred (see `plan.md` Phase 6).
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

/// Return `num` evenly spaced values from `start` toward `stop`.
///
/// When `endpoint` is `true`, a non-empty result ends at `stop`; otherwise
/// the samples cover the half-open interval `[start, stop)`. `num == 0`
/// returns an empty array, and `num == 1` returns `[start]`. Both bounds must
/// be finite.
pub fn linspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<Array<f64>> {
    validate_finite_bounds("linspace", start, stop)?;
    let values = linear_values(start, stop, num, endpoint);
    Array::from_vec(values, &[num])
}

/// Return numbers spaced evenly on a logarithmic scale.
///
/// The exponents are the values from
/// `linspace(start, stop, num, endpoint)` and each result is `base.powf(x)`.
/// `base` must be finite and strictly positive. `num == 0` returns an empty
/// array, while `num == 1` returns `[base.powf(start)]`.
pub fn logspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    base: f64,
) -> Result<Array<f64>> {
    validate_finite_bounds("logspace", start, stop)?;
    if !base.is_finite() || base <= 0.0 {
        return Err(Error::InvalidArgument(
            "logspace base must be finite and greater than zero".into(),
        ));
    }

    let values = linear_values(start, stop, num, endpoint)
        .into_iter()
        .map(|exponent| base.powf(exponent))
        .collect();
    Array::from_vec(values, &[num])
}

/// Return `num` values evenly spaced on a geometric scale.
///
/// `start` and `stop` must be finite, non-zero, and have the same sign.
/// Negative bounds are supported; their magnitudes are interpolated in log
/// space and the negative sign is restored. When `endpoint` is `true`, a
/// non-empty result ends exactly at `stop`. `num == 0` returns an empty array,
/// and `num == 1` returns `[start]`.
pub fn geomspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<Array<f64>> {
    validate_finite_bounds("geomspace", start, stop)?;
    if start == 0.0 || stop == 0.0 {
        return Err(Error::InvalidArgument(
            "geomspace bounds must not be zero".into(),
        ));
    }
    if start.is_sign_negative() != stop.is_sign_negative() {
        return Err(Error::InvalidArgument(
            "geomspace bounds must have the same sign".into(),
        ));
    }

    let negative = start.is_sign_negative();
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();
    let mut values: Vec<f64> =
        linear_values(log_start, log_stop, num, endpoint)
            .into_iter()
            .map(|value| {
                let magnitude = value.exp();
                if negative {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect();

    if let Some(first) = values.first_mut() {
        *first = start;
    }
    if endpoint && num > 1 {
        values[num - 1] = stop;
    }
    Array::from_vec(values, &[num])
}

fn validate_finite_bounds(function: &str, start: f64, stop: f64) -> Result<()> {
    if !start.is_finite() || !stop.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "{function} bounds must be finite"
        )));
    }
    Ok(())
}

fn linear_values(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }

    let divisor = if endpoint { num - 1 } else { num } as f64;
    let mut values = Vec::with_capacity(num);
    for index in 0..num {
        let fraction = index as f64 / divisor;
        let value = if start.is_sign_negative() != stop.is_sign_negative() {
            // Avoid overflowing `stop - start` for large opposite-sign bounds.
            start * (1.0 - fraction) + stop * fraction
        } else {
            // Same-sign subtraction cannot overflow, and this form avoids a
            // potentially overflowing sum of two rounded weighted terms.
            start + (stop - start) * fraction
        };
        values.push(value);
    }
    values[0] = start;
    if endpoint {
        values[num - 1] = stop;
    }
    values
}

/// Axis ordering used by [`meshgrid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshgridIndexing {
    /// Cartesian (`xy`) indexing: swap the first two output dimensions.
    Xy,
    /// Matrix (`ij`) indexing: preserve input dimension order.
    Ij,
}

/// Build coordinate grids from same-dtype one-dimensional input arrays.
///
/// One output is returned for each input. Every output has the common shape
/// formed from the input lengths; [`MeshgridIndexing::Xy`] swaps the first two
/// dimensions when at least two arrays are supplied, while
/// [`MeshgridIndexing::Ij`] preserves their order. Outputs are read-only
/// broadcast views; contiguous inputs share their corresponding input buffer,
/// while a non-contiguous input may first be copied by reshape. An empty input
/// slice returns an empty vector.
pub fn meshgrid<T: Scalar>(
    arrays: &[&Array<T>],
    indexing: MeshgridIndexing,
) -> Result<Vec<Array<T>>> {
    for array in arrays {
        if array.ndim() != 1 {
            return Err(Error::InvalidArgument(format!(
                "meshgrid inputs must be 1-D, got ndim={}",
                array.ndim()
            )));
        }
    }

    let ndim = arrays.len();
    let mut output_shape: Vec<usize> =
        arrays.iter().map(|array| array.shape()[0]).collect();
    if indexing == MeshgridIndexing::Xy && ndim >= 2 {
        output_shape.swap(0, 1);
    }

    arrays
        .iter()
        .enumerate()
        .map(|(input_axis, array)| {
            let output_axis = match (indexing, input_axis) {
                (MeshgridIndexing::Xy, 0) if ndim >= 2 => 1,
                (MeshgridIndexing::Xy, 1) if ndim >= 2 => 0,
                _ => input_axis,
            };
            let mut reshape = vec![1_isize; ndim];
            reshape[output_axis] =
                isize::try_from(array.shape()[0]).map_err(|_| {
                    Error::InvalidArgument(
                        "meshgrid input length exceeds isize::MAX".into(),
                    )
                })?;
            array.reshape(&reshape)?.broadcast_to(&output_shape)
        })
        .collect()
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
    fn arange_step_zero_err() {
        assert!(arange(0, 3, 0).is_err());
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
        assert!(geomspace(0.0, 1.0, 0, true).is_err());
        assert!(geomspace(-1.0, 1.0, 5, true).is_err());
        assert!(logspace(0.0, 1.0, 5, true, 0.0).is_err());
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
    fn meshgrid_empty_single_and_invalid_rank() {
        let none: Vec<&Array<i64>> = Vec::new();
        assert!(meshgrid(&none, MeshgridIndexing::Xy).unwrap().is_empty());

        let x = Array::from_slice(&[1_i64, 2], &[2]).unwrap();
        let one = meshgrid(&[&x], MeshgridIndexing::Xy).unwrap();
        assert_eq!(one[0].shape(), &[2]);

        let matrix = Array::from_slice(&[1_i64, 2], &[1, 2]).unwrap();
        assert!(meshgrid(&[&matrix], MeshgridIndexing::Ij).is_err());
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
