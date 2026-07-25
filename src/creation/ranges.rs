//! Evenly and logarithmically spaced 1-D sequences.
//!
//! These functions mirror `numpy.arange`, `linspace`, `logspace`, and
//! `geomspace`. Integer [`arange`] uses widened `i128` arithmetic when
//! computing lengths and values to reduce overflow risk before casting back
//! to `i64`.

use crate::array::Array;
use crate::error::{Error, Result};
use crate::shape::checked_allocation_len;

/// Return evenly spaced `i64` values on the half-open interval `[start, stop)`.
///
/// Values are `start`, `start + step`, `start + 2 * step`, … while the
/// sequence remains inside the interval. This mirrors NumPy's `arange`.
///
/// **Step sign rules:** When `step > 0`, elements are produced while
/// `start < stop`. When `step < 0`, elements are produced while
/// `start > stop`. If the sign of `step` disagrees with the direction from
/// `start` to `stop`, the output is empty (length zero).
///
/// **Empty output:** Returns a 1-D array of length zero when no values fall
/// in the interval, including `start == stop` with any non-zero step.
///
/// # Arguments
///
/// * `start` - First value (always included when the sequence is non-empty).
/// * `stop` - Exclusive upper bound for `step > 0`, exclusive lower bound
///   for `step < 0`.
/// * `step` - Spacing between consecutive values; must not be zero.
///
/// # Returns
///
/// A 1-D [`Array<i64>`] containing the arithmetic progression.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - `step` is
///   zero, the computed length overflows `usize`, or allocation limits are
///   exceeded.
///
/// # Examples
///
/// ```rust
/// use sdnp::arange;
///
/// assert_eq!(arange(0, 5, 1).unwrap().to_vec(), vec![0, 1, 2, 3, 4]);
/// assert_eq!(arange(5, 0, -2).unwrap().to_vec(), vec![5, 3, 1]);
/// assert_eq!(arange(0, 5, -1).unwrap().to_vec(), Vec::<i64>::new());
/// ```
pub fn arange(start: i64, stop: i64, step: i64) -> Result<Array<i64>> {
    let len = arange_length(start, stop, step)?;
    checked_allocation_len::<i64>(len)?;
    let mut values = Vec::with_capacity(len);
    let start = i128::from(start);
    let step = i128::from(step);
    for index in 0..len {
        let value = start + index as i128 * step;
        values.push(value as i64);
    }
    Array::from_vec(values, &[len])
}

/// Compute the element count for an `arange` specification.
///
/// Uses `i128` intermediates so length arithmetic does not overflow `i64`
/// before the final cast to `usize`.
///
/// # Arguments
///
/// * `start` — first value of the progression
/// * `stop` — exclusive bound (direction depends on `step` sign)
/// * `step` — spacing between consecutive values; must not be zero
///
/// # Returns
///
/// The number of elements that [`arange`] would produce (may be zero).
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — `step` is zero, or computed length
///   overflows `usize`
fn arange_length(start: i64, stop: i64, step: i64) -> Result<usize> {
    if step == 0 {
        return Err(Error::InvalidArgument(
            "arange step must not be zero".into(),
        ));
    }
    let start = i128::from(start);
    let stop = i128::from(stop);
    let step = i128::from(step);
    let len = if step > 0 && start < stop {
        (stop - start - 1) / step + 1
    } else if step < 0 && start > stop {
        (start - stop - 1) / -step + 1
    } else {
        0
    };
    usize::try_from(len).map_err(|_| {
        Error::InvalidArgument("arange length overflows usize".into())
    })
}

/// Return `arange(0, stop, 1)` — NumPy's one-argument form.
///
/// Produces the half-open range `[0, stop)` with unit step.
///
/// # Arguments
///
/// * `stop` - Exclusive upper bound (see [`arange`]).
///
/// # Returns
///
/// A 1-D [`Array<i64>`] of values `0, 1, …, stop - 1` when `stop > 0`.
///
/// # Errors
///
/// Same as [`arange`]: [`Error::InvalidArgument`](crate::Error::InvalidArgument)
/// when allocation limits are exceeded.
///
/// # Examples
///
/// ```rust
/// use sdnp::arange_stop;
///
/// assert_eq!(arange_stop(4).unwrap().to_vec(), vec![0, 1, 2, 3]);
/// ```
pub fn arange_stop(stop: i64) -> Result<Array<i64>> {
    arange(0, stop, 1)
}

/// Return `num` evenly spaced floating-point values from `start` to `stop`.
///
/// Samples are placed on a linear scale. When `endpoint` is `true`, both
/// `start` and `stop` are included and the spacing is `(stop - start) /
/// (num - 1)` for `num > 1`. When `endpoint` is `false`, the interval is
/// treated as half-open toward `stop`: samples span `[start, stop)` with
/// spacing `(stop - start) / num`, and the final sample is omitted.
///
/// For `num == 0`, returns an empty vector. For `num == 1`, returns `[start]`
/// regardless of `endpoint`.
///
/// # Arguments
///
/// * `start` - Value at the first sample (always pinned exactly).
/// * `stop` - Target end value; included only when `endpoint` is `true` and
///   `num > 1`.
/// * `num` - Number of samples.
/// * `endpoint` - Whether `stop` is the last sample.
///
/// # Returns
///
/// A 1-D [`Array<f64>`] of linearly interpolated values.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - `num`
///   exceeds allocation limits.
///
/// # Examples
///
/// ```rust
/// use sdnp::linspace;
///
/// let a = linspace(0.0, 1.0, 5, true).unwrap();
/// assert_eq!(a.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
/// ```
pub fn linspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<Array<f64>> {
    checked_allocation_len::<f64>(num)?;
    let values = linear_values(start, stop, num, endpoint);
    Array::from_vec(values, &[num])
}

/// Return `num` values evenly spaced on a logarithmic scale (by exponent).
///
/// Exponents are linearly spaced between `start` and `stop`, then each sample
/// is `base.powf(exponent)`. The `endpoint` flag controls whether `stop` is
/// included as the final exponent, using the same rules as [`linspace`].
///
/// # Arguments
///
/// * `start` - Exponent of the first sample.
/// * `stop` - Exponent of the last sample when `endpoint` is `true`.
/// * `num` - Number of samples.
/// * `endpoint` - Include `stop` as the final exponent (see [`linspace`]).
/// * `base` - Base raised to each exponent.
///
/// # Returns
///
/// A 1-D [`Array<f64>`] of logarithmically spaced values.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - `num`
///   exceeds allocation limits.
///
/// # Examples
///
/// ```rust
/// use sdnp::logspace;
///
/// let a = logspace(0.0, 2.0, 3, true, 10.0).unwrap();
/// assert_eq!(a.to_vec(), vec![1.0, 10.0, 100.0]);
/// ```
pub fn logspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    base: f64,
) -> Result<Array<f64>> {
    checked_allocation_len::<f64>(num)?;
    let values = linear_values(start, stop, num, endpoint)
        .into_iter()
        .map(|exponent| base.powf(exponent))
        .collect();
    Array::from_vec(values, &[num])
}

/// Return `num` values evenly spaced on a multiplicative (geometric) scale.
///
/// Samples lie on a geometric progression from `start` to `stop`. Endpoints
/// are pinned exactly to `start` and (when `endpoint` is `true` and
/// `num > 1`) to `stop`, avoiding log/exp rounding drift. Negative endpoints
/// preserve sign on each sample.
///
/// The `endpoint` flag follows the same semantics as [`linspace`].
///
/// # Arguments
///
/// * `start` - First sample (always included exactly).
/// * `stop` - Last sample when `endpoint` is `true` and `num > 1`.
/// * `num` - Number of samples.
/// * `endpoint` - Include `stop` as the final sample (see [`linspace`]).
///
/// # Returns
///
/// A 1-D [`Array<f64>`] of geometrically spaced values.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - `num`
///   exceeds allocation limits.
///
/// # Examples
///
/// ```rust
/// use sdnp::geomspace;
///
/// let a = geomspace(1.0, 8.0, 4, true).unwrap();
/// assert_eq!(a.size(), 4);
/// assert_eq!(a.get(&[0]).unwrap(), 1.0);
/// assert_eq!(a.get(&[3]).unwrap(), 8.0);
/// ```
pub fn geomspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<Array<f64>> {
    checked_allocation_len::<f64>(num)?;
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

    // Pin endpoints exactly to avoid log/exp rounding drift.
    if let Some(first) = values.first_mut() {
        *first = start;
    }
    if endpoint && num > 1 {
        values[num - 1] = stop;
    }
    Array::from_vec(values, &[num])
}

/// Shared linear interpolation core for `linspace`, `logspace`, and
/// `geomspace`.
///
/// Endpoints are pinned exactly after interpolation to avoid floating-point
/// drift. When `endpoint` is `false`, samples span a half-open interval.
///
/// # Arguments
///
/// * `start` — value at the first sample
/// * `stop` — target end value (included only when `endpoint` is `true`
///   and `num > 1`)
/// * `num` — number of samples to generate
/// * `endpoint` — whether `stop` is the last sample
///
/// # Returns
///
/// A length-`num` vector of linearly interpolated values (possibly empty).
///
/// # Errors
///
/// Never fails; allocation failure panics like ordinary `Vec` growth.
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
        // Interpolate in magnitude when endpoints straddle zero.
        let value = if start.is_sign_negative() != stop.is_sign_negative() {
            start * (1.0 - fraction) + stop * fraction
        } else {
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
