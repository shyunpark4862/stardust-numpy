use crate::array::Array;
use crate::error::{Error, Result};

/// Evenly spaced `i64` values in the half-open interval `[start, stop)`.
///
/// `step` must be non-zero.
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
