use crate::array::Array;
use crate::error::Result;

/// Evenly spaced `i64` values in the half-open interval `[start, stop)`.
pub fn arange(start: i64, stop: i64, step: i64) -> Result<Array<i64>> {
    let len = arange_length(start, stop, step);
    let mut values = Vec::with_capacity(len);
    let start = i128::from(start);
    let step = i128::from(step);
    for index in 0..len {
        let value = start + index as i128 * step;
        values.push(value as i64);
    }
    Array::from_vec(values, &[len])
}

fn arange_length(start: i64, stop: i64, step: i64) -> usize {
    debug_assert!(step != 0, "arange step must not be zero");
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
    usize::try_from(len).expect("arange length overflows usize")
}

/// `arange(0, stop, 1)` — NumPy's single-argument `np.arange(n)`.
pub fn arange_stop(stop: i64) -> Result<Array<i64>> {
    arange(0, stop, 1)
}

/// Return `num` evenly spaced values from `start` toward `stop`.
pub fn linspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<Array<f64>> {
    let values = linear_values(start, stop, num, endpoint);
    Array::from_vec(values, &[num])
}

/// Return numbers spaced evenly on a logarithmic scale.
pub fn logspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    base: f64,
) -> Result<Array<f64>> {
    let values = linear_values(start, stop, num, endpoint)
        .into_iter()
        .map(|exponent| base.powf(exponent))
        .collect();
    Array::from_vec(values, &[num])
}

/// Return `num` values evenly spaced on a geometric scale.
pub fn geomspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<Array<f64>> {
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
