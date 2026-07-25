//! NumPy-style array string formatting.

use sdnp::Complex64;

use crate::inner::ArrayInner;

const MAX_ITEMS: usize = 1000;
const EDGE_ITEMS: usize = 3;

pub fn array_repr(inner: &ArrayInner) -> pyo3::PyResult<String> {
    let dtype = inner.dtype().name();
    let body = format_body(inner, inner.shape(), 0)?;
    Ok(format!("array({body}, dtype={dtype})"))
}

fn format_body(
    inner: &ArrayInner,
    shape: &[usize],
    depth: usize,
) -> pyo3::PyResult<String> {
    if shape.is_empty() {
        return Ok(format_scalar(inner.item_scalar()?));
    }
    if shape.len() == 1 {
        return format_1d(inner);
    }
    let n = shape[0];
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let sub = slice_index(inner, i)?;
        parts.push(format_body(&sub, &shape[1..], depth + 1)?);
    }
    Ok(format!(
        "[{}]",
        parts.join(",\n ".repeat(depth + 1).trim_end())
    ))
}

fn format_1d(inner: &ArrayInner) -> pyo3::PyResult<String> {
    let n = inner.size();
    if n <= MAX_ITEMS {
        return Ok(format!("[{}]", format_elements(inner, 0, n)?.join(", ")));
    }
    let mut parts = format_elements(inner, 0, EDGE_ITEMS)?;
    parts.push("...".to_string());
    parts.extend(format_elements(inner, n - EDGE_ITEMS, n)?);
    Ok(format!("[{}]", parts.join(", ")))
}

fn format_elements(
    inner: &ArrayInner,
    start: usize,
    stop: usize,
) -> pyo3::PyResult<Vec<String>> {
    let mut out = Vec::new();
    match inner {
        ArrayInner::Bool(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(if *v { "True".into() } else { "False".into() });
            }
        }
        ArrayInner::I64(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(v.to_string());
            }
        }
        ArrayInner::F64(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(format_float(*v));
            }
        }
        ArrayInner::C64(a) => {
            let flat: Vec<_> = a.flat().collect();
            for v in &flat[start..stop] {
                out.push(format_complex(*v));
            }
        }
    }
    Ok(out)
}

fn slice_index(inner: &ArrayInner, i: usize) -> pyo3::PyResult<ArrayInner> {
    use sdnp::{gather, IndexSpec};

    use crate::error::map_sdnp;
    let spec = vec![IndexSpec::Index(i as i64)];
    Ok(match inner {
        ArrayInner::Bool(a) => ArrayInner::Bool(map_sdnp(gather(a, &spec))?),
        ArrayInner::I64(a) => ArrayInner::I64(map_sdnp(gather(a, &spec))?),
        ArrayInner::F64(a) => ArrayInner::F64(map_sdnp(gather(a, &spec))?),
        ArrayInner::C64(a) => ArrayInner::C64(map_sdnp(gather(a, &spec))?),
    })
}

fn format_scalar(s: crate::unwrap::PyScalar) -> String {
    use crate::unwrap::PyScalar;
    match s {
        PyScalar::Bool(v) => {
            if v {
                "True".into()
            } else {
                "False".into()
            }
        }
        PyScalar::I64(v) => v.to_string(),
        PyScalar::F64(v) => format_float(v),
        PyScalar::C64(v) => format_complex(v),
    }
}

fn format_float(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "inf".into()
        } else {
            "-inf".into()
        }
    } else {
        v.to_string()
    }
}

fn format_complex(v: Complex64) -> String {
    if v.im == 0.0 {
        format!("({}+0j)", format_float(v.re))
    } else if v.im.is_sign_positive() {
        format!("({}+{}j)", format_float(v.re), format_float(v.im))
    } else {
        format!("({}{}j)", format_float(v.re), format_float(v.im))
    }
}
