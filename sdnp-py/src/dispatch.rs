//! Binary dtype promotion and ufunc dispatch.

use pyo3::prelude::*;
use sdnp::{
    absolute, add, conj, divide, equal, greater, greater_equal, imag, isfinite,
    isinf, isnan, less, less_equal, logical_and, logical_not, logical_or,
    multiply, negative, not_equal, power, real, remainder, subtract,
    trunc_divide, Array, Complex64,
};

use crate::coerce::{coerce_array_like, coerce_scalar};
use crate::dtype::PyDType;
use crate::error::{map_sdnp, value_error, zero_division_error};
use crate::inner::ArrayInner;
use crate::unwrap::{finish, PyScalar};

pub fn binary_op(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    op: BinaryOp,
) -> PyResult<PyObject> {
    if is_scalar(left) && is_scalar(right) {
        let ls = coerce_scalar(left)?;
        let rs = coerce_scalar(right)?;
        let dt = ls.dtype().promote(rs.dtype());
        return scalar_binary(py, ls, rs, dt, op);
    }
    let (larr, rarr) = promote_arrays(left, right)?;
    let inner = match (larr, rarr) {
        (ArrayInner::Bool(l), ArrayInner::Bool(r)) => {
            dispatch_bool_bool(&l, &r, op)?
        }
        (ArrayInner::I64(l), ArrayInner::I64(r)) => {
            dispatch_i64_i64(&l, &r, op)?
        }
        (ArrayInner::F64(l), ArrayInner::F64(r)) => {
            dispatch_f64_f64(&l, &r, op)?
        }
        (ArrayInner::C64(l), ArrayInner::C64(r)) => {
            dispatch_c64_c64(&l, &r, op)?
        }
        _ => {
            return Err(value_error("internal dtype mismatch after promotion"))
        }
    };
    finish(py, inner)
}

pub fn unary_op(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    op: UnaryOp,
) -> PyResult<PyObject> {
    if is_scalar(obj) {
        let s = coerce_scalar(obj)?;
        let inner = scalar_to_inner_zero_d(&s);
        return finish(py, dispatch_unary_inner(inner, op)?);
    }
    let arr = coerce_array_like(obj, None)?;
    finish(py, dispatch_unary_inner(arr.inner, op)?)
}

fn is_scalar(obj: &Bound<'_, PyAny>) -> bool {
    crate::coerce::coerce_scalar(obj).is_ok()
        && !obj.extract::<PyRef<crate::array::PyArray>>().is_ok()
}

fn promote_arrays(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<(ArrayInner, ArrayInner)> {
    let l = coerce_array_like(left, None)?;
    let r = coerce_array_like(right, None)?;
    let dt = l.inner.dtype().promote(r.inner.dtype());
    Ok((cast_inner(l.inner, dt)?, cast_inner(r.inner, dt)?))
}

pub(crate) fn cast_inner(
    inner: ArrayInner,
    dt: PyDType,
) -> PyResult<ArrayInner> {
    if inner.dtype() == dt {
        return Ok(inner);
    }
    let source = inner.dtype();
    match (inner, dt) {
        (ArrayInner::Bool(a), PyDType::I64) => {
            Ok(ArrayInner::I64(map_sdnp(a.astype())?))
        }
        (ArrayInner::Bool(a), PyDType::F64) => {
            Ok(ArrayInner::F64(map_sdnp(a.astype())?))
        }
        (ArrayInner::Bool(a), PyDType::C64) => {
            Ok(ArrayInner::C64(map_sdnp(a.astype())?))
        }
        (ArrayInner::I64(a), PyDType::Bool) => {
            Ok(ArrayInner::Bool(map_sdnp(a.astype())?))
        }
        (ArrayInner::I64(a), PyDType::F64) => {
            Ok(ArrayInner::F64(map_sdnp(a.astype())?))
        }
        (ArrayInner::I64(a), PyDType::C64) => {
            Ok(ArrayInner::C64(map_sdnp(a.astype())?))
        }
        (ArrayInner::F64(a), PyDType::Bool) => {
            Ok(ArrayInner::Bool(map_sdnp(a.astype())?))
        }
        (ArrayInner::F64(a), PyDType::I64) => {
            Ok(ArrayInner::I64(map_sdnp(a.astype())?))
        }
        (ArrayInner::F64(a), PyDType::C64) => {
            Ok(ArrayInner::C64(map_sdnp(a.astype())?))
        }
        (ArrayInner::C64(a), PyDType::Bool) => {
            Ok(ArrayInner::Bool(map_sdnp(a.astype())?))
        }
        (ArrayInner::C64(a), PyDType::I64) => {
            Ok(ArrayInner::I64(map_sdnp(a.astype())?))
        }
        (ArrayInner::C64(a), PyDType::F64) => {
            Ok(ArrayInner::F64(map_sdnp(a.astype())?))
        }
        _ => Err(value_error(format!(
            "cannot cast {} array to {}",
            source.name(),
            dt.name()
        ))),
    }
}

fn scalar_to_inner_zero_d(s: &PyScalar) -> ArrayInner {
    crate::inner::scalar_to_inner(s)
}

fn scalar_binary(
    py: Python<'_>,
    left: PyScalar,
    right: PyScalar,
    dt: PyDType,
    op: BinaryOp,
) -> PyResult<PyObject> {
    let l = cast_scalar(left, dt)?;
    let r = cast_scalar(right, dt)?;
    if matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::And
            | BinaryOp::Or
    ) {
        let value = scalar_predicate(&l, &r, op)?;
        return finish(
            py,
            ArrayInner::Bool(map_sdnp(Array::from_vec(vec![value], &[]))?),
        );
    }
    let inner = match dt {
        PyDType::Bool => {
            let l = match l {
                PyScalar::Bool(v) => v,
                _ => unreachable!(),
            };
            let r = match r {
                PyScalar::Bool(v) => v,
                _ => unreachable!(),
            };
            ArrayInner::Bool(map_sdnp(Array::from_vec(
                vec![bool_op(l, r, op)?],
                &[],
            ))?)
        }
        PyDType::I64 => {
            let l = match l {
                PyScalar::I64(v) => v,
                _ => unreachable!(),
            };
            let r = match r {
                PyScalar::I64(v) => v,
                _ => unreachable!(),
            };
            ArrayInner::I64(map_sdnp(Array::from_vec(
                vec![i64_op(l, r, op)?],
                &[],
            ))?)
        }
        PyDType::F64 => {
            let l = match l {
                PyScalar::F64(v) => v,
                _ => unreachable!(),
            };
            let r = match r {
                PyScalar::F64(v) => v,
                _ => unreachable!(),
            };
            ArrayInner::F64(map_sdnp(Array::from_vec(
                vec![f64_op(l, r, op)?],
                &[],
            ))?)
        }
        PyDType::C64 => {
            let l = match l {
                PyScalar::C64(v) => v,
                _ => unreachable!(),
            };
            let r = match r {
                PyScalar::C64(v) => v,
                _ => unreachable!(),
            };
            ArrayInner::C64(map_sdnp(Array::from_vec(
                vec![c64_op(l, r, op)?],
                &[],
            ))?)
        }
    };
    finish(py, inner)
}

fn scalar_predicate(
    left: &PyScalar,
    right: &PyScalar,
    op: BinaryOp,
) -> PyResult<bool> {
    let unsupported_order =
        || value_error("ordering comparisons not supported for complex");
    Ok(match (left, right) {
        (PyScalar::Bool(left), PyScalar::Bool(right)) => match op {
            BinaryOp::Eq => left == right,
            BinaryOp::Ne => left != right,
            BinaryOp::And => *left && *right,
            BinaryOp::Or => *left || *right,
            _ => {
                return Err(value_error(
                    "ordering comparisons not supported for bool",
                ))
            }
        },
        (PyScalar::I64(left), PyScalar::I64(right)) => match op {
            BinaryOp::Eq => left == right,
            BinaryOp::Ne => left != right,
            BinaryOp::Lt => left < right,
            BinaryOp::Le => left <= right,
            BinaryOp::Gt => left > right,
            BinaryOp::Ge => left >= right,
            BinaryOp::And => *left != 0 && *right != 0,
            BinaryOp::Or => *left != 0 || *right != 0,
            _ => unreachable!("predicate operation checked by caller"),
        },
        (PyScalar::F64(left), PyScalar::F64(right)) => match op {
            BinaryOp::Eq => left == right,
            BinaryOp::Ne => left != right,
            BinaryOp::Lt => left < right,
            BinaryOp::Le => left <= right,
            BinaryOp::Gt => left > right,
            BinaryOp::Ge => left >= right,
            BinaryOp::And => *left != 0.0 && *right != 0.0,
            BinaryOp::Or => *left != 0.0 || *right != 0.0,
            _ => unreachable!("predicate operation checked by caller"),
        },
        (PyScalar::C64(left), PyScalar::C64(right)) => match op {
            BinaryOp::Eq => left == right,
            BinaryOp::Ne => left != right,
            BinaryOp::And => {
                *left != Complex64::new(0.0, 0.0)
                    && *right != Complex64::new(0.0, 0.0)
            }
            BinaryOp::Or => {
                *left != Complex64::new(0.0, 0.0)
                    || *right != Complex64::new(0.0, 0.0)
            }
            _ => return Err(unsupported_order()),
        },
        _ => {
            return Err(value_error(
                "internal scalar dtype mismatch after promotion",
            ))
        }
    })
}

fn cast_scalar(s: PyScalar, dt: PyDType) -> PyResult<PyScalar> {
    if s.dtype() == dt {
        return Ok(s);
    }
    Ok(match (s, dt) {
        (PyScalar::Bool(v), PyDType::I64) => PyScalar::I64(i64::from(v)),
        (PyScalar::Bool(v), PyDType::F64) => {
            PyScalar::F64(if v { 1.0 } else { 0.0 })
        }
        (PyScalar::Bool(v), PyDType::C64) => {
            PyScalar::C64(Complex64::new(if v { 1.0 } else { 0.0 }, 0.0))
        }
        (PyScalar::I64(v), PyDType::F64) => PyScalar::F64(v as f64),
        (PyScalar::I64(v), PyDType::C64) => {
            PyScalar::C64(Complex64::new(v as f64, 0.0))
        }
        (PyScalar::F64(v), PyDType::C64) => {
            PyScalar::C64(Complex64::new(v, 0.0))
        }
        (other, _) => other,
    })
}

#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Abs,
    Not,
    IsNan,
    IsInf,
    IsFinite,
    Conj,
    Real,
    Imag,
}

fn dispatch_unary_inner(
    inner: ArrayInner,
    op: UnaryOp,
) -> PyResult<ArrayInner> {
    use UnaryOp::*;
    match (inner, op) {
        (ArrayInner::Bool(a), Neg) => {
            Ok(ArrayInner::Bool(map_sdnp(negative(&a))?))
        }
        (ArrayInner::I64(a), Neg) => {
            Ok(ArrayInner::I64(map_sdnp(negative(&a))?))
        }
        (ArrayInner::F64(a), Neg) => {
            Ok(ArrayInner::F64(map_sdnp(negative(&a))?))
        }
        (ArrayInner::C64(a), Neg) => {
            Ok(ArrayInner::C64(map_sdnp(negative(&a))?))
        }
        (ArrayInner::Bool(a), Abs) => {
            Ok(ArrayInner::Bool(map_sdnp(absolute(&a))?))
        }
        (ArrayInner::I64(a), Abs) => {
            Ok(ArrayInner::I64(map_sdnp(absolute(&a))?))
        }
        (ArrayInner::F64(a), Abs) => {
            Ok(ArrayInner::F64(map_sdnp(absolute(&a))?))
        }
        (ArrayInner::C64(a), Abs) => {
            Ok(ArrayInner::F64(map_sdnp(absolute(&a))?))
        }
        (ArrayInner::Bool(a), Not) => {
            Ok(ArrayInner::Bool(map_sdnp(logical_not(&a))?))
        }
        (ArrayInner::I64(a), Not) => {
            Ok(ArrayInner::Bool(map_sdnp(logical_not(&a))?))
        }
        (ArrayInner::F64(a), Not) => {
            Ok(ArrayInner::Bool(map_sdnp(logical_not(&a))?))
        }
        (ArrayInner::C64(a), Not) => {
            Ok(ArrayInner::Bool(map_sdnp(logical_not(&a))?))
        }
        (ArrayInner::F64(a), IsNan) => {
            Ok(ArrayInner::Bool(map_sdnp(isnan(&a))?))
        }
        (ArrayInner::F64(a), IsInf) => {
            Ok(ArrayInner::Bool(map_sdnp(isinf(&a))?))
        }
        (ArrayInner::F64(a), IsFinite) => {
            Ok(ArrayInner::Bool(map_sdnp(isfinite(&a))?))
        }
        (ArrayInner::C64(a), IsNan) => {
            Ok(ArrayInner::Bool(map_sdnp(isnan(&a))?))
        }
        (ArrayInner::C64(a), IsInf) => {
            Ok(ArrayInner::Bool(map_sdnp(isinf(&a))?))
        }
        (ArrayInner::C64(a), IsFinite) => {
            Ok(ArrayInner::Bool(map_sdnp(isfinite(&a))?))
        }
        (ArrayInner::C64(a), Conj) => Ok(ArrayInner::C64(map_sdnp(conj(&a))?)),
        (ArrayInner::C64(a), Real) => Ok(ArrayInner::F64(map_sdnp(real(&a))?)),
        (ArrayInner::C64(a), Imag) => Ok(ArrayInner::F64(map_sdnp(imag(&a))?)),
        (inner, _) => Err(value_error(format!(
            "unsupported unary operation for dtype {}",
            inner.dtype().name()
        ))),
    }
}

fn dispatch_bool_bool(
    l: &Array<bool>,
    r: &Array<bool>,
    op: BinaryOp,
) -> PyResult<ArrayInner> {
    Ok(match op {
        BinaryOp::And => ArrayInner::Bool(map_sdnp(logical_and(l, r))?),
        BinaryOp::Or => ArrayInner::Bool(map_sdnp(logical_or(l, r))?),
        BinaryOp::Eq => ArrayInner::Bool(map_sdnp(equal(l, r))?),
        BinaryOp::Ne => ArrayInner::Bool(map_sdnp(not_equal(l, r))?),
        _ => return Err(value_error("unsupported bool binary op")),
    })
}

fn dispatch_i64_i64(
    l: &Array<i64>,
    r: &Array<i64>,
    op: BinaryOp,
) -> PyResult<ArrayInner> {
    Ok(match op {
        BinaryOp::Add => ArrayInner::I64(map_sdnp(add(l, r))?),
        BinaryOp::Sub => ArrayInner::I64(map_sdnp(subtract(l, r))?),
        BinaryOp::Mul => ArrayInner::I64(map_sdnp(multiply(l, r))?),
        BinaryOp::Div => ArrayInner::I64(map_sdnp(divide(l, r))?),
        BinaryOp::FloorDiv => ArrayInner::I64(map_sdnp(trunc_divide(l, r))?),
        BinaryOp::Mod => ArrayInner::I64(map_sdnp(remainder(l, r))?),
        BinaryOp::Pow => ArrayInner::I64(map_sdnp(power(l, r))?),
        BinaryOp::Eq => ArrayInner::Bool(map_sdnp(equal(l, r))?),
        BinaryOp::Ne => ArrayInner::Bool(map_sdnp(not_equal(l, r))?),
        BinaryOp::Lt => ArrayInner::Bool(map_sdnp(less(l, r))?),
        BinaryOp::Le => ArrayInner::Bool(map_sdnp(less_equal(l, r))?),
        BinaryOp::Gt => ArrayInner::Bool(map_sdnp(greater(l, r))?),
        BinaryOp::Ge => ArrayInner::Bool(map_sdnp(greater_equal(l, r))?),
        BinaryOp::And => ArrayInner::Bool(map_sdnp(logical_and(l, r))?),
        BinaryOp::Or => ArrayInner::Bool(map_sdnp(logical_or(l, r))?),
    })
}

fn dispatch_f64_f64(
    l: &Array<f64>,
    r: &Array<f64>,
    op: BinaryOp,
) -> PyResult<ArrayInner> {
    Ok(match op {
        BinaryOp::Add => ArrayInner::F64(map_sdnp(add(l, r))?),
        BinaryOp::Sub => ArrayInner::F64(map_sdnp(subtract(l, r))?),
        BinaryOp::Mul => ArrayInner::F64(map_sdnp(multiply(l, r))?),
        BinaryOp::Div => ArrayInner::F64(map_sdnp(divide(l, r))?),
        BinaryOp::FloorDiv => ArrayInner::F64(map_sdnp(trunc_divide(l, r))?),
        BinaryOp::Mod => ArrayInner::F64(map_sdnp(remainder(l, r))?),
        BinaryOp::Pow => ArrayInner::F64(map_sdnp(power(l, r))?),
        BinaryOp::Eq => ArrayInner::Bool(map_sdnp(equal(l, r))?),
        BinaryOp::Ne => ArrayInner::Bool(map_sdnp(not_equal(l, r))?),
        BinaryOp::Lt => ArrayInner::Bool(map_sdnp(less(l, r))?),
        BinaryOp::Le => ArrayInner::Bool(map_sdnp(less_equal(l, r))?),
        BinaryOp::Gt => ArrayInner::Bool(map_sdnp(greater(l, r))?),
        BinaryOp::Ge => ArrayInner::Bool(map_sdnp(greater_equal(l, r))?),
        BinaryOp::And => ArrayInner::Bool(map_sdnp(logical_and(l, r))?),
        BinaryOp::Or => ArrayInner::Bool(map_sdnp(logical_or(l, r))?),
    })
}

fn dispatch_c64_c64(
    l: &Array<Complex64>,
    r: &Array<Complex64>,
    op: BinaryOp,
) -> PyResult<ArrayInner> {
    Ok(match op {
        BinaryOp::Add => ArrayInner::C64(map_sdnp(add(l, r))?),
        BinaryOp::Sub => ArrayInner::C64(map_sdnp(subtract(l, r))?),
        BinaryOp::Mul => ArrayInner::C64(map_sdnp(multiply(l, r))?),
        BinaryOp::Div => ArrayInner::C64(map_sdnp(divide(l, r))?),
        BinaryOp::FloorDiv => ArrayInner::C64(map_sdnp(trunc_divide(l, r))?),
        BinaryOp::Mod => ArrayInner::C64(map_sdnp(remainder(l, r))?),
        BinaryOp::Pow => ArrayInner::C64(map_sdnp(power(l, r))?),
        BinaryOp::Eq => ArrayInner::Bool(map_sdnp(equal(l, r))?),
        BinaryOp::Ne => ArrayInner::Bool(map_sdnp(not_equal(l, r))?),
        BinaryOp::And => ArrayInner::Bool(map_sdnp(logical_and(l, r))?),
        BinaryOp::Or => ArrayInner::Bool(map_sdnp(logical_or(l, r))?),
        _ => {
            return Err(value_error(
                "ordering comparisons not supported for complex",
            ))
        }
    })
}

fn bool_op(l: bool, r: bool, op: BinaryOp) -> PyResult<bool> {
    Ok(match op {
        BinaryOp::And => l && r,
        BinaryOp::Or => l || r,
        BinaryOp::Eq => l == r,
        BinaryOp::Ne => l != r,
        _ => return Err(value_error("unsupported bool scalar op")),
    })
}

fn i64_op(l: i64, r: i64, op: BinaryOp) -> PyResult<i64> {
    Ok(match op {
        BinaryOp::Add => l.wrapping_add(r),
        BinaryOp::Sub => l.wrapping_sub(r),
        BinaryOp::Mul => l.wrapping_mul(r),
        BinaryOp::Div | BinaryOp::FloorDiv => {
            if r == 0 {
                return Err(zero_division_error("integer division by zero"));
            }
            l.checked_div(r)
                .ok_or_else(|| value_error("integer division overflow"))?
        }
        BinaryOp::Mod => {
            if r == 0 {
                return Err(zero_division_error("integer remainder by zero"));
            }
            l.checked_rem(r)
                .ok_or_else(|| value_error("integer remainder overflow"))?
        }
        BinaryOp::Pow => {
            let exponent = u32::try_from(r).map_err(|_| {
                value_error("i64 power exponent must be in 0..=u32::MAX")
            })?;
            l.pow(exponent)
        }
        BinaryOp::Eq => i64::from(l == r),
        BinaryOp::Ne => i64::from(l != r),
        BinaryOp::Lt => i64::from(l < r),
        BinaryOp::Le => i64::from(l <= r),
        BinaryOp::Gt => i64::from(l > r),
        BinaryOp::Ge => i64::from(l >= r),
        BinaryOp::And => i64::from(l != 0 && r != 0),
        BinaryOp::Or => i64::from(l != 0 || r != 0),
    })
}

fn f64_op(l: f64, r: f64, op: BinaryOp) -> PyResult<f64> {
    Ok(match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        BinaryOp::FloorDiv => (l / r).trunc(),
        BinaryOp::Mod => l % r,
        BinaryOp::Pow => l.powf(r),
        BinaryOp::Eq => {
            if l == r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::Ne => {
            if l != r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::Lt => {
            if l < r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::Le => {
            if l <= r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::Gt => {
            if l > r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::Ge => {
            if l >= r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::And => {
            if l != 0.0 && r != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::Or => {
            if l != 0.0 || r != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    })
}

fn c64_op(l: Complex64, r: Complex64, op: BinaryOp) -> PyResult<Complex64> {
    Ok(match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        BinaryOp::FloorDiv => Complex64::new((l / r).re.trunc(), 0.0),
        BinaryOp::Mod => l % r,
        BinaryOp::Pow => l.powc(r),
        BinaryOp::Eq => {
            if (l.re == r.re) && (l.im == r.im) {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
        BinaryOp::Ne => {
            if (l.re != r.re) || (l.im != r.im) {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
        BinaryOp::And => {
            if l.re != 0.0 || l.im != 0.0 {
                if r.re != 0.0 || r.im != 0.0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
        BinaryOp::Or => {
            if (l.re != 0.0 || l.im != 0.0) || (r.re != 0.0 || r.im != 0.0) {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
        _ => {
            return Err(value_error(
                "ordering comparisons not supported for complex",
            ))
        }
    })
}

pub fn py_binary(
    py: Python<'_>,
    l: &Bound<'_, PyAny>,
    r: &Bound<'_, PyAny>,
    op: BinaryOp,
) -> PyResult<PyObject> {
    binary_op(py, l, r, op)
}

pub fn py_unary(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    op: UnaryOp,
) -> PyResult<PyObject> {
    unary_op(py, obj, op)
}
