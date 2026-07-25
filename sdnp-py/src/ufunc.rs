//! Ufunc free functions.

use pyo3::prelude::*;

use crate::dispatch::{py_binary, py_unary, BinaryOp, UnaryOp};

macro_rules! unary_ufunc {
    ($pyname:ident, $op:ident) => {
        #[pyfunction]
        pub fn $pyname(
            py: Python<'_>,
            obj: &Bound<'_, PyAny>,
        ) -> PyResult<PyObject> {
            py_unary(py, obj, UnaryOp::$op)
        }
    };
}

macro_rules! binary_ufunc {
    ($pyname:ident, $op:ident) => {
        #[pyfunction]
        pub fn $pyname(
            py: Python<'_>,
            a: &Bound<'_, PyAny>,
            b: &Bound<'_, PyAny>,
        ) -> PyResult<PyObject> {
            py_binary(py, a, b, BinaryOp::$op)
        }
    };
}

unary_ufunc!(negative, Neg);
unary_ufunc!(absolute, Abs);

binary_ufunc!(add, Add);
binary_ufunc!(subtract, Sub);
binary_ufunc!(multiply, Mul);
binary_ufunc!(divide, Div);
binary_ufunc!(trunc_divide, FloorDiv);
binary_ufunc!(remainder, Mod);
binary_ufunc!(power, Pow);

binary_ufunc!(equal, Eq);
binary_ufunc!(not_equal, Ne);
binary_ufunc!(less, Lt);
binary_ufunc!(less_equal, Le);
binary_ufunc!(greater, Gt);
binary_ufunc!(greater_equal, Ge);

binary_ufunc!(logical_and, And);
binary_ufunc!(logical_or, Or);

#[pyfunction]
pub fn logical_not(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    py_unary(py, obj, UnaryOp::Not)
}

unary_ufunc!(isnan, IsNan);
unary_ufunc!(isinf, IsInf);
unary_ufunc!(isfinite, IsFinite);
unary_ufunc!(conj, Conj);
unary_ufunc!(real, Real);
unary_ufunc!(imag, Imag);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    m.add_function(wrap_pyfunction!(trunc_divide, m)?)?;
    m.add_function(wrap_pyfunction!(remainder, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(negative, m)?)?;
    m.add_function(wrap_pyfunction!(absolute, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(logical_not, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(conj, m)?)?;
    m.add_function(wrap_pyfunction!(real, m)?)?;
    m.add_function(wrap_pyfunction!(imag, m)?)?;
    Ok(())
}
