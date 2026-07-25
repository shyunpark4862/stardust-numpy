//! Module-level ufunc wrappers registered on `sdnp`.
//!
//! Each free function delegates to [`crate::dispatch::py_binary`] or
//! [`crate::dispatch::py_unary`], which handle dtype promotion, scalar/array
//! mixing, and 0-D unwrap. Duplicating operators here lets users call
//! `sdnp.add(a, b)` as well as `a + b`.

use pyo3::prelude::*;

use crate::dispatch::{py_binary, py_unary, BinaryOp, UnaryOp};

/// Expand one unary ufunc binding from an op enum variant.
macro_rules! unary_ufunc {
    (
        $(#[$meta:meta])*
        $pyname:ident, $op:ident
    ) => {
        $(#[$meta])*
        #[pyfunction]
        pub fn $pyname(
            py: Python<'_>,
            obj: &Bound<'_, PyAny>,
        ) -> PyResult<PyObject> {
            py_unary(py, obj, UnaryOp::$op)
        }
    };
}

/// Expand one binary ufunc binding from an op enum variant.
macro_rules! binary_ufunc {
    (
        $(#[$meta:meta])*
        $pyname:ident, $op:ident
    ) => {
        $(#[$meta])*
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

unary_ufunc!(
    /// Element-wise negation (`-a`).
    ///
    /// Delegates to [`crate::dispatch::py_unary`] with dtype promotion and
    /// 0-D unwrap.
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Negated array, or a bare Python scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported operand type.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1, -2, 3])
    /// assert np.negative(a).to_list() == [-1, 2, -3]
    /// ```
    negative, Neg
);

unary_ufunc!(
    /// Element-wise absolute value (`abs(a)`).
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Absolute-value array with promoted dtype.
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported operand type.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([-1, 0, 2])
    /// assert np.absolute(a).to_list() == [1, 0, 2]
    /// ```
    absolute, Abs
);

binary_ufunc!(
    /// Element-wise addition (`a + b`).
    ///
    /// Scalars broadcast against arrays; dtypes promote per NumPy rules.
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Sum array, or a bare Python scalar when the result is 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.add(np.array([1, 2]), 10).to_list() == [11, 12]
    /// ```
    add, Add
);

binary_ufunc!(
    /// Element-wise subtraction (`a - b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Difference array, or a bare Python scalar when 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.subtract(10, np.array([1, 2])).to_list() == [9, 8]
    /// ```
    subtract, Sub
);

binary_ufunc!(
    /// Element-wise multiplication (`a * b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Product array, or a bare Python scalar when 0-D.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.multiply(np.array([2, 3]), 4).to_list() == [8, 12]
    /// ```
    multiply, Mul
);

binary_ufunc!(
    /// Element-wise true division (`a / b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Dividend (array or scalar).
    /// * `b` - Divisor (array or scalar).
    ///
    /// # Returns
    ///
    /// Quotient array as float64 (or complex when promoted).
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — division by zero or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.divide(np.array([1, 4]), 2).to_list() == [0.5, 2.0]
    /// ```
    divide, Div
);

binary_ufunc!(
    /// Element-wise floor division (`a // b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Dividend (array or scalar).
    /// * `b` - Divisor (array or scalar).
    ///
    /// # Returns
    ///
    /// Floor-quotient array with promoted dtype.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — division by zero or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.trunc_divide(np.array([7, 8]), 3).to_list() == [2, 2]
    /// ```
    trunc_divide, FloorDiv
);

binary_ufunc!(
    /// Element-wise remainder (`a % b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Dividend (array or scalar).
    /// * `b` - Divisor (array or scalar).
    ///
    /// # Returns
    ///
    /// Remainder array with promoted dtype.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — division by zero or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.remainder(np.array([7, 8]), 3).to_list() == [1, 2]
    /// ```
    remainder, Mod
);

binary_ufunc!(
    /// Element-wise exponentiation (`a ** b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Base (array or scalar).
    /// * `b` - Exponent (array or scalar).
    ///
    /// # Returns
    ///
    /// Power array with promoted dtype.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — invalid power or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.power(np.array([2, 3]), 2).to_list() == [4, 9]
    /// ```
    power, Pow
);

binary_ufunc!(
    /// Element-wise equality test (`a == b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array of pairwise comparisons.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.equal(np.array([1, 2]), 2).to_list() == [False, True]
    /// ```
    equal, Eq
);

binary_ufunc!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array of pairwise comparisons.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.not_equal(np.array([1, 2]), 2).to_list() == [True, False]
    /// ```
    not_equal, Ne
);

binary_ufunc!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array of pairwise comparisons.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.less(np.array([1, 3]), 2).to_list() == [True, False]
    /// ```
    less, Lt
);

binary_ufunc!(
    /// Element-wise less-or-equal test (`a <= b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array of pairwise comparisons.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.less_equal(np.array([1, 3]), 2).to_list() == [True, False]
    /// ```
    less_equal, Le
);

binary_ufunc!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array of pairwise comparisons.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.greater(np.array([1, 3]), 2).to_list() == [False, True]
    /// ```
    greater, Gt
);

binary_ufunc!(
    /// Element-wise greater-or-equal test (`a >= b`).
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array of pairwise comparisons.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// assert np.greater_equal(np.array([1, 3]), 2).to_list() == [False, True]
    /// ```
    greater_equal, Ge
);

binary_ufunc!(
    /// Element-wise logical AND on truthy values.
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([True, False, True])
    /// b = np.array([True, True, False])
    /// assert np.logical_and(a, b).to_list() == [True, False, False]
    /// ```
    logical_and, And
);

binary_ufunc!(
    /// Element-wise logical OR on truthy values.
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand (array or scalar).
    /// * `b` - Right operand (array or scalar).
    ///
    /// # Returns
    ///
    /// Boolean array.
    ///
    /// # Errors
    ///
    /// * `TypeError` — incompatible operands.
    /// * `ValueError` — broadcast or core failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([True, False, True])
    /// b = np.array([True, True, False])
    /// assert np.logical_or(a, b).to_list() == [True, True, True]
    /// ```
    logical_or, Or
);

/// Element-wise logical NOT on truthy values.
///
/// # Arguments
///
/// * `obj` - Array or scalar operand.
///
/// # Returns
///
/// Boolean array, or a bare Python `bool` when 0-D.
///
/// # Errors
///
/// * `TypeError` — unsupported operand type.
/// * `ValueError` — core ufunc failure.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert np.logical_not(np.array([True, False])).to_list() == [False, True]
/// ```
#[pyfunction]
pub fn logical_not(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    py_unary(py, obj, UnaryOp::Not)
}

unary_ufunc!(
    /// Element-wise NaN test (float64 only).
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Boolean array marking NaN elements.
    ///
    /// # Errors
    ///
    /// * `TypeError` — non-floating operand.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1.0, float("nan")])
    /// assert np.isnan(a).to_list() == [False, True]
    /// ```
    isnan, IsNan
);

unary_ufunc!(
    /// Element-wise infinity test (float64 only).
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Boolean array marking infinite elements.
    ///
    /// # Errors
    ///
    /// * `TypeError` — non-floating operand.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1.0, float("inf")])
    /// assert np.isinf(a).to_list() == [False, True]
    /// ```
    isinf, IsInf
);

unary_ufunc!(
    /// Element-wise finite test (float64 only).
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Boolean array marking finite elements.
    ///
    /// # Errors
    ///
    /// * `TypeError` — non-floating operand.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1.0, float("inf")])
    /// assert np.isfinite(a).to_list() == [True, False]
    /// ```
    isfinite, IsFinite
);

unary_ufunc!(
    /// Element-wise complex conjugate.
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Conjugated array with the input dtype.
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported operand type.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1 + 2j, 3 - 4j])
    /// assert np.conj(a).to_list() == [1 - 2j, 3 + 4j]
    /// ```
    conj, Conj
);

unary_ufunc!(
    /// Element-wise real part extraction.
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Real-part array (float64 for complex input).
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported operand type.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1 + 2j, 3 - 4j])
    /// assert np.real(a).to_list() == [1.0, 3.0]
    /// ```
    real, Real
);

unary_ufunc!(
    /// Element-wise imaginary part extraction.
    ///
    /// # Arguments
    ///
    /// * `obj` - Array or scalar operand.
    ///
    /// # Returns
    ///
    /// Imaginary-part array (float64 for complex input).
    ///
    /// # Errors
    ///
    /// * `TypeError` — unsupported operand type.
    /// * `ValueError` — core ufunc failure.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    ///
    /// a = np.array([1 + 2j, 3 - 4j])
    /// assert np.imag(a).to_list() == [2.0, -4.0]
    /// ```
    imag, Imag
);

/// Attach all ufunc callables to the extension module.
///
/// Registers arithmetic, comparison, logical, and floating-point test
/// free functions on the `sdnp` module object.
///
/// # Arguments
///
/// * `m` - Bound reference to the `sdnp` extension module.
///
/// # Returns
///
/// `Ok(())` when every callable is registered successfully.
///
/// # Errors
///
/// Returns `PyErr` if PyO3 function wrapping or registration fails.
///
/// # Examples
///
/// ```python
/// import sdnp as np
///
/// assert callable(np.add)
/// assert callable(np.logical_not)
/// ```
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
