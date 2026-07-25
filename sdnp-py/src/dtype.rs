//! Runtime dtype tags for Python-facing dispatch.
//!
//! Rust generics in the core are monomorphized at compile time. At the Python
//! boundary we erase to [`PyDType`] and match on four variants. Promotion
//! rules mirror NumPy-style widening so binary ops pick a common dtype before
//! calling typed kernels.

use pyo3::prelude::*;
use pyo3::types::PyType;
use sdnp::Complex64;

/// Supported element types exposed to Python.
///
/// Each variant corresponds to one storage layout in [`ArrayInner`] and one
/// family of monomorphized kernels in the Rust core. Python scalars and array
/// literals are classified into these tags before dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PyDType {
    /// Boolean storage (`bool` / `np.bool_`).
    Bool,
    /// Signed 64-bit integer storage (`int` / `np.int64`).
    I64,
    /// IEEE double storage (`float` / `np.float64`).
    F64,
    /// Complex double storage (`complex` / `np.complex128`).
    C64,
}

impl PyDType {
    /// Widen `self` and `other` to the dtype both operands can represent.
    ///
    /// Binary ufuncs and constructors pick a common dtype using NumPy-like
    /// promotion: bool → int → float → complex. This runs at the Python
    /// boundary before typed kernel dispatch; the core never sees mixed dtypes
    /// on one ufunc call.
    ///
    /// # Arguments
    ///
    /// * `self` — Dtype of the left or first operand.
    /// * `other` — Dtype of the right or second operand.
    ///
    /// # Returns
    ///
    /// The promoted [`PyDType`] tag for both operands.
    ///
    /// # Errors
    ///
    /// Infallible; promotion always yields one of the four variants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sdnp_py::dtype::PyDType;
    ///
    /// assert_eq!(PyDType::Bool.promote(PyDType::F64), PyDType::F64);
    /// assert_eq!(PyDType::I64.promote(PyDType::C64), PyDType::C64);
    /// ```
    pub fn promote(self, other: Self) -> Self {
        use PyDType::*;
        match (self, other) {
            (Bool, Bool) => Bool,
            (Bool, I64) | (I64, Bool) => I64,
            (Bool, F64) | (F64, Bool) | (I64, F64) | (F64, I64) => F64,
            (I64, I64) => I64,
            (F64, F64) => F64,
            // Complex wins over any real dtype.
            (C64, _) | (_, C64) => C64,
        }
    }

    /// Return the Python `type` object for this dtype (for the `dtype` getter).
    ///
    /// Array objects expose a read-only `dtype` property that returns the
    /// builtin Python type (`bool`, `int`, `float`, or `complex`) rather than
    /// a NumPy dtype object, keeping the binding lightweight.
    ///
    /// # Arguments
    ///
    /// * `py` — GIL token for allocating Python types.
    ///
    /// # Returns
    ///
    /// `Ok(Bound<PyType>)` for the corresponding builtin type.
    ///
    /// # Errors
    ///
    /// Infallible for all four variants under normal PyO3 initialization.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    /// assert np.zeros(3, dtype=bool).dtype is bool
    /// assert np.zeros(3).dtype is float
    /// ```
    pub fn python_type<'py>(
        self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyType>> {
        match self {
            PyDType::Bool => Ok(py.get_type::<pyo3::types::PyBool>()),
            PyDType::I64 => Ok(py.get_type::<pyo3::types::PyInt>()),
            PyDType::F64 => Ok(py.get_type::<pyo3::types::PyFloat>()),
            PyDType::C64 => Ok(py.get_type::<pyo3::types::PyComplex>()),
        }
    }

    /// Stable string name used in repr and error messages.
    ///
    /// These names appear in validation errors (e.g. dtype mismatch during
    /// `concatenate`) and in array string representations.
    ///
    /// # Arguments
    ///
    /// None — this is a pure tag query on `self`.
    ///
    /// # Returns
    ///
    /// One of `"bool"`, `"int64"`, `"float64"`, or `"complex128"`.
    pub fn name(self) -> &'static str {
        match self {
            PyDType::Bool => "bool",
            PyDType::I64 => "int64",
            PyDType::F64 => "float64",
            PyDType::C64 => "complex128",
        }
    }

    /// Parse a Python type object or type name into [`PyDType`].
    ///
    /// Creation APIs accept `dtype=bool`, `dtype=int`, bare type objects, or
    /// instances. Unsupported types fail with [`PyTypeError`] at the boundary.
    ///
    /// # Arguments
    ///
    /// * `obj` — Python object from a `dtype` keyword or similar.
    ///
    /// # Returns
    ///
    /// `Ok(PyDType)` when the object maps to a supported storage type.
    ///
    /// # Errors
    ///
    /// * [`PyTypeError`] — unsupported type or type name.
    /// * PyO3 extract errors — malformed `__name__` attribute.
    ///
    /// # Examples
    ///
    /// ```python
    /// import sdnp as np
    /// np.zeros(3, dtype=int)     # int64 storage
    /// np.zeros(3, dtype=complex) # complex128 storage
    /// ```
    pub fn from_python_type(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Accept concrete instances (e.g. `int`) or bare type objects.
        if obj.is_instance_of::<pyo3::types::PyBool>() {
            Ok(PyDType::Bool)
        } else if obj.is_instance_of::<pyo3::types::PyInt>() {
            Ok(PyDType::I64)
        } else if obj.is_instance_of::<pyo3::types::PyFloat>() {
            Ok(PyDType::F64)
        } else if obj.is_instance_of::<pyo3::types::PyComplex>() {
            Ok(PyDType::C64)
        } else if let Ok(name) = obj.getattr("__name__") {
            let name: String = name.extract()?;
            match name.as_str() {
                "bool" => Ok(PyDType::Bool),
                "int" => Ok(PyDType::I64),
                "float" => Ok(PyDType::F64),
                "complex" => Ok(PyDType::C64),
                other => Err(crate::error::type_error(format!(
                    "unsupported dtype: {other}"
                ))),
            }
        } else {
            Err(crate::error::type_error(format!(
                "unsupported dtype object: {obj}"
            )))
        }
    }
}

/// Extract a Python bool; bool is a subclass of int, so check it first.
///
/// Scalar coercion for bool arrays and conditions must distinguish `True`/
/// `False` from integer 0/1. PyO3's `extract::<bool>()` handles subclassing
/// correctly when bool is tested before int paths.
///
/// # Arguments
///
/// * `obj` — Python scalar from a ufunc operand or keyword.
///
/// # Returns
///
/// `Ok(bool)` on successful extraction.
///
/// # Errors
///
/// PyO3 conversion errors when `obj` is not bool-coercible.
pub(crate) fn scalar_to_bool(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.extract::<bool>()
}

/// Extract a Python int as `i64`.
///
/// Integer array literals and scalar operands are narrowed to `i64`, matching
/// the core's [`PyDType::I64`] storage width.
///
/// # Arguments
///
/// * `obj` — Python int or int-coercible object.
///
/// # Returns
///
/// `Ok(i64)` when the value fits in 64 bits.
///
/// # Errors
///
/// PyO3 overflow or type errors when extraction fails.
pub(crate) fn scalar_to_i64(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    obj.extract::<i64>()
}

/// Extract a Python float as `f64`.
///
/// Real floating operands map to [`PyDType::F64`] storage. Integers are not
/// accepted here; callers promote via dtype rules first.
///
/// # Arguments
///
/// * `obj` — Python float or float-coercible object.
///
/// # Returns
///
/// `Ok(f64)` on successful extraction.
///
/// # Errors
///
/// PyO3 type errors when `obj` is not float-coercible.
pub(crate) fn scalar_to_f64(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    obj.extract::<f64>()
}

/// Extract a Python complex as [`Complex64`].
///
/// Complex scalars are downcast to [`PyComplex`] and copied into the core's
/// `Complex64` newtype for C64 kernels.
///
/// # Arguments
///
/// * `obj` — Python complex number.
///
/// # Returns
///
/// `Ok(Complex64)` with real and imaginary parts.
///
/// # Errors
///
/// PyO3 downcast failure when `obj` is not a [`PyComplex`] instance.
pub(crate) fn scalar_to_c64(obj: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    let complex = obj.downcast::<pyo3::types::PyComplex>()?;
    Ok(Complex64::new(complex.real(), complex.imag()))
}
