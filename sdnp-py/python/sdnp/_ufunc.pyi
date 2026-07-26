"""Binary and unary element-wise operation model.

The functions declared below are thin Python bindings over the same dispatch
layer used by ``Array`` operators.  For example, ``add(a, b)`` and ``a + b``
ultimately select the same Rust addition kernel; ``negative(a)`` and ``-a``
likewise share one unary kernel.  The module-level forms do not accept NumPy's
``out=`` or ``where=`` keyword arguments and always produce a new result.

BINARY OPERATIONS.  A binary operation accepts two scalar or array operands.
When both operands are bare Python scalars, the binding takes a scalar fast
path: it determines their dtypes, promotes them, computes one result, and
returns a native Python scalar.  If either operand is an ``Array``, both sides
are converted to arrays, promoted to one common storage dtype, and broadcast
to a common shape before a typed Rust kernel runs.

DTYPE PROMOTION.  The common promotion order is
``bool < int < float < complex``.  Thus adding an integer array to a float
scalar produces float storage, while mixing real and complex values produces
complex storage.  Promotion happens once before kernel dispatch; inner loops
operate on a concrete monomorphized type rather than checking dtype for every
element.  Individual operations can override the ordinary result dtype:
comparisons and logical operations return ``bool``; true division returns
``float`` or ``complex``; complex ``absolute`` returns real magnitudes.

BROADCASTING.  Binary array shapes align from the trailing axis.  Two aligned
dimensions are compatible when they are equal or either one is ``1``; a
missing leading dimension behaves as ``1``.  Scalar operands therefore
broadcast across the entire array.  Incompatible dimensions raise
``ValueError`` before element evaluation.  The output shape is the resulting
broadcast shape.

UNARY OPERATIONS.  A unary operation accepts one scalar or array and performs
no shape broadcasting.  Bare scalars are temporarily represented as internal
0-D arrays so they can use the same typed kernel as ordinary arrays, then are
immediately unwrapped back to Python scalars.  Array inputs preserve their
shape, although the result dtype may change for operations such as
``absolute``, ``real``, ``imag``, ``isnan``, ``isinf``, ``isfinite``, and
``logical_not``.

RESULTS AND 0-D UNWRAP.  Element-wise operations allocate a result buffer and
do not mutate their operands.  Any internal rank-0 result is returned as
``bool``, ``int``, ``float``, or ``complex`` rather than as an ``Array``.
Results with one or more dimensions are returned as ``Array`` objects.

ERRORS AND DOMAIN RULES.  Unsupported operand types fail during Python-side
coercion with ``TypeError``.  Unsupported operation/dtype combinations,
broadcast failures, and invalid domains surface as ``ValueError`` or the
operation-specific exception documented on the function.  Integer division
and remainder by zero are checked and fail; float and complex division use
their typed numeric kernels.  Integer powers additionally validate exponent
range.  Ordering operations are unavailable for complex values, while
equality and inequality remain supported.

PERFORMANCE.  For an output containing ``n`` elements, ordinary unary and
binary kernels require O(n) time and O(n) output space.  C-contiguous operands
use flat fast paths.  Strided, transposed, and broadcast inputs remain valid;
the core walks coalesced runs or general strides as needed.  Repeatedly mixing
dtypes may allocate promoted temporary arrays, so explicitly casting reusable
operands can avoid repeated conversion.
"""

from ._array import (
    AnyOperand,
    ArrayResult,
    Operand,
    RealScalar,
    Scalar,
    ScalarT,
)

def add(a: AnyOperand, b: AnyOperand) -> ArrayResult[Scalar]:
    """Add two operands element by element.

    Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or scalar
        Dtype-promoted sum.

    Raises
    ------
    TypeError
        If an operand type is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.add(sdnp.array([1, 2]), 10).to_list()
    [11, 12]
    """

def subtract(a: AnyOperand, b: AnyOperand) -> ArrayResult[Scalar]:
    """Subtract ``b`` from ``a`` element by element.

    Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or scalar
        Dtype-promoted difference.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.subtract(10, sdnp.array([1, 2])).to_list()
    [9, 8]
    """

def multiply(a: AnyOperand, b: AnyOperand) -> ArrayResult[Scalar]:
    """Multiply two operands element by element.

    Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or scalar
        Dtype-promoted product.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.multiply(sdnp.array([2, 3]), 4).to_list()
    [8, 12]
    """

def divide(
    a: AnyOperand,
    b: AnyOperand,
) -> ArrayResult[float | complex]:
    """Divide ``a`` by ``b`` element by element.

    Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible dividend and divisor.

    Returns
    -------
    Array or scalar
        Float64 quotient, or complex when promoted.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        For division by zero or incompatible shapes.

    Examples
    --------
    >>> sdnp.divide(sdnp.array([1, 4]), 2).to_list()
    [0.5, 2.0]
    """

def trunc_divide(
    a: AnyOperand,
    b: AnyOperand,
) -> ArrayResult[Scalar]:
    """Apply truncating floor division element by element.

    This is the explicit name for integer-style floor division in sdnp.
    The ``//`` operator on :class:`Array` delegates to the same Rust kernel.
    Unlike NumPy, which uses true division for ``/`` on integers, sdnp
    promotes ``/`` to float; use this function (or ``//``) when you want
    truncating quotients.  Operands broadcast and dtype-promote before a
    monomorphized divide kernel runs in O(n) over the output size.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible dividend and divisor.

    Returns
    -------
    Array or scalar
        Promoted integral-style quotient.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        For division by zero or incompatible shapes.

    Examples
    --------
    >>> sdnp.trunc_divide(sdnp.array([7, 8]), 3).to_list()
    [2, 2]
    """

def remainder(a: AnyOperand, b: AnyOperand) -> ArrayResult[Scalar]:
    """Return element-wise remainders.

    Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible dividend and divisor.

    Returns
    -------
    Array or scalar
        Promoted remainder values.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        For division by zero or incompatible shapes.

    Examples
    --------
    >>> sdnp.remainder(sdnp.array([7, 8]), 3).to_list()
    [1, 2]
    """

def power(a: AnyOperand, b: AnyOperand) -> ArrayResult[Scalar]:
    """Raise ``a`` to powers from ``b`` element by element.

    Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible bases and exponents.

    Returns
    -------
    Array or scalar
        Promoted power values.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        For invalid powers or incompatible shapes.

    Examples
    --------
    >>> sdnp.power(sdnp.array([2, 3]), 2).to_list()
    [4, 9]
    """

def negative(obj: Operand[ScalarT]) -> ArrayResult[ScalarT]:
    """Negate an operand element by element.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array or scalar
        Input values.

    Returns
    -------
    Array or scalar
        Negated values preserving the applicable dtype.

    Raises
    ------
    TypeError
        If the operand or dtype is unsupported.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.negative(sdnp.array([1, -2])).to_list()
    [-1, 2]
    """

def absolute(obj: AnyOperand) -> ArrayResult[RealScalar]:
    """Return element-wise absolute values.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array or scalar
        Input values.

    Returns
    -------
    Array or scalar
        Magnitudes; complex values produce float64 magnitudes.

    Raises
    ------
    TypeError
        If the operand is unsupported.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.absolute(sdnp.array([-1, 2])).to_list()
    [1, 2]
    """

def equal(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Test equality element by element.

    Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or bool
        Pairwise equality result.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.equal(sdnp.array([1, 2]), 2).to_list()
    [False, True]
    """

def not_equal(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Test inequality element by element.

    Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or bool
        Pairwise inequality result.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.not_equal(sdnp.array([1, 2]), 2).to_list()
    [True, False]
    """

def less(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Test ``a < b`` element by element.

    Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible ordered operands.

    Returns
    -------
    Array or bool
        Pairwise comparison.

    Raises
    ------
    TypeError
        If operands cannot be ordered.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.less(sdnp.array([1, 3]), 2).to_list()
    [True, False]
    """

def less_equal(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Test ``a <= b`` element by element.

    Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible ordered operands.

    Returns
    -------
    Array or bool
        Pairwise comparison.

    Raises
    ------
    TypeError
        If operands cannot be ordered.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.less_equal(sdnp.array([1, 3]), 2).to_list()
    [True, False]
    """

def greater(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Test ``a > b`` element by element.

    Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible ordered operands.

    Returns
    -------
    Array or bool
        Pairwise comparison.

    Raises
    ------
    TypeError
        If operands cannot be ordered.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.greater(sdnp.array([1, 3]), 2).to_list()
    [False, True]
    """

def greater_equal(
    a: AnyOperand,
    b: AnyOperand,
) -> ArrayResult[bool]:
    """Test ``a >= b`` element by element.

    Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible ordered operands.

    Returns
    -------
    Array or bool
        Pairwise comparison.

    Raises
    ------
    TypeError
        If operands cannot be ordered.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.greater_equal(sdnp.array([1, 3]), 2).to_list()
    [False, True]
    """

def logical_and(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Compute logical AND from element truth values.

    Boolean arrays are combined element-wise in a monomorphized kernel after broadcasting.  Rank-0 results unwrap to Python ``bool`` scalars.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or bool
        Boolean conjunction.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.logical_and(sdnp.array([1, 0]), True).to_list()
    [True, False]
    """

def logical_or(a: AnyOperand, b: AnyOperand) -> ArrayResult[bool]:
    """Compute logical OR from element truth values.

    Boolean arrays are combined element-wise in a monomorphized kernel after broadcasting.  Rank-0 results unwrap to Python ``bool`` scalars.

    Parameters
    ----------
    a, b : Array or scalar
        Broadcast-compatible operands.

    Returns
    -------
    Array or bool
        Boolean disjunction.

    Raises
    ------
    TypeError
        If an operand is unsupported.
    ValueError
        If shapes cannot broadcast.

    Examples
    --------
    >>> sdnp.logical_or(sdnp.array([1, 0]), False).to_list()
    [True, False]
    """

def logical_not(obj: AnyOperand) -> ArrayResult[bool]:
    """Invert element truth values.

    Boolean arrays are combined element-wise in a monomorphized kernel after broadcasting.  Rank-0 results unwrap to Python ``bool`` scalars.

    Parameters
    ----------
    obj : Array or scalar
        Input values.

    Returns
    -------
    Array or bool
        Element-wise logical negation.

    Raises
    ------
    TypeError
        If the operand is unsupported.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.logical_not(sdnp.array([True, False])).to_list()
    [False, True]
    """

def isnan(obj: Operand[float | complex]) -> ArrayResult[bool]:
    """Test float64 and complex128 values for NaN.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array of float or complex, or scalar
        Input values. A complex value is NaN when either component is NaN.

    Returns
    -------
    Array or bool
        True where values are NaN.

    Raises
    ------
    TypeError
        If the operand is neither float64 nor complex128.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.isnan(sdnp.array([1.0, float("nan")])).to_list()
    [False, True]
    """

def isinf(obj: Operand[float | complex]) -> ArrayResult[bool]:
    """Test float64 and complex128 values for infinity.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array of float or complex, or scalar
        Input values. A complex value is infinite when either component is
        infinite.

    Returns
    -------
    Array or bool
        True where values are infinite.

    Raises
    ------
    TypeError
        If the operand is neither float64 nor complex128.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.isinf(sdnp.array([1.0, float("inf")])).to_list()
    [False, True]
    """

def isfinite(obj: Operand[float | complex]) -> ArrayResult[bool]:
    """Test float64 and complex128 values for finiteness.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array of float or complex, or scalar
        Input values. A complex value is finite only when both components are
        finite.

    Returns
    -------
    Array or bool
        True where values are neither NaN nor infinite.

    Raises
    ------
    TypeError
        If the operand is neither float64 nor complex128.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.isfinite(sdnp.array([1.0, float("inf")])).to_list()
    [True, False]
    """

def conj(obj: Operand[complex]) -> ArrayResult[complex]:
    """Return complex conjugates element by element.

    Only ``complex128`` storage is accepted.  The unary kernel negates each
    imaginary part in a single O(n) pass; real inputs are rejected at the
    boundary because they are not tagged as complex storage.

    Parameters
    ----------
    obj : Array of complex or complex
        Complex128 input values.

    Returns
    -------
    Array or complex
        Complex128 values with negated imaginary components.

    Raises
    ------
    TypeError
        If the operand is not complex128.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.conj(sdnp.array([1 + 2j])).to_list()
    [(1-2j)]
    """

def real(obj: Operand[complex]) -> ArrayResult[float]:
    """Return real components element by element.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array of complex or complex
        Complex128 input values.

    Returns
    -------
    Array or float
        Float64 real components.

    Raises
    ------
    TypeError
        If the operand is not complex128.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.real(sdnp.array([1 + 2j])).to_list()
    [1.0]
    """

def imag(obj: Operand[complex]) -> ArrayResult[float]:
    """Return imaginary components element by element.

    The input is evaluated in a monomorphized unary kernel after dtype tagging at the Python boundary.  C-contiguous arrays take a single O(n) pass; non-contiguous layouts may iterate coalesced stride segments.  Rank-0 outputs unwrap to Python scalars.  There is no ``out=`` or ``where=`` keyword support.

    Parameters
    ----------
    obj : Array of complex or complex
        Complex128 input values.

    Returns
    -------
    Array or scalar
        Float64 imaginary components for complex input.

    Raises
    ------
    TypeError
        If the operand is not complex128.
    ValueError
        If the core operation fails.

    Examples
    --------
    >>> sdnp.imag(sdnp.array([1 + 2j])).to_list()
    [2.0]
    """
