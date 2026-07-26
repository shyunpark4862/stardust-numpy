"""Conditional selection, nonzero coordinates, and clipping.

``where`` broadcasts a boolean condition and two value operands to a common
shape, promotes the value operands, and writes a fresh selected output in
O(n).  ``nonzero`` scans logical C-order values and materializes one integer
coordinate array per source axis; both time and coordinate storage are O(n) in
the worst case.

``clip`` allocates a copy and clamps each real input value after converting
bounds to the input dtype.  Either bound may be ``None``.  Complex clipping is
rejected because sdnp defines no ordering for complex values.

All results are independent arrays.  These functions do not provide NumPy's
``out=`` keyword, and ``where`` does not expose the one-argument coordinate
form; use ``nonzero`` explicitly.
"""

from ._array import Array, ArrayLike, RealScalar, Scalar, ScalarT

def where(
    condition: Array[bool],
    x: ArrayLike[Scalar] | Scalar,
    y: ArrayLike[Scalar] | Scalar,
) -> Array[Scalar]:
    """Choose values from ``x`` or ``y`` according to a mask.

    Index or mask processing parses selections in Rust and may allocate a new output buffer.  Fancy and boolean indexing copy gathered elements rather than returning writable views.

    Parameters
    ----------
    condition : Array of bool
        Boolean selection mask.
    x, y : array-like or scalar
        True and false branches. Their dtypes are promoted.

    Returns
    -------
    Array
        Broadcast result with promoted branch dtype.

    Raises
    ------
    TypeError
        If the condition is not a boolean array.
    ValueError
        If operands cannot broadcast or coercion fails.

    Examples
    --------
    >>> c = sdnp.array([True, False])
    >>> sdnp.where(c, 1, 0).to_list()
    [1, 0]
    """

def nonzero(a: Array[Scalar]) -> tuple[Array[int], ...]:
    """Return coordinates of nonzero elements.

    Index or mask processing parses selections in Rust and may allocate a new output buffer.  Fancy and boolean indexing copy gathered elements rather than returning writable views.

    Parameters
    ----------
    a : Array
        Input of any supported dtype.

    Returns
    -------
    tuple of Array
        One one-dimensional int64 coordinate array per input dimension.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If coordinate extraction fails.

    Examples
    --------
    >>> r, c = sdnp.nonzero(sdnp.array([[0, 1], [2, 0]]))
    >>> r.to_list(), c.to_list()
    ([0, 1], [1, 0])
    """

def clip(
    a: Array[ScalarT],
    min: RealScalar | None,
    max: RealScalar | None,
) -> Array[ScalarT]:
    """Limit values to a closed scalar interval.

    The input is never modified in place; a new buffer is allocated and each
    element is clamped after casting ``min`` and ``max`` bounds to the array's
    storage dtype.  ``None`` on either side leaves that bound open.  Complex
    arrays are rejected—clip applies only to boolean, integer, and float
    inputs with real scalar bounds.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    min, max : real scalar or None
        Lower and upper bounds; ``None`` leaves that side unbounded.

    Returns
    -------
    Array
        Clipped copy with the input dtype.

    Raises
    ------
    TypeError
        If a bound is complex or input is zero-dimensional.
    ValueError
        For complex input, nonscalar bounds, or core failure.

    Examples
    --------
    >>> sdnp.clip(sdnp.array([-1, 5, 10]), 0, 8).to_list()
    [0, 5, 8]
    """
