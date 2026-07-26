"""Reductions, index reductions, and cumulative scans.

Reduction functions normalize ``axis`` at the Python boundary and dispatch to
a monomorphized Rust accumulator.  ``axis=None`` reduces the full logical
C-order array; a specific axis removes that dimension unless the operation is
a cumulative scan.  A result with no remaining dimensions is unwrapped to a
Python scalar.

DTYPE AND NAN POLICY.  Accumulator and output dtypes follow each function's
signature.  Boolean sums and products promote to integer, means and statistical
operations produce floating or complex-compatible results as documented.
``nan_policy='propagate'`` preserves NaN influence, while ``'ignore'`` skips
NaN values on supported floating paths.

COMPLEXITY.  Full reductions inspect each input element once: O(n) time and
O(output size) storage.  Axis reductions use contiguous fast paths when layout
allows and stride-aware traversal otherwise.  ``cumsum`` and ``cumprod``
allocate an output matching the input shape and perform one ordered scan.

NUMPY DIFFERENCES.  There is no ``out=``, ``where=``, or general ``dtype=``
reduction keyword.  Complex ordering reductions such as ``min``,
``max``, ``argmin``, and ``argmax`` are not supported.
"""

from ._array import (
    Array,
    ArrayResult,
    Axis,
    NanPolicy,
    RealScalar,
    Scalar,
    ScalarT,
)

def sum(
    a: Array[ScalarT],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[ScalarT | int]:
    """Sum elements over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Input values.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes with length one.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or scalar
        Sum; boolean input accumulates as signed 64-bit integer.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If axes conflict, an axis is invalid, or policy is unknown.

    Examples
    --------
    >>> sdnp.sum(sdnp.array([1, 2, 3]))
    6
    """

def prod(
    a: Array[ScalarT],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[ScalarT | int]:
    """Multiply elements over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Input values.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes with length one.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or scalar
        Product; boolean input accumulates as integer.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If axes or NaN policy are invalid.

    Examples
    --------
    >>> sdnp.prod(sdnp.array([2, 3, 4]))
    24
    """

def min(
    a: Array[ScalarT],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[ScalarT]:
    """Return minima over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Non-complex input.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or scalar
        Minimum values with input dtype.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex dtype, empty slices, invalid axes, or invalid policy.

    Examples
    --------
    >>> sdnp.min(sdnp.array([3, 1, 2]))
    1
    """

def max(
    a: Array[ScalarT],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[ScalarT]:
    """Return maxima over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Non-complex input.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or scalar
        Maximum values with input dtype.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex dtype, empty slices, invalid axes, or invalid policy.

    Examples
    --------
    >>> sdnp.max(sdnp.array([3, 1, 2]))
    3
    """

def mean(
    a: Array[ScalarT],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[float | complex]:
    """Return arithmetic means over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Input values.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or scalar
        Float64 for boolean/integer/float input, complex for complex input.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For empty slices, invalid axes, or invalid policy.

    Examples
    --------
    >>> sdnp.mean(sdnp.array([2, 4, 6]))
    4.0
    """

def var(
    a: Array[RealScalar],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[float]:
    """Return population variance over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or float
        Float64 population variance with ``ddof=0``.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex input, empty slices, invalid axes, or invalid policy.

    Examples
    --------
    >>> sdnp.var(sdnp.array([1.0, 2.0, 3.0]))
    1.0
    """

def std(
    a: Array[RealScalar],
    *,
    axis: Axis | None = None,
    axes: Axis | None = None,
    keepdims: bool = False,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[float]:
    """Return population standard deviation over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    axis, axes : int, sequence of int, or None, optional
        Mutually exclusive reduction-axis specifications.
    keepdims : bool, optional
        Retain reduced axes.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or float
        Square root of population variance.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex input, empty slices, invalid axes, or invalid policy.

    Examples
    --------
    >>> sdnp.std(sdnp.array([1.0, 2.0, 3.0]))
    1.0
    """

def any(
    a: Array[Scalar],
    *,
    axis: Axis | None = None,
    keepdims: bool = False,
) -> ArrayResult[bool]:
    """Test whether any values are truthy over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Input of any supported dtype.
    axis : int, sequence of int, or None, optional
        Reduction axes.
    keepdims : bool, optional
        Retain reduced axes.

    Returns
    -------
    Array or bool
        Boolean reduction result.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If an axis is invalid.

    Examples
    --------
    >>> sdnp.any(sdnp.array([0, 0, 1]))
    True
    """

def all(
    a: Array[Scalar],
    *,
    axis: Axis | None = None,
    keepdims: bool = False,
) -> ArrayResult[bool]:
    """Test whether all values are truthy over selected axes.

    Reduction runs in a typed Rust kernel after optional axis normalization.  When an axis is specified, the output rank drops by one along that axis; a full reduction may return a Python scalar (0-D unwrap).  Contiguous inputs use fast single-pass accumulation where the layout allows.

    Parameters
    ----------
    a : Array
        Input of any supported dtype.
    axis : int, sequence of int, or None, optional
        Reduction axes.
    keepdims : bool, optional
        Retain reduced axes.

    Returns
    -------
    Array or bool
        Boolean reduction result.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If an axis is invalid.

    Examples
    --------
    >>> sdnp.all(sdnp.array([1, 1, 1]))
    True
    """

def argmin(
    a: Array[RealScalar],
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[int]:
    """Return indices of minimum values.

    The Rust reduction scans along the chosen axis (or the full flattened
    buffer when ``axis=None``), honoring ``nan_policy`` for floating inputs.
    For boolean input the scan stops at the first ``False`` because no later
    value can be smaller, matching NumPy's specialized bool kernel.  Only real
    dtypes (``bool``, ``int``, ``float``) are supported; complex arrays are
    rejected.  A full reduction unwraps to a Python ``int`` index rather than
    a 0-D array.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    axis : int or None, optional
        Reduction axis; ``None`` returns a flat index.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or int
        Signed 64-bit indices.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex input, empty or all-NaN slices, or invalid axis.

    Examples
    --------
    >>> sdnp.argmin(sdnp.array([3, 1, 2]))
    1
    """

def argmax(
    a: Array[RealScalar],
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
) -> ArrayResult[int]:
    """Return indices of maximum values.

    Same implementation strategy as :func:`argmin`, but selects the index of
    the largest element along the chosen axis.  For boolean input the scan
    stops at the first ``True`` because no later value can be larger.  Float
    ``nan_policy`` controls NaN handling; complex input is rejected.  Full
    reductions unwrap to a Python ``int`` scalar.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    axis : int or None, optional
        Reduction axis; ``None`` returns a flat index.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array or int
        Signed 64-bit indices.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex input, empty or all-NaN slices, or invalid axis.

    Examples
    --------
    >>> sdnp.argmax(sdnp.array([3, 1, 2]))
    0
    """

def cumsum(
    a: Array[ScalarT],
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
) -> Array[ScalarT | int]:
    """Return cumulative sums.

    A single pass over the selected axis accumulates partial sums in O(n)
    time, writing into a new buffer the same shape as the input.  Boolean
    inputs promote to ``int`` in the output.  ``nan_policy`` applies to
    floating reductions.

    Parameters
    ----------
    a : Array
        Input values.
    axis : int or None, optional
        Accumulation axis; ``None`` uses flat C order.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array
        Cumulative values; boolean input promotes to integer.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If the axis or NaN policy is invalid.

    Examples
    --------
    >>> sdnp.cumsum(sdnp.array([1, 2, 3])).to_list()
    [1, 3, 6]
    """

def cumprod(
    a: Array[ScalarT],
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
) -> Array[ScalarT | int]:
    """Return cumulative products.

    Like :func:`cumsum`, but multiplies along the axis in one O(n) forward
    scan into a freshly allocated buffer.  Boolean inputs promote to ``int``.
    Floating ``nan_policy`` is honored where applicable.

    Parameters
    ----------
    a : Array
        Input values.
    axis : int or None, optional
        Accumulation axis; ``None`` uses flat C order.
    nan_policy : {'propagate', 'ignore'}, optional
        Float64 NaN handling.

    Returns
    -------
    Array
        Cumulative values; boolean input promotes to integer.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If the axis or NaN policy is invalid.

    Examples
    --------
    >>> sdnp.cumprod(sdnp.array([1, 2, 3])).to_list()
    [1, 2, 6]
    """
