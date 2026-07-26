"""Array construction, ranges, coordinate grids, and matrix factories.

Factories validate Python values and shapes at the PyO3 boundary, choose one of
the four supported storage dtypes, and construct typed Rust arrays.  Unless an
individual function explicitly returns a view, creation allocates a fresh
C-contiguous ``Arc<Vec<T>>`` buffer that does not alias its inputs.

GENERAL FACTORIES.  ``array`` recursively flattens rectangular nested input,
infers or applies a dtype, and rejects ragged data and bare scalar 0-D arrays.
``zeros`` and ``ones`` default to float storage; ``full`` infers storage from
the fill value.  Their time and space cost is O(product(shape)).

RANGES AND SAMPLES.  ``arange`` emits integer steps.  ``linspace`` uses a
constant additive interval, ``logspace`` raises a base to linearly spaced
exponents, and ``geomspace`` uses a constant multiplicative ratio.  Each
allocates O(num) float or integer storage.  ``endpoint=False`` excludes the
specified stop under the rules documented by each function.

COORDINATES AND MATRICES.  ``meshgrid`` broadcasts one-dimensional coordinate
arrays into dense grids.  Identity and triangular factories allocate every
matrix element, so an n by m result costs O(n * m).  ``tril`` and ``triu`` copy
input values before zeroing one side of a diagonal; ``diag`` either gathers a
matrix diagonal or expands a vector into a new diagonal matrix.

NUMPY DIFFERENCES.  Dtypes are Python type objects rather than ``numpy.dtype``
instances, bool is rejected by numeric triangular factories, and no NumPy
buffer can be adopted without copying.
"""

from typing import overload

from ._array import (
    Array, ArrayLike, ArrayResult, DType, MeshgridIndexing, NumericT, Scalar,
    ScalarT, Shape,
)

def array(
    obj: ArrayLike[ScalarT] | ScalarT,
    *,
    dtype: type[ScalarT] | None = None,
    shape: Shape | None = None,
) -> Array[ScalarT]:
    """Create an array from nested data or a shaped scalar fill.

    Nested Python sequences are walked recursively in Rust to infer shape and
    dtype, then copied into a fresh ``Arc<Vec<T>>`` buffer—there is no
    zero-copy import from lists.  A bare scalar requires an explicit
    ``shape=`` because sdnp rejects rank-0 arrays from Python.  Ragged
    nesting (rows of unequal length) raises ``ValueError``.  When ``dtype=``
    is omitted, element types are inferred with promotion to a common tag;
    explicit ``dtype`` forces conversion during fill.

    Parameters
    ----------
    obj : Array, nested sequence, or scalar
        Source values. A bare scalar requires ``shape``.
    dtype : {bool, int, float, complex}, optional
        Explicit storage type; otherwise inferred from nested values.
    shape : int or sequence of int, optional
        If supplied, treat ``obj`` as one scalar and fill this shape.

    Returns
    -------
    Array
        Homogeneous array with at least one dimension.

    Raises
    ------
    TypeError
        If data nesting or ``dtype`` is unsupported.
    ValueError
        If a bare scalar lacks ``shape`` or the shape is invalid.

    Examples
    --------
    >>> sdnp.array([[1, 2], [3, 4]]).shape
    [2, 2]
    >>> sdnp.array(5, shape=(2,)).to_list()
    [5, 5]
    """

@overload
def zeros(shape: Shape, *, dtype: None = None) -> Array[float]: ...
@overload
def zeros(shape: Shape, *, dtype: type[ScalarT]) -> Array[ScalarT]: ...
def zeros(
    shape: Shape,
    *,
    dtype: DType[Scalar] | None = None,
) -> Array[Scalar]:
    """Return an array filled with zeros.

    A fresh ``Arc<Vec<T>>`` buffer is allocated in Rust with the requested shape—no views of existing storage.  Default float dtype applies when ``dtype`` is omitted for numeric factories.

    Parameters
    ----------
    shape : int or sequence of int
        Output dimensions.
    dtype : {bool, int, float, complex}, optional
        Storage type. The default is ``float``.

    Returns
    -------
    Array
        Zero-filled array.

    Raises
    ------
    TypeError
        If ``dtype`` or a shape component is unsupported.
    ValueError
        If the shape is invalid or allocation fails.

    Examples
    --------
    >>> sdnp.zeros((2, 2), dtype=int).to_list()
    [[0, 0], [0, 0]]
    """

@overload
def ones(shape: Shape, *, dtype: None = None) -> Array[float]: ...
@overload
def ones(shape: Shape, *, dtype: type[ScalarT]) -> Array[ScalarT]: ...
def ones(
    shape: Shape,
    *,
    dtype: DType[Scalar] | None = None,
) -> Array[Scalar]:
    """Return an array filled with ones.

    A fresh ``Arc<Vec<T>>`` buffer is allocated in Rust with the requested shape—no views of existing storage.  Default float dtype applies when ``dtype`` is omitted for numeric factories.

    Parameters
    ----------
    shape : int or sequence of int
        Output dimensions.
    dtype : {bool, int, float, complex}, optional
        Storage type. The default is ``float``.

    Returns
    -------
    Array
        One-filled array.

    Raises
    ------
    TypeError
        If ``dtype`` or a shape component is unsupported.
    ValueError
        If the shape is invalid or allocation fails.

    Examples
    --------
    >>> sdnp.ones(3, dtype=int).to_list()
    [1, 1, 1]
    """

def full(shape: Shape, fill_value: ScalarT) -> Array[ScalarT]:
    """Return an array filled with one scalar value.

    A fresh ``Arc<Vec<T>>`` buffer is allocated in Rust with the requested shape—no views of existing storage.  Default float dtype applies when ``dtype`` is omitted for numeric factories.

    Parameters
    ----------
    shape : int or sequence of int
        Output dimensions.
    fill_value : scalar
        Value copied into every element; it determines the dtype.

    Returns
    -------
    Array
        Filled array.

    Raises
    ------
    TypeError
        If ``fill_value`` is not a supported scalar.
    ValueError
        If the shape is invalid or allocation fails.

    Examples
    --------
    >>> sdnp.full((2, 2), 7).to_list()
    [[7, 7], [7, 7]]
    """

def arange(start: int, stop: int | None = None, step: int = 1) -> Array[int]:
    """Return evenly spaced signed 64-bit integers.

    The Rust core builds a one-dimensional ``i64`` buffer in O(n) where *n* is
    the number of steps.  When ``stop`` is omitted, ``start`` acts as the
    exclusive upper bound and the interval is ``[0, start)`` with default
    ``step=1``—matching NumPy's ``arange(n)`` calling convention.  Only
    integer endpoints and steps are supported; there is no floating ``arange``.

    Parameters
    ----------
    start : int
        Inclusive start, or exclusive stop when ``stop`` is omitted.
    stop : int, optional
        Exclusive upper bound.
    step : int, optional
        Nonzero spacing, default 1.

    Returns
    -------
    Array of int
        One-dimensional integer range.

    Raises
    ------
    ValueError
        If ``step`` is zero or range construction fails.

    Examples
    --------
    >>> sdnp.arange(2, 8, 2).to_list()
    [2, 4, 6]
    """

def linspace(
    start: float,
    stop: float,
    num: int,
    *,
    endpoint: bool = True,
) -> Array[float]:
    """Return evenly spaced floating samples.

    The result contains ``num`` values separated by a constant additive step.
    With ``endpoint=True``, that step is ``(stop - start) / (num - 1)`` and
    both bounds are included; otherwise it is ``(stop - start) / num`` and
    ``stop`` is omitted.  ``num=0`` returns an empty array, while ``num=1``
    returns ``[start]``.

    Parameters
    ----------
    start, stop : float
        Finite interval bounds.
    num : int
        Number of samples.
    endpoint : bool, optional
        Include ``stop`` when true.

    Returns
    -------
    Array of float
        One-dimensional float64 samples.

    Raises
    ------
    ValueError
        If bounds are non-finite or sample generation fails.

    Examples
    --------
    >>> sdnp.linspace(0.0, 1.0, 3).to_list()
    [0.0, 0.5, 1.0]
    """

def logspace(
    start: float,
    stop: float,
    num: int,
    *,
    endpoint: bool = True,
    base: float = 10.0,
) -> Array[float]:
    """Return samples evenly spaced on a logarithmic scale.

    The function first forms ``num`` linearly spaced exponents between
    ``start`` and ``stop``, then stores ``base ** exponent`` for each one.
    Thus ``logspace(0, 2, 3)`` produces powers ``10**0``, ``10**1``, and
    ``10**2``.  With ``endpoint=False``, the final exponent ``stop`` is
    omitted.

    Parameters
    ----------
    start, stop : float
        Finite exponents delimiting the interval.
    num : int
        Number of samples.
    endpoint : bool, optional
        Include the final exponent.
    base : float, optional
        Positive base other than one.

    Returns
    -------
    Array of float
        One-dimensional float64 values ``base ** exponent``.

    Raises
    ------
    ValueError
        If bounds or base are invalid.

    Examples
    --------
    >>> sdnp.logspace(0.0, 2.0, 3).to_list()
    [1.0, 10.0, 100.0]
    """

def geomspace(
    start: float,
    stop: float,
    num: int,
    *,
    endpoint: bool = True,
) -> Array[float]:
    """Return samples evenly spaced on a geometric progression.

    The result contains ``num`` values separated by a constant multiplicative
    ratio rather than a constant additive step.  With ``endpoint=True`` and
    ``num > 1``, the ratio is ``(stop / start) ** (1 / (num - 1))``; for
    example, ``geomspace(1, 8, 4)`` produces ``[1, 2, 4, 8]``.  Negative
    bounds preserve their sign, and ``endpoint=False`` omits ``stop``.

    Parameters
    ----------
    start, stop : float
        Finite, nonzero bounds having the same sign.
    num : int
        Number of samples.
    endpoint : bool, optional
        Include ``stop`` when true.

    Returns
    -------
    Array of float
        One-dimensional float64 geometric samples.

    Raises
    ------
    ValueError
        If bounds are zero, non-finite, or have differing signs.

    Examples
    --------
    >>> sdnp.geomspace(1.0, 8.0, 4).to_list()
    [1.0, 2.0, 4.0, 8.0]
    """

def meshgrid(
    *arrays: Array[ScalarT],
    indexing: MeshgridIndexing = "xy",
) -> tuple[Array[ScalarT], ...]:
    """Return coordinate matrices from one-dimensional coordinate arrays.

    Each input vector is broadcast into a grid according to ``indexing``:
    ``"xy"`` uses Cartesian (default NumPy 2-D) axis ordering, while ``"ij"``
    uses matrix indexing.  All operands must share the same non-boolean dtype.
    The Rust implementation materializes broadcast views—O(product of grid
    shape) output size.  Calling with no coordinate arrays returns an empty
    tuple rather than raising.

    Parameters
    ----------
    *arrays : Array
        Coordinate vectors sharing one non-boolean dtype.
    indexing : {'xy', 'ij'}, optional
        Cartesian or matrix indexing convention.

    Returns
    -------
    tuple of Array
        One broadcast grid per coordinate vector.

    Raises
    ------
    TypeError
        If an operand is not an array.
    ValueError
        If dtypes differ, boolean input is used, vectors are invalid, or
        ``indexing`` is unrecognized.

    Examples
    --------
    >>> x, y = sdnp.meshgrid(sdnp.arange(2), sdnp.arange(3))
    >>> x.shape
    [3, 2]
    """

@overload
def eye(n: int, *, dtype: None = None) -> Array[float]: ...
@overload
def eye(n: int, *, dtype: type[NumericT]) -> Array[NumericT]: ...
def eye(
    n: int,
    *,
    dtype: type[int] | type[float] | type[complex] | None = None,
) -> Array[int | float | complex]:
    """Return a square identity matrix.

    The Rust factory allocates an ``n * n`` buffer and writes ones on the main
    diagonal and zeros elsewhere in O(n^2) time.  Boolean ``dtype`` is
    rejected; default storage is ``float`` when ``dtype`` is omitted.

    Parameters
    ----------
    n : int
        Number of rows and columns.
    dtype : {int, float, complex}, optional
        Storage type, default ``float``.

    Returns
    -------
    Array
        Two-dimensional identity matrix.

    Raises
    ------
    TypeError
        If ``dtype`` is unsupported.
    ValueError
        If ``bool`` is requested or allocation fails.

    Examples
    --------
    >>> sdnp.eye(2).to_list()
    [[1.0, 0.0], [0.0, 1.0]]
    """

@overload
def eye_with(
    n: int,
    m: int,
    *,
    k: int = 0,
    dtype: None = None,
) -> Array[float]: ...
@overload
def eye_with(
    n: int,
    m: int,
    *,
    k: int = 0,
    dtype: type[NumericT],
) -> Array[NumericT]: ...
def eye_with(
    n: int,
    m: int,
    *,
    k: int = 0,
    dtype: type[int] | type[float] | type[complex] | None = None,
) -> Array[int | float | complex]:
    """Return a rectangular identity-like matrix.

    Like :func:`eye`, this fills a fresh ``(n, m)`` buffer in O(n * m) time,
    placing ones along the diagonal selected by offset ``k``.  Positive ``k``
    shifts the diagonal upward; negative values shift downward.  Boolean
    ``dtype`` is rejected—use ``int``, ``float``, or ``complex``.

    Parameters
    ----------
    n, m : int
        Row and column counts.
    k : int, optional
        Diagonal offset; positive values select an upper diagonal.
    dtype : {int, float, complex}, optional
        Storage type, default ``float``.

    Returns
    -------
    Array
        Matrix containing ones on the selected diagonal.

    Raises
    ------
    TypeError
        If ``dtype`` is unsupported.
    ValueError
        If ``bool`` is requested or construction fails.

    Examples
    --------
    >>> sdnp.eye_with(2, 3, k=1).shape
    [2, 3]
    """

@overload
def tri(n: int, *, dtype: None = None) -> Array[float]: ...
@overload
def tri(n: int, *, dtype: type[NumericT]) -> Array[NumericT]: ...
def tri(
    n: int,
    *,
    dtype: type[int] | type[float] | type[complex] | None = None,
) -> Array[int | float | complex]:
    """Return a square lower-triangular matrix of ones.

    Fills an ``n * n`` buffer with ones on and below the main diagonal and
    zeros above in O(n^2) time.  Boolean ``dtype`` is not supported; default
    storage is ``float``.

    Parameters
    ----------
    n : int
        Side length.
    dtype : {int, float, complex}, optional
        Storage type, default ``float``.

    Returns
    -------
    Array
        Lower-triangular matrix.

    Raises
    ------
    TypeError
        If ``dtype`` is unsupported.
    ValueError
        If ``bool`` is requested or construction fails.

    Examples
    --------
    >>> sdnp.tri(2).to_list()
    [[1.0, 0.0], [1.0, 1.0]]
    """

@overload
def tri_with(
    n: int,
    m: int,
    k: int = 0,
    *,
    dtype: None = None,
) -> Array[float]: ...
@overload
def tri_with(
    n: int,
    m: int,
    k: int = 0,
    *,
    dtype: type[NumericT],
) -> Array[NumericT]: ...
def tri_with(
    n: int,
    m: int,
    k: int = 0,
    *,
    dtype: type[int] | type[float] | type[complex] | None = None,
) -> Array[int | float | complex]:
    """Return a rectangular lower-triangular matrix.

    Extends :func:`tri` to non-square ``(n, m)`` shapes.  The Rust kernel
    fills a new buffer in O(n * m), writing ones on and below the ``k``-th
    diagonal and zeros elsewhere.  Boolean ``dtype`` is not supported.

    Parameters
    ----------
    n, m : int
        Row and column counts.
    k : int, optional
        Boundary diagonal offset.
    dtype : {int, float, complex}, optional
        Storage type, default ``float``.

    Returns
    -------
    Array
        Matrix containing ones at and below diagonal ``k``.

    Raises
    ------
    TypeError
        If ``dtype`` is unsupported.
    ValueError
        If ``bool`` is requested or construction fails.

    Examples
    --------
    >>> sdnp.tri_with(2, 3).shape
    [2, 3]
    """

def tril(array: Array[NumericT], k: int = 0) -> Array[NumericT]:
    """Return a copy with elements above a diagonal zeroed.

    The input is copied in Rust and elements strictly above diagonal ``k``
    are set to zero—O(n) over the buffer.  This always allocates new storage;
    it is not an in-place mask view.  Boolean dtype is rejected.

    Parameters
    ----------
    array : Array
        Input with at least two dimensions and non-boolean dtype.
    k : int, optional
        Diagonal above which values are set to zero.

    Returns
    -------
    Array
        Lower-triangular copy with unchanged dtype.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If input is boolean, has too few dimensions, or the core fails.

    Examples
    --------
    >>> sdnp.tril(sdnp.array([[1, 2], [3, 4]])).to_list()
    [[1, 0], [3, 4]]
    """

def triu(array: Array[NumericT], k: int = 0) -> Array[NumericT]:
    """Return a copy with elements below a diagonal zeroed.

    Symmetric to :func:`tril`: the array is copied and elements strictly
    below diagonal ``k`` are zeroed in O(n) time.  Boolean inputs are not
    supported.

    Parameters
    ----------
    array : Array
        Input with at least two dimensions and non-boolean dtype.
    k : int, optional
        Diagonal below which values are set to zero.

    Returns
    -------
    Array
        Upper-triangular copy with unchanged dtype.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If input is boolean, has too few dimensions, or the core fails.

    Examples
    --------
    >>> sdnp.triu(sdnp.array([[1, 2], [3, 4]])).to_list()
    [[1, 2], [0, 4]]
    """

def diag(array: Array[NumericT], k: int = 0) -> ArrayResult[NumericT]:
    """Extract a diagonal or construct a diagonal matrix.

    For a 2-D input, the Rust core returns a zero-copy strided view along
    diagonal ``k`` in O(ndim) time.  For a 1-D input, it allocates a square
    matrix with those values on the main diagonal and zeros elsewhere.
    Boolean input is rejected because ``diag`` is treated as a numeric matrix
    operation.

    Parameters
    ----------
    array : Array
        One-dimensional vector or two-dimensional non-boolean matrix.
    k : int, optional
        Diagonal offset.

    Returns
    -------
    Array or scalar
        Matrix for one-dimensional input, or diagonal for two-dimensional
        input. The input dtype is preserved.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If input is boolean or its rank is not one or two.

    Examples
    --------
    >>> sdnp.diag(sdnp.array([[1, 2], [3, 4]])).to_list()
    [1, 4]
    """
