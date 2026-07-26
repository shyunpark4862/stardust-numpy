"""N-dimensional iterator classes and factory functions.

``ndindex`` is the only streaming iterator in this group: it keeps one current
coordinate and yields C-order tuples with O(rank) state.  ``__len__`` reports
remaining coordinates.

``ndenumerate`` materializes C-order values and their indices before returning,
while ``nditer`` validates one or two same-dtype operands, broadcasts them, and
collects all aligned scalar steps up front.  Their construction therefore
requires O(n) time and storage; later ``__next__`` calls are O(1).

All yielded elements are native Python scalars rather than 0-D arrays.  This
``nditer`` is intentionally narrower than NumPy's implementation: it has no
flags, write-back operands, arbitrary operand count, or mixed-dtype promotion.
"""

from collections.abc import Iterator
from typing import overload, override

from ._array import Array, Scalar, ScalarT, Shape

class _NdIndexIterator(Iterator[tuple[int, ...]]):
    """C-order iterator over every multi-index for a fixed shape.

    Runtime class name ``ndindex`` (this stub uses ``_NdIndexIterator`` because
    the factory function shares the public name).  Obtained from :func:`ndindex`.

    STREAMING CORE ITERATOR.  Unlike ``flatiter`` / ``ndenumerate`` / ``nditer``
    on the Python side, ``ndindex`` wraps the Rust ``NdIndex`` state machine
    directly.  Coordinates are generated incrementally in C-order (last axis
    varies fastest) with O(1) amortized work per step and O(rank) memory for
    the current index vector—no up-front allocation of all tuples.

    LENGTH.  ``__len__`` reports the number of indices remaining, including
    the one that would be returned by the next ``__next__``.  It decreases as
    iteration proceeds and reaches ``0`` after exhaustion.  This matches tests
    that call ``len(it)`` mid-loop.

    EMPTY SHAPES.  Any zero-length dimension produces zero indices—for example
    ``ndindex((0,))`` and ``ndindex((2, 0, 3))`` yield nothing.  A scalar
    shape concept with no axes is not exposed from Python arrays (sdnp rejects
    0-D arrays), but ``ndindex`` itself accepts any valid shape vector.

    NUMPY DIFFERENCE.  NumPy ``np.ndindex(2, 3)`` is variadic; sdnp requires a
    single ``shape`` argument—``ndindex((2, 3))`` or ``ndindex(3)`` for 1-D.

    Yields
    ------
    tuple of int
        Next coordinate, e.g. ``(0, 1, 2)``.

    See Also
    --------
    ndindex : factory function
    ndenumerate : pairs each index with an array element

    Examples
    --------
    >>> list(sdnp.ndindex((2, 2)))
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> len(sdnp.ndindex((2, 3)))
    6
    """

    @override
    def __iter__(self) -> _NdIndexIterator:
        """Return this iterator.

        Returns
        -------
        ndindex
            ``self``.

        Examples
        --------
        >>> it = sdnp.ndindex((1,))
        >>> iter(it) is it
        True
        """

    @override
    def __next__(self) -> tuple[int, ...]:
        """Return the next C-order index.

        Returns
        -------
        tuple of int
            Next coordinate.

        Raises
        ------
        StopIteration
            When every coordinate has been visited.

        Examples
        --------
        >>> next(sdnp.ndindex((2,)))
        (0,)
        """

    def __len__(self) -> int:
        """Return the number of remaining indices.

        Returns
        -------
        int
            Exact remaining iteration length.

        Examples
        --------
        >>> len(sdnp.ndindex((2, 3)))
        6
        """

class _NdEnumerateIterator[ScalarT: Scalar](
    Iterator[tuple[tuple[int, ...], ScalarT]]
):
    """C-order iterator yielding ``(multi_index, scalar)`` pairs for one array.

    Runtime class name ``ndenumerate``; returned from :func:`ndenumerate`.

    FULL MATERIALIZATION AT CREATION.  ``ndenumerate(a)`` rejects 0-D inputs,
    then (1) flattens ``a`` with ``to_vec()`` into a ``Vec<PyScalar>`` in
    logical C-order—O(n) element copy for non-contiguous arrays—and (2) walks
    an ``NdIndex`` over ``a.shape`` in lockstep, storing every
    ``(index_tuple, scalar)`` pair in a Python-side ``Vec``.  ``__next__`` only
    advances a cursor through that buffer.  Memory is O(n) in the array size;
    iteration does not touch the live array after construction.

    SCALAR UNWRAP.  Values are always native Python scalars, never 0-D
    ``Array`` objects, even when the source dtype is ``bool``, ``complex``, etc.

    NO ``__len__``.  Unlike ``ndindex``, the enumerate object does not support
    ``len()``—calling it raises ``TypeError``.

    ORDER.  Pair order matches C-order flat traversal.  Non-contiguous views
    enumerate in logical C-order of the view, consistent with :attr:`Array.flat`.

    Yields
    ------
    tuple
        ``(index_tuple, scalar)`` for the next element.

    See Also
    --------
    ndenumerate : factory function
    ndindex : indices without values
    flatiter : values without indices

    Examples
    --------
    >>> list(sdnp.ndenumerate(sdnp.array([10, 20])))
    [((0,), 10), ((1,), 20)]
    >>> list(sdnp.ndenumerate(sdnp.array([[1, 2], [3, 4]])))
    [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]
    """

    @override
    def __iter__(self) -> _NdEnumerateIterator[ScalarT]:
        """Return this iterator.

        Returns
        -------
        ndenumerate
            ``self``.

        Examples
        --------
        >>> it = sdnp.ndenumerate(sdnp.array([1]))
        >>> iter(it) is it
        True
        """

    @override
    def __next__(self) -> tuple[tuple[int, ...], ScalarT]:
        """Return the next index and scalar.

        Returns
        -------
        tuple
            ``(index_tuple, scalar)``.

        Raises
        ------
        StopIteration
            When all elements have been visited.

        Examples
        --------
        >>> next(sdnp.ndenumerate(sdnp.array([5])))
        ((0,), 5)
        """

class _NdIterIterator[ScalarT: Scalar, ItemT](Iterator[ItemT]):
    """Broadcast lockstep iterator over one or two same-dtype arrays.

    Runtime class name ``nditer``; returned from :func:`nditer`.  This is a
    deliberately restricted subset of NumPy ``numpy.nditer``—no flags, no
    write-back, no arbitrary operand count.

    OPERAND RULES.      The sole argument must be a ``tuple`` of one or two
    sdnp ``Array`` objects (not bare scalars, not NumPy arrays).  Every operand
    must have ``ndim >= 1``, the same dtype, and shapes that broadcast
    together under sdnp rules.  Violations raise ``TypeError`` or
    ``ValueError`` at construction.

    FULL MATERIALIZATION AT CREATION.  The Python binding calls the Rust
    ``nditer`` kernel (which can stream in the core), but immediately collects
    all broadcast steps into a ``Vec<Vec<PyScalar>>``.  Each ``__next__``
    pops one precomputed step.  Memory is O(product of broadcast shape) times
    operand count; construction pays the full traversal cost up front.

    YIELD SHAPE.  One operand: each step is a bare scalar (not a 1-tuple)—
    ``list(nditer((a,)))`` looks like ``[1, 2, 3]``.  Two operands: each step
    is a 2-tuple of scalars aligned at the same broadcast coordinate.  All
    values are 0-D unwrapped scalars.

    ORDER.  Broadcast C-order—last axis of the broadcast result varies fastest.
    Non-contiguous operands follow logical element order, not raw buffer order.

    NO ``__len__``.  Calling ``len(nditer(...))`` raises ``TypeError``.

    Yields
    ------
    scalar or tuple of scalar
        Next aligned element(s) at one broadcast coordinate.

    See Also
    --------
    nditer : factory function

    Examples
    --------
    >>> a = sdnp.array([1, 2])
    >>> b = sdnp.array([10, 20])
    >>> list(sdnp.nditer((a, b)))
    [(1, 10), (2, 20)]
    >>> left = sdnp.array([[1], [2]])
    >>> right = sdnp.array([[10, 20, 30]])
    >>> list(sdnp.nditer((left, right)))[0]
    (1, 10)
    """

    @override
    def __iter__(self) -> _NdIterIterator[ScalarT, ItemT]:
        """Return this iterator.

        Returns
        -------
        nditer
            ``self``.

        Examples
        --------
        >>> it = sdnp.nditer((sdnp.array([1]),))
        >>> iter(it) is it
        True
        """

    @override
    def __next__(self) -> ItemT:
        """Return the next aligned value or value tuple.

        Returns
        -------
        scalar or tuple of scalar
            One scalar for one operand, or a pair for two operands.

        Raises
        ------
        StopIteration
            When the broadcast iteration is complete.

        Examples
        --------
        >>> next(sdnp.nditer((sdnp.array([3]),)))
        3
        """

def ndindex(shape: Shape) -> _NdIndexIterator:
    """Return a C-order iterator over every multi-index for one shape.

    This is the sdnp equivalent of ``numpy.ndindex``.  The factory parses
    ``shape`` (an ``int`` or ``Sequence[int]``), allocates a Rust ``NdIndex``
    state object, and wraps it for Python.  Iteration is streaming: tuples
    are produced one at a time with O(1) amortized cost per step, and
    ``len(iterator)`` tracks how many remain.

    CALLING CONVENTION.  NumPy allows ``np.ndindex(2, 3)`` (variadic dimensions).
    sdnp accepts exactly one ``shape`` argument.  Use ``ndindex((2, 3))`` for
    2-D, or ``ndindex(5)`` for a 1-D shape (yielding ``(0,)``, ``(1,)``, …).

    COORDINATE ORDER.  Last axis varies fastest—the same convention as C-order
    flattening, :func:`ndenumerate`, and :func:`nditer`.  For shape ``(2, 3)``
    the sequence begins ``(0, 0), (0, 1), (0, 2), (1, 0), …``.

    EMPTY SHAPES.  If any dimension is zero, the product is zero and the
    iterator is immediately exhausted with ``len == 0``.

    COMPLEXITY.  O(product(shape)) steps total, O(rank) auxiliary memory, no
    up-front
    tuple buffer.

    Parameters
    ----------
    shape : int or sequence of int
        Iteration bounds supplied as a single argument.

    Returns
    -------
    ndindex
        Streaming C-order multi-index iterator supporting ``__len__``.

    Raises
    ------
    TypeError
        If shape components are not integers.
    ValueError
        If dimensions are invalid or overflow allocation limits.

    See Also
    --------
    ndenumerate : attach array values to each index
    _NdIndexIterator : iterator protocol details

    Examples
    --------
    >>> list(sdnp.ndindex((2, 2)))
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> list(sdnp.ndindex(3))
    [(0,), (1,), (2,)]
    >>> len(sdnp.ndindex((2, 3)))
    6
    """

def ndenumerate(a: Array[ScalarT]) -> _NdEnumerateIterator[ScalarT]:
    """Return C-order ``(index, value)`` pairs for every element of ``a``.

    Equivalent to ``numpy.ndenumerate``.  At construction time the implementation
    flattens the array and walks indices in lockstep—see
    :class:`_NdEnumerateIterator` for the full materialization model.

    The source must have ``ndim >= 1``; 0-D arrays raise ``TypeError``.  Each
    value is a Python scalar (0-D unwrap).  Non-contiguous arrays are copied to
    C-order during the flatten step, so pair ordering follows logical
    C-order of the view, not parent buffer layout.

    After construction, mutating ``a`` does not change what the iterator yields.
    There is no ``len(ndenumerate(a))``.

    Parameters
    ----------
    a : Array
        Source array with at least one dimension.

    Returns
    -------
    ndenumerate
        Iterator of ``(index_tuple, scalar)`` pairs.

    Raises
    ------
    TypeError
        If ``a`` is zero-dimensional.
    ValueError
        If shape/index setup fails in the Rust core.

    See Also
    --------
    ndindex : indices without values
    Array.flat : flat values without indices

    Examples
    --------
    >>> list(sdnp.ndenumerate(sdnp.array([10, 20])))
    [((0,), 10), ((1,), 20)]
    >>> view = sdnp.arange(6).reshape(2, 3)[:, ::-1]
    >>> next(sdnp.ndenumerate(view))
    ((0, 0), 2)
    """

@overload
def nditer(
    operands: tuple[Array[ScalarT]],
) -> _NdIterIterator[ScalarT, ScalarT]: ...
@overload
def nditer(
    operands: tuple[Array[ScalarT], Array[ScalarT]],
) -> _NdIterIterator[ScalarT, tuple[ScalarT, ScalarT]]: ...
def nditer(
    operands: tuple[Array[ScalarT]] | tuple[Array[ScalarT], Array[ScalarT]],
) -> _NdIterIterator[ScalarT, ScalarT | tuple[ScalarT, ScalarT]]:
    """Return a broadcast lockstep iterator over one or two arrays.

    Restricted port of ``numpy.nditer`` for sdnp's four dtypes.  Pass operands
    as a tuple—``nditer((a,))`` or ``nditer((a, b))``—not as separate
    arguments.

    VALIDATION AT CONSTRUCTION.  The tuple length must be 1 or 2.  Each entry
    must be an ``Array`` with ``ndim >= 1``.  All operands must share one
    dtype and broadcast to a common shape; otherwise ``ValueError``.  Non-array
    entries raise ``TypeError`` (``nditer must be an sdnp.Array``).

    EXECUTION MODEL.  The Rust core can iterate broadcast coordinates with a
    contiguous fast path when every operand is C-contiguous, but the Python
    wrapper collects the entire traversal into a vector before returning control.
    Expect O(product of broadcast shape) time and memory at creation; ``__next__``
    is cheap.

    YIELD RULES.  One operand -> bare scalars.  Two operands -> 2-tuples of
    scalars at aligned positions.  Empty broadcast shape yields an exhausted
    iterator immediately.

    NumPy's full ``nditer`` (flags, op axes, write-back) is not implemented.

    Parameters
    ----------
    operands : tuple of one or two Array
        Same-dtype, broadcast-compatible sources.

    Returns
    -------
    nditer
        Materialized broadcast iterator.

    Raises
    ------
    TypeError
        If an operand is not an array or is zero-dimensional.
    ValueError
        If operand count is not one or two, dtypes differ, or shapes cannot
        broadcast.

    See Also
    --------
    _NdIterIterator : protocol and materialization details

    Examples
    --------
    >>> a = sdnp.array([1, 2])
    >>> list(sdnp.nditer((a, a)))
    [(1, 1), (2, 2)]
    >>> list(sdnp.nditer((sdnp.arange(6).reshape(2, 3)[:, ::-1],)))
    [2, 1, 0, 5, 4, 3]
    """
