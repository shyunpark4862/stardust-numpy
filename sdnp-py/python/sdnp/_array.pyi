"""Array types, common typing aliases, and built-in iteration protocols.

The compiled extension stores each ``Array`` in one of four monomorphized
Rust buffers: ``bool``, ``i64``, ``f64``, or ``Complex64``.  Shape and stride
metadata are measured in elements and describe a logical C-order array over an
``Arc<Vec<T>>`` allocation.  Basic slicing and axis permutation can return
shared-buffer views; mutation first detaches shared storage through
copy-on-write.

PUBLIC SHAPE POLICY.  Python callers only receive arrays with one or more
dimensions.  Internal 0-D results are unwrapped to native Python scalars.
Consequently indexing, reductions, squeezing, and operators use
``ArrayResult[T]`` where a scalar result is possible.

VIEW AND COPY COSTS.  Metadata-only operations such as transpose and many
basic slices are O(1).  ``copy()``, dtype conversion, flattening, and
non-contiguous reshape paths require O(n) time and storage.  Fancy and boolean
indexing always materialize owned C-contiguous output.

CONTIGUOUS FAST PATH.  A C-contiguous array stores logical C-order neighbors
next to one another in the backing buffer.  Its innermost non-trivial axis has
stride ``1``, and each outer stride equals the next inner stride multiplied by
the next inner axis length.  Element-wise kernels can therefore borrow one
flat slice and process it without reconstructing N-dimensional coordinates.
This is the cheapest traversal path, but it still allocates a new result for
operations whose API promises owned output.

STRIDE COALESCING.  Non-contiguous input does not immediately force a copy.
After broadcasting has aligned every operand to one logical shape, the Rust
core builds a joint ``CoalescedLayout``.  Axes of length ``1`` are discarded
because advancing them cannot change a buffer position.  Starting at the
innermost axis and moving outward, two adjacent axes are merged when every
operand satisfies
``outer_stride == inner_stride * inner_axis_length``.  This equality proves
that replacing the nested loops with one fixed-stride loop visits exactly the
same elements in logical C order.  All operands must satisfy it together:
a transpose, slice, or broadcast pattern in either operand can stop a merge
that would have been valid for the other operand alone.  Checked arithmetic is
used while combining axis lengths and strides; overflow simply leaves the axes
separate instead of risking an incorrect traversal.

RUN PLANNING.  ``RunPlan`` turns the coalesced layout into an outer run grid
and one innermost fixed-stride run.  A ``StrideCursor`` advances the base
offsets between grid cells without materializing coordinate tuples.  Within
each run, stride ``1`` is classified as ``UnitStride`` and enables direct slice
loops, stride ``0`` is ``Repeated`` and reuses one broadcast value, and every
other stride is ``Strided`` and advances a running buffer offset.  Negative
strides can still coalesce when the same equality holds, but they use the
general strided loop rather than the unit-stride slice path.  A transposed
layout commonly leaves several shorter runs because adjacent strides no
longer satisfy the merge relation.

PERFORMANCE CONSEQUENCES.  Constructing the plan costs O(ndim * operands) time
and small metadata storage; visiting the data remains O(n).  Coalescing does
not improve asymptotic complexity, but it replaces repeated index unraveling
and deeply nested axis loops with a much smaller number of branch-light inner
runs.  Ufuncs, reductions, logical C-order collection, iteration helpers, and
several indexing paths share this strategy.  It preserves view semantics and
often avoids an O(n) contiguous copy, although kernels that specifically
require flat storage may still materialize one.

NUMPY DIFFERENCES.  ``shape`` and ``strides`` return lists, strides count
elements rather than bytes, only four dtypes exist, 0-D arrays are not exposed,
and views use stricter copy-on-write mutation semantics.
"""

from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import Literal, TypeVar, overload, override

type Scalar = bool | int | float | complex
ScalarT = TypeVar("ScalarT", bound=Scalar)
DTypeT = TypeVar("DTypeT", bool, int, float, complex)
NumericT = TypeVar("NumericT", int, float, complex)
type RealScalar = bool | int | float
type DType[ScalarT: Scalar] = type[ScalarT]
type Shape = int | Sequence[int]
type Axis = int | Sequence[int]
type IndexAtom = int | slice | None | EllipsisType | Array[bool] | Array[int]
type Index = IndexAtom | tuple[IndexAtom, ...]
type ArrayResult[ScalarT: Scalar] = Array[ScalarT] | ScalarT
type NestedList[ScalarT: Scalar] = list[ScalarT | NestedList[ScalarT]]
type ArrayLike[ScalarT: Scalar] = (
    Array[ScalarT] | Sequence[ScalarT | ArrayLike[ScalarT]]
)
type Operand[ScalarT: Scalar] = Array[ScalarT] | ScalarT
type AnyArray = Array[bool] | Array[int] | Array[float] | Array[complex]
type AnyOperand = AnyArray | Scalar
type NanPolicy = Literal["propagate", "ignore"]
type MeshgridIndexing = Literal["xy", "ij"]

class Array[ScalarT: Scalar]:
    """Homogeneous N-dimensional array backed by Rust ``Array<T>``.

    Each instance wraps one of four monomorphized storages (``bool``,
    ``i64``, ``f64``, ``Complex64``) in row-major layout with per-axis
    strides counted in elements, not bytes.  You cannot call ``Array(...)``
    from Python—no ``#[new]`` constructor is registered—so create arrays
    through :func:`array`, :func:`zeros`, :func:`ones`, and the other module
    factories.

    Slicing and many view operations share the underlying ``Arc`` buffer.
    Indexed assignment via :meth:`__setitem__` may detach that buffer through
    copy-on-write before writing, so two Python names never silently alias
    the same mutable storage the way NumPy sometimes allows.  Dunder operators
    delegate to the same dtype-promotion and monomorphized ufunc kernels as
    module-level functions like :func:`add`; scalar operands broadcast, and
    comparisons return ``Array[bool]`` or a bare ``bool`` when the result
    would be rank-0.

    There is no ``.flags``, ``.base``, ``.data``, or ``np.dtype`` metadata—
    :attr:`dtype` is a Python ``type``.  Prefer contiguous inputs for ufuncs
    and reductions; non-contiguous arrays still work but may walk coalesced
    stride segments instead of a single flat loop.  Use :meth:`copy` when you
    need a writable C-contiguous owned buffer before heavy in-place indexing.

    Parameters
    ----------
    None
        This class cannot be instantiated from Python.

    Attributes
    ----------
    shape, strides, ndim, size, dtype
        Metadata accessors (raise on internal 0-D buffers).
    T
        Swap the last two axes (matrix transpose for 2-D).
    flat
        C-order scalar iterator (materialized at creation).

    See Also
    --------
    array, zeros, ones

    Examples
    --------
    >>> import sdnp
    >>> a = sdnp.array([[1, 2], [3, 4]])
    >>> a.shape
    [2, 2]
    >>> (a + 1).to_list()
    [[2, 3], [4, 5]]
    >>> a.transpose().to_list()
    [[1, 3], [2, 4]]
    """

    @property
    def shape(self) -> list[int]:
        """Return the length of each axis as a Python list.

        The Rust core exposes ``Array<T>::shape`` directly, so this read is
        O(1) and never touches the data buffer.  NumPy returns an immutable
        ``tuple``; sdnp allocates a fresh ``list[int]`` on every access, so
        mutating the returned list does not change the array.  Public
        ``Array`` instances always have ``ndim >= 1``—rank-0 buffers exist
        only inside ufunc plumbing and raise ``TypeError`` here if they leak.

        Returns
        -------
        list of int
            ``[shape[0], shape[1], ...]`` in C-order axis order.

        Raises
        ------
        TypeError
            If the array is zero-dimensional (internal error if seen from
            public API).

        Examples
        --------
        >>> sdnp.array([[1, 2], [3, 4]]).shape
        [2, 2]
        """

    @property
    def strides(self) -> list[int]:
        """Return per-axis strides measured in elements, not bytes.

        Strides are fixed when an array is constructed or viewed and describe
        how far to advance in the flat buffer for a unit step along each
        axis.  Broadcast-expanded axes may carry stride ``0``.  Ufunc and
        reduction kernels use these values to detect C-contiguity—a trailing
        stride of ``1`` marks a contiguous tail—so understanding element
        strides matters for performance tuning.  NumPy's ``ndarray.strides``
        are in bytes; multiply sdnp strides by the element size (1 for
        ``bool``, 8 for ``int``/``float``/``complex`` components) if you need
        byte distances for mental comparison with NumPy.

        Returns
        -------
        list of int
            Distance in elements between adjacent indices along each axis.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> sdnp.zeros((2, 3)).strides
        [3, 1]
        """

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions (rank).

        This is ``len(shape)`` and is always at least 1 for arrays visible
        from Python.  NumPy exposes 0-D ``ndarray`` objects with
        ``ndim == 0``; sdnp unwraps rank-0 results to scalars instead, so
        you should never see a user-facing array with zero dimensions.

        Returns
        -------
        int
            ``len(shape)``, always ``>= 1`` for user-visible arrays.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> sdnp.zeros((2, 3)).ndim
        2
        """

    @property
    def size(self) -> int:
        """Return the total element count (product of ``shape``).

        The value is the product of all shape dimensions.  Overflow during
        that product is checked in the Rust core at array creation time and
        surfaces as ``ValueError`` there—not on this property access, which
        simply reads the already-validated metadata.

        Returns
        -------
        int
            Number of scalar elements in the array.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> sdnp.zeros((2, 3)).size
        6
        """

    @property
    def dtype(self) -> DType[ScalarT]:
        """Return the Python type object for this array's storage dtype.

        The tagged Rust storage enum maps to one of ``bool``, ``int``,
        ``float``, or ``complex`` via ``PyDType::python_type``.  NumPy
        returns rich ``np.dtype`` objects with itemsize and kind metadata;
        sdnp exposes only the four fixed storages, where ``int`` always means
        signed 64-bit ``i64``, ``float`` means ``f64``, and ``complex`` means
        ``complex128``.  Factory functions accept these types or their
        ``__name__`` strings (``"int"``, ``"float"``, …)—not ``np.float32``
        or structured dtypes.

        Returns
        -------
        type
            One of ``bool``, ``int``, ``float``, ``complex``.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> sdnp.array([1, 2]).dtype is int
        True
        >>> sdnp.zeros(3).dtype is float
        True
        """

    @property
    def T(self) -> Array[ScalarT]:
        """Return a view with the last two axes exchanged (matrix transpose).

        This calls ``Array::transpose`` in the Rust core, swapping only the
        trailing two axes in O(1) by exchanging shape and stride metadata—no
        element movement.  For ``ndim < 2`` the shape is unchanged, matching
        NumPy.  Unlike NumPy's ``.T``, which reverses all axes for 1-D arrays,
        sdnp only swaps the last pair; use :meth:`permute_axes` for general
        reordering.  Subsequent in-place writes through the view may trigger
        copy-on-write detachment of the shared buffer.

        Returns
        -------
        Array
            Array sharing or copying storage depending on subsequent writes;
            dtype unchanged.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        See Also
        --------
        transpose, permute_axes

        Examples
        --------
        >>> sdnp.zeros((2, 3)).T.shape
        [3, 2]
        """

    @property
    def flat(self) -> flatiter[ScalarT]:
        """Return a C-order scalar iterator over every element.

        Accessing ``a.flat`` constructs a :class:`flatiter` immediately.  The
        Rust core calls ``Array::to_vec()``, which always materializes logical
        C-order elements—non-contiguous and transposed arrays pay a full O(n)
        copy even when the source is a view.  The iterator then walks an owned
        ``IntoIter``; it does not read live from the parent buffer.  Mutating
        ``a`` after creating ``flat`` therefore does not change what the
        iterator yields.

        NumPy's ``ndarray.flat`` is a mutable, array-aware iterator that
        supports assignment (``a.flat[i] = x``) and can reflect some updates to
        the base array.  sdnp's ``flatiter`` is read-only forward iteration
        only—no ``__getitem__`` / ``__setitem__`` on the iterator object.  For
        large arrays, prefer ufuncs, reductions, or slice-based access instead
        of ``list(a.flat)``.

        See :class:`flatiter` for protocol details.

        Returns
        -------
        flatiter
            Iterator yielding native scalars (never 0-D ``Array`` objects).

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> list(sdnp.array([[1, 2], [3, 4]]).flat)
        [1, 2, 3, 4]
        """

    def copy(self) -> Array[ScalarT]:
        """Return a deep, C-contiguous, independently owned copy.

        The Rust core always allocates a fresh ``Arc`` buffer in C-contiguous
        layout via ``Array::copy``, regardless of how non-contiguous or
        view-based the source is.  Writes to the copy never affect the
        original.  If you plan heavy indexed writes on a view that still
        shares storage with other live references, calling ``copy()`` once
        upfront avoids repeated copy-on-write detachments during assignment.
        NumPy's ``copy()`` respects order flags; sdnp always normalizes to
        C-contiguous owned storage.

        Returns
        -------
        Array
            New array; writes to the copy never affect the original.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> a = sdnp.array([1, 2])
        >>> b = a.copy()
        >>> b[0] = 9
        >>> a[0]
        1
        """

    def astype(self, dtype: type[DTypeT]) -> Array[DTypeT]:
        """Cast every element to a new storage dtype.

        The Rust ``cast_inner`` path always allocates new storage through
        typed ``Array::astype`` kernels—there is no view cast.  NumPy's
        ``astype(copy=False)`` may return a view when dtypes are compatible;
        sdnp always copies.  Casting ``float`` with NaN to ``int`` follows
        Rust cast semantics and may raise ``ValueError`` on invalid values.

        Parameters
        ----------
        dtype : {bool, int, float, complex}
            Target type object (not ``np.dtype``).

        Returns
        -------
        Array
            Newly allocated array; the input is never modified in place.

        Raises
        ------
        TypeError
            If ``dtype`` is unsupported or the array is zero-dimensional.
        ValueError
            If the cast pair is unsupported or an element conversion fails.

        Examples
        --------
        >>> sdnp.array([1.2, 2.8]).astype(int).to_list()
        [1, 2]
        """

    def reshape(self, shape: Shape) -> Array[ScalarT]:
        """Give the array a new shape without changing total element count.

        When the source layout is C-contiguous, ``reshape`` returns a
        zero-copy view with recalculated strides in O(1).  Non-contiguous
        inputs are copied to a contiguous buffer first, so chained reshapes
        on transposed or sliced arrays can silently incur O(n) copies—keep
        arrays C-contiguous beforehand when performance matters.  NumPy
        supports ``reshape(order='F')``; sdnp always follows C-order layout
        semantics.

        Parameters
        ----------
        shape : int or sequence of int
            New dimensions.  One dimension may be ``-1`` (inferred).

        Returns
        -------
        Array
            Reshaped array with the same dtype and element count.

        Raises
        ------
        TypeError
            If shape components are not integers.
        ValueError
            If the product of dimensions does not match ``size``.

        Examples
        --------
        >>> sdnp.array([1, 2, 3, 4]).reshape((2, 2)).shape
        [2, 2]
        """

    def squeeze(self, axis: Axis | None = None) -> ArrayResult[ScalarT]:
        """Remove length-1 axes from the shape.

        The Rust core adjusts shape and strides in place when possible,
        producing a view without copying elements.  When every axis is
        removed—e.g. ``squeeze()`` on a ``(1,)`` vector—the result unwraps
        to a Python scalar rather than a 0-D ``Array``, unlike NumPy which
        would return a rank-0 ``ndarray``.

        Parameters
        ----------
        axis : int, sequence of int, or None, optional
            Axes to squeeze.  ``None`` removes all singleton dimensions.

        Returns
        -------
        Array or scalar
            Squeezed result.  If all axes are removed, returns a Python
            scalar (0-D unwrap), not an ``Array``.

        Raises
        ------
        TypeError
            If ``axis`` is not integer-like.
        ValueError
            If a selected axis does not have length 1 or is out of range.

        Examples
        --------
        >>> sdnp.zeros((1, 3, 1)).squeeze().shape
        [3]
        """

    def transpose(self) -> Array[ScalarT]:
        """Swap the final two axes (same semantics as :attr:`T`).

        This is the method form of the ``.T`` property and calls the same
        ``transpose_inner`` Rust path: an O(1) stride swap with no element
        movement.  Subsequent writes through the returned view may trigger
        copy-on-write detachment of the shared buffer.

        Returns
        -------
        Array
            View with trailing axes exchanged; dtype unchanged.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        See Also
        --------
        T, permute_axes

        Examples
        --------
        >>> sdnp.zeros((2, 3)).transpose().shape
        [3, 2]
        """

    def permute_axes(self, axes: Axis) -> Array[ScalarT]:
        """Reorder axes by a full permutation.

        Rust ``Array::permute_axes`` recomputes shape and strides from the
        supplied permutation, typically producing a zero-copy view in O(ndim)
        metadata work without moving elements.  Unlike ``np.transpose(a, axes)``,
        sdnp requires a complete permutation of ``range(ndim)``—each axis
        index exactly once.  Writes through the permuted view may later
        CoW-detach the shared buffer.

        Parameters
        ----------
        axes : int or sequence of int
            Permutation of ``range(ndim)`` — each axis index exactly once.

        Returns
        -------
        Array
            View with rearranged shape/strides when possible.

        Raises
        ------
        TypeError
            If an entry is not an integer.
        ValueError
            If axes are duplicated, missing, or out of range.

        Examples
        --------
        >>> sdnp.zeros((2, 3, 4)).permute_axes((2, 0, 1)).shape
        [4, 2, 3]
        """

    def to_list(self) -> NestedList[ScalarT]:
        """Recursively convert the array to nested Python lists.

        The implementation walks axis 0 recursively, calling ``gather`` for
        sub-arrays and boxing scalars via ``scalar_from_item``.  Complex
        values become ``complex`` objects.  This allocates a Python object
        per element—O(n) Python overhead—so it is fine for debugging or
        small arrays but unsuitable for hot loops on large data; keep
        computation in ``Array`` form instead.  Behavior is similar to
        NumPy's ``tolist()``, except sdnp has no ``item()`` on 0-D arrays
        because rank-0 buffers are never exposed.

        Returns
        -------
        list
            Nested lists mirroring ``shape``; leaves are Python scalars.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> sdnp.array([[1, 2], [3, 4]]).to_list()
        [[1, 2], [3, 4]]
        """

    @override
    def __repr__(self) -> str:
        """Return a NumPy-style ``array(...)`` representation string.

        Formatting walks the array structure in Rust and mirrors NumPy's
        visual style.  Long one-dimensional arrays may be abbreviated with
        an ellipsis to keep output readable.

        Returns
        -------
        str
            Text beginning with ``array(``.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> repr(sdnp.array([1, 2])).startswith("array(")
        True
        """

    @override
    def __str__(self) -> str:
        """Return the same human-readable text as :func:`repr`.

        Returns
        -------
        str
            NumPy-style array representation.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> a = sdnp.array([1, 2])
        >>> str(a) == repr(a)
        True
        """

    @overload
    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> ArrayResult[ScalarT]: ...
    @overload
    def __getitem__(self, index: Index) -> ArrayResult[ScalarT]: ...
    def __getitem__(self, index: Index) -> ArrayResult[ScalarT]:
        """Select elements using NumPy-style basic, fancy, and boolean indexing.

        Indexing is one of the most intricate parts of the sdnp Python surface.
        Every ``a[index]`` call walks the same pipeline: the Python object is
        parsed into a flat list of core ``IndexSpec`` entries in
        ``index_parse.rs``, structurally validated against the source
        ``shape``, normalized and expanded in the Rust ``prepare_index``
        pass (ellipsis fill, boolean-to-coordinate conversion, negative-index
        resolution, fancy broadcast), and finally executed by ``gather`` in the
        ``sdnp`` crate.  The result is wrapped as a new ``Array`` or unwrapped
        to a native Python scalar when the output rank would be zero.

        WHAT YOU CAN PASS.  At the Python boundary a top-level index is
        either a single component or a ``tuple`` of components.  There is no
        extra wrapping layer—a bare ``int`` indexes axis 0 directly, while
        ``(i, j)`` is equivalent to passing two axis slots in order.  Each
        component may be:

        * an ``int`` (including negative wrap-around indices);
        * a ``slice`` with integer ``start`` / ``stop`` / ``step`` (``None``
          components keep NumPy defaults; ``step == 0`` is rejected);
        * ``...`` (``Ellipsis``), which may appear at most once and stands
          in for a run of full-axis slices ``:`` on every axis not yet
          consumed by earlier slots;
        * ``None``, which inserts a length-1 newaxis (use this instead of
          ``numpy.newaxis``—sdnp does not expose that constant);
        * an ``Array`` used as a fancy index, which must have ``dtype is
          int`` or ``dtype is bool`` and ``ndim >= 1``.

        Anything else raises ``TypeError`` with a message beginning
        ``index must be int, slice, ellipsis, None, or array``.  In particular
        float indices, complex indices, strings, plain Python
        ``list`` / ``tuple`` index arrays, and 0-D index arrays are
        rejected before the core runs.  Fancy indices must be sdnp ``Array``
        objects, not NumPy ``ndarray`` instances (there is no buffer-protocol
        interop).

        BASIC INDEXING (VIEWS WITHOUT COPYING).  When the prepared index
        contains only integers, slices, ``NewAxis``, and expanded ellipsis
        (no integer-array or boolean-array slots), ``gather`` takes the
        basic path: it computes output ``shape``, ``strides``, and a buffer
        ``offset`` and returns an ``Array`` that shares the parent's
        ``Arc`` storage.  Integer indices collapse the corresponding source
        axis into the buffer offset rather than producing an output axis—so
        ``a[0, 1]`` on a ``(2, 2)`` matrix yields a scalar, while ``a[0]``
        yields a 1-D view.  Slices preserve an axis (possibly with reversed or
        strided layout).  ``None`` inserts ``shape == 1`` with stride 0.
        Trailing axes you omit are implicitly filled with full slices, matching
        NumPy's rule that ``a[0]`` on a 2-D array is ``a[0, :]``.

        Because basic results are views, reading through them is O(1) setup
        plus whatever work your downstream kernel performs; no element copy
        occurs at index time.  If you later mutate a basic view (or assign
        through it via :meth:`__setitem__`), copy-on-write may detach the
        shared buffer first so other Python names holding the original array
        are not silently updated—see :meth:`__setitem__` below.

        FANCY INTEGER INDEXING (ALWAYS COPIES).  When any index slot is an
        ``Array[int]``, preparation broadcasts all integer index arrays to
        a common shape (same rules as element-wise ufuncs).  The fancy gather
        path allocates a new C-contiguous buffer and copies source elements
        in broadcast-result order—O(k) time and memory for *k* output elements.
        Output rank combines (a) axes from slices and ``newaxis`` slots and
        (b) the broadcast fancy shape; the relative ordering follows NumPy's
        adjacent-vs-separated fancy rules recorded in ``FancyLayout``.  The
        returned array never aliases the parent; mutating it does not
        affect the source, and vice versa.

        BOOLEAN INDEXING (MASKS BECOME COORDINATE LISTS).  A boolean index
        array must match the exact shape of the slice of source axes it
        replaces.  For example, on a ``(3, 4, 5)`` array, a mask of shape
        ``(4, 5)`` can follow a leading integer or slice slot that fixes axis
        0, but a mask of shape ``(3, 4)`` cannot replace axes 1–2 unless axis
        0 was already consumed.  At validation time sdnp checks
        ``mask.shape == source.shape[axis:axis+mask.ndim]``; mismatch raises
        ``IndexError: boolean index shape ... does not match indexed
        dimensions ...``.  During preparation each mask is converted internally
        to integer coordinate arrays (one per mask axis) listing the ``True``
        positions; execution then follows the fancy copy path.  Boolean fancy
        results are independent copies, like integer fancy results.

        OUTPUT TYPE (ARRAY VS SCALAR).  sdnp never exposes 0-D ``Array``
        objects to Python.  When every indexed axis is an integer (after
        ellipsis expansion)—for example ``a[1, 2]`` or ``a[(-1, -1)]`` on a 2-D
        array—the gathered 0-D buffer is unwrapped to ``bool``, ``int``,
        ``float``, or ``complex``.  Any remaining output axis yields an
        ``Array`` with ``ndim >= 1``.  Type checkers see this split via the
        overloads above: pure slice / ``None`` / ``...`` paths return
        ``Array[ScalarT]``; integer and general ``Index`` paths return
        ``ArrayResult[ScalarT]`` (``Array | scalar``).

        STRUCTURAL VALIDATION ERRORS (BEFORE GATHER).  Besides type errors,
        parsing enforces: at most one ellipsis; not more index axes consumed
        than ``a.ndim`` (otherwise ``IndexError: too many indices``); boolean
        shape alignment as described; resolved integer indices within bounds
        (``IndexError`` on out-of-range); non-zero slice step (``ValueError``).
        Integer fancy arrays that cannot broadcast together surface as
        ``ValueError`` from the core broadcast layer.

        PERFORMANCE.  Prefer basic slicing on hot paths—``a[i:j]``,
        ``a[:, k]``, ``a[..., None]``—to avoid O(k) allocations.  Fancy and
        boolean selections are correct and NumPy-compatible but materialize
        data.  Repeated fancy reads should be cached in a variable if the
        selected region is reused.

        Parameters
        ----------
        index : int, slice, ellipsis, None, tuple, or Array
            NumPy-style index.  See the prose above for the full grammar.

        Returns
        -------
        Array or scalar
            Selected values.  Basic slice/newaxis/ellipsis paths return a
            view sharing storage; fancy/boolean paths return an owned
            copy.  Fully integer indices unwrap to a Python scalar.

        Raises
        ------
        TypeError
            Invalid index type; 0-D fancy index array; float/complex fancy
            index; indexing a leaked 0-D array.
        IndexError
            Out-of-bounds integer index; too many indices; duplicate ellipsis;
            boolean mask shape mismatch.
        ValueError
            Zero slice step; fancy broadcast failure; core gather error.

        See Also
        --------
        __setitem__ : write through compatible index expressions

        Examples
        --------
        Basic integer, slice, and scalar unwrap::

            >>> a = sdnp.array([[1, 2], [3, 4]])
            >>> a[0, 1]          # scalar unwrap
            2
            >>> a[:, 0].to_list()  # view along axis 1
            [1, 3]
            >>> a[::-1, ::-1].to_list()  # reversed view, same buffer
            [[4, 3], [2, 1]]

        Newaxis and ellipsis::

            >>> sdnp.arange(6).reshape(2, 3)[None, ..., 0].shape
            [1, 2]

        Fancy integer indexing (copy)::

            >>> src = sdnp.array([[10, 20], [30, 40]])
            >>> picked = src[sdnp.array([1, 0])]
            >>> picked.to_list()
            [[30, 40], [10, 20]]

        Boolean mask (copy)::

            >>> v = sdnp.array([1, 2, 3, 4])
            >>> v[sdnp.array([True, False, True, False])].to_list()
            [1, 3]
        """

    def __setitem__(
        self, index: Index, value: ScalarT | Array[ScalarT]
    ) -> None:
        """Assign a scalar or broadcastable array to indexed locations in place.

        Assignment mirrors the read path documented in :meth:`__getitem__` up
        to the core dispatch boundary: the index is parsed into ``IndexSpec``,
        validated, prepared (ellipsis expansion, boolean conversion, fancy
        broadcast), and then applied via ``scatter`` (scalar) or
        ``scatter_array`` (array value).  The target ``Array`` is mutated in
        place when writable; there is no ``out=`` parameter and no silent
        return of a new array.

        SCALAR ASSIGNMENT AND DTYPE COERCION.  When ``value`` is not an
        ``Array``, it is coerced through the same scalar rules as ufuncs.
        Same-dtype writes go straight to the core.  Cross-dtype scalar
        assignment follows NumPy-like widening and narrowing at the Python
        boundary before ``scatter`` runs—for example ``float`` into ``int``
        truncates, ``bool`` into ``int`` becomes ``0``/``1``, and real scalars
        into ``complex`` gain a zero imaginary part.  The notable exception is
        boolean storage: only ``bool`` scalars (or ``int`` interpreted as
        ``0``/non-zero) may be written; assigning ``float`` or ``complex`` to
        a ``bool`` array raises ``ValueError: cannot assign value to bool
        array``.  Array-valued assignment with mismatched dtype is handled by
        casting the source array to the destination dtype via
        ``cast_inner`` and retrying recursively.

        ARRAY ASSIGNMENT AND BROADCASTING.  When ``value`` is an ``Array``,
        it must have ``ndim >= 1`` (0-D value arrays are rejected).  The core
        broadcasts ``value`` to the shape of the indexed region—the same
        shape :meth:`__getitem__` would return for that index—then copies
        element-wise into the selected locations.  NumPy-style leading,
        trailing, and middle broadcasting are supported (see tests with shapes
        like ``(1, 2, 5)`` into a ``(3, 4, 5)`` slice).  If the broadcast
        cannot be aligned, ``ValueError`` is raised.

        BASIC VS FANCY WRITE PATHS.  As with reads, indices without integer
        or boolean arrays take the basic scatter path: the indexed region is
        treated as a strided view into the destination buffer.  Contiguous
        selections use bulk ``fill`` or ``copy_from_slice`` fast paths; general
        strided selections walk coalesced stride runs.  Indices that include
        fancy integer arrays or boolean masks (after internal coordinate
        expansion) take the fancy scatter path, which iterates every
        selected buffer offset in fancy-result order and writes there.  Both
        paths require the destination to be writable.

        COPY-ON-WRITE BEFORE MUTATION.  sdnp arrays share ``Arc`` buffers
        between views.  Before any in-place scatter, the core calls
        ``ensure_unique_storage_for_write``: if this ``Array`` still shares
        its buffer with other live views, the buffer is copied once and
        only this handle is detached.  That is why assigning through a slice
        view of ``b = a[::-1]`` updates ``b`` but leaves ``a`` unchanged when
        they previously shared storage—you get NumPy-like view-read semantics
        with stricter write isolation than some NumPy write-sharing cases.

        OVERLAPPING SELF-ASSIGNMENT.  When the assignment source array
        shares a buffer with the destination—as in ``a[:, 1:] = a[:, :-1]``—the
        core detects buffer overlap and copies the source values first
        (``prepare_scatter_source``) before writing, so elements are not read
        after they have been overwritten.  This matches NumPy's overlapping
        assignment behavior.

        READ-ONLY TARGETS.  Broadcast-expanded views (for example grids
        from :func:`meshgrid`) are marked read-only at the Rust level.
        Attempting ``grid[...] = 0`` raises ``ValueError: read-only array``.
        Fancy read results are independent copies and are writable if you
        hold the only reference, but assigning into a fancy slice of the
        original still follows the parent's writability and CoW rules.

        WHAT IS NOT SUPPORTED.  The same index restrictions as
        :meth:`__getitem__` apply (no float indices, no 0-D index arrays, at
        most one ellipsis, boolean shape must match).  There is no multi-field
        ``a[indices] = values`` with mismatched fancy shapes beyond broadcast
        rules, and no ``a[mask] = other_array`` where ``other_array`` has a
        different dtype without an explicit cast path through ``cast_inner``.

        Parameters
        ----------
        index : int, slice, ellipsis, None, tuple, or Array
            Destination selection; same grammar as :meth:`__getitem__`.
        value : scalar or Array
            Value(s) to write.  Scalars are coerced; arrays are broadcast to
            the indexed output shape.

        Raises
        ------
        TypeError
            Invalid index type; 0-D index or value array; assignment to a 0-D
            target array.
        IndexError
            Out-of-bounds index; boolean mask shape mismatch; too many indices.
        ValueError
            Incompatible broadcast shape; bool-array dtype violation;
            read-only destination; slice step zero; scatter/cast failure.

        See Also
        --------
        __getitem__ : read path, view-vs-copy rules, index grammar

        Examples
        --------
        Scalar write through basic indexing::

            >>> a = sdnp.zeros((2, 2))
            >>> a[0, :] = 1.0
            >>> a.to_list()
            [[1.0, 1.0], [0.0, 0.0]]

        Array assignment with broadcasting::

            >>> b = sdnp.zeros((3, 4, 5))
            >>> b[1, 1:3, :] = sdnp.ones((1, 2, 5))
            >>> int(b[1, 1, 0])
            1

        Fancy and boolean assignment (copies source layout, mutates parent)::

            >>> c = sdnp.array([10, 20, 30, 40])
            >>> c[sdnp.array([True, False, True, False])] = -1
            >>> c.to_list()
            [-1, 20, -1, 40]

        Copy-on-write: view write does not affect the original::

            >>> d = sdnp.array([[1, 2], [3, 4]])
            >>> v = d[::-1]
            >>> v[...] = 0
            >>> d.to_list()
            [[1, 2], [3, 4]]
        """

    def __len__(self) -> int:
        """Return the length of axis zero (``shape[0]``).

        This is equivalent to NumPy's ``len(a)`` for arrays with at least one
        dimension.  Zero-dimensional internal buffers raise ``TypeError``.

        Returns
        -------
        int
            ``shape[0]``.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        Examples
        --------
        >>> len(sdnp.zeros((3, 2)))
        3
        """

    def __iter__(self) -> axis0iter[ScalarT]:
        """Return an iterator over leading-axis slices (``for row in a``).

        Calling ``iter(a)`` builds an :class:`axis0iter`.  At construction the
        Rust core runs ``iter_axis0().collect()``: every slice along axis 0 is
        turned into a shared-buffer view up front—O(shape[0]) view objects,
        not an O(n) element copy.  Each ``__next__`` then pops one pre-built
        view in O(1).  This design avoids holding a borrow on the parent array
        across Python iteration steps.

        For ``ndim >= 2``, each step yields an ``Array`` with rank reduced by
        one (a row, a plane front, etc.).  For a one-dimensional source,
        axis-0 slices are internally 0-D buffers that unwrap to Python
        scalars—``list(sdnp.array([1, 2, 3]))`` is ``[1, 2, 3]``, not a list
        of arrays.  Empty axis 0 (``shape[0] == 0``) yields an exhausted
        iterator immediately.

        Views yielded from ``__iter__`` share the parent's ``Arc`` buffer.
        Mutating a yielded sub-array may trigger copy-on-write on that sub-array
        handle without affecting siblings or the parent, following the same CoW
        rules as :meth:`__getitem__`.  This differs from :attr:`flat`, which
        copies all elements at iterator construction.

        Returns
        -------
        axis0iter
            Iterator over axis-0 slices or 1-D scalars.

        Raises
        ------
        TypeError
            If the array is zero-dimensional.

        See Also
        --------
        flat : C-order scalar iteration with full materialization
        axis0iter : iterator protocol details

        Examples
        --------
        >>> rows = list(sdnp.arange(6).reshape(2, 3))
        >>> rows[0].to_list()
        [0, 1, 2]
        >>> list(sdnp.array([1, 2]))
        [1, 2]
        """

    def __add__(self, other: AnyOperand) -> ArrayResult[Scalar]:
        """Add ``other`` element by element.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible right operand.

        Returns
        -------
        Array or scalar
            Dtype-promoted sum.

        Raises
        ------
        TypeError
            For an unsupported operand.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 2]) + 1).to_list()
        [2, 3]
        """

    def __sub__(self, other: AnyOperand) -> ArrayResult[Scalar]:
        """Subtract ``other`` element by element.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible right operand.

        Returns
        -------
        Array or scalar
            Dtype-promoted difference.

        Raises
        ------
        TypeError
            For an unsupported operand.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([3, 4]) - 1).to_list()
        [2, 3]
        """

    def __mul__(self, other: AnyOperand) -> ArrayResult[Scalar]:
        """Multiply by ``other`` element by element.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible right operand.

        Returns
        -------
        Array or scalar
            Dtype-promoted product.

        Raises
        ------
        TypeError
            For an unsupported operand.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([2, 3]) * 2).to_list()
        [4, 6]
        """

    def __truediv__(
        self, other: AnyOperand
    ) -> ArrayResult[float | complex]:
        """Divide by ``other`` element by element.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible divisor.

        Returns
        -------
        Array or scalar
            Floating or complex quotient.

        Raises
        ------
        TypeError
            For an unsupported operand.
        ValueError
            For broadcast failure or division by zero.

        Examples
        --------
        >>> (sdnp.array([2, 4]) / 2).to_list()
        [1.0, 2.0]
        """

    def __floordiv__(self, other: AnyOperand) -> ArrayResult[Scalar]:
        """Apply element-wise truncating floor division.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible divisor.

        Returns
        -------
        Array or scalar
            Promoted quotient.

        Raises
        ------
        TypeError
            For unsupported operands.
        ValueError
            For broadcast failure or division by zero.

        Examples
        --------
        >>> (sdnp.array([7, 8]) // 3).to_list()
        [2, 2]
        """

    def __mod__(self, other: AnyOperand) -> ArrayResult[Scalar]:
        """Compute the element-wise remainder.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible divisor.

        Returns
        -------
        Array or scalar
            Promoted remainder.

        Raises
        ------
        TypeError
            For unsupported operands.
        ValueError
            For broadcast failure or division by zero.

        Examples
        --------
        >>> (sdnp.array([7, 8]) % 3).to_list()
        [1, 2]
        """

    def __pow__(
        self,
        other: AnyOperand,
        modulus: None = None,
    ) -> ArrayResult[Scalar]:
        """Raise elements to powers from ``other``.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible exponents.
        modulus : None, optional
            Modular array power is not supported.

        Returns
        -------
        Array or scalar
            Promoted power result.

        Raises
        ------
        TypeError
            If a non-``None`` modulus or unsupported operand is supplied.
        ValueError
            For invalid powers or broadcast failure.

        Examples
        --------
        >>> (sdnp.array([2, 3]) ** 2).to_list()
        [4, 9]
        """

    def __neg__(self) -> ArrayResult[ScalarT]:
        """Negate every element.

        Returns
        -------
        Array or scalar
            Negated result.

        Raises
        ------
        TypeError
            If the dtype does not support negation.
        ValueError
            If the underlying operation fails.

        Examples
        --------
        >>> (-sdnp.array([1, -2])).to_list()
        [-1, 2]
        """

    def __abs__(self) -> ArrayResult[RealScalar]:
        """Return element-wise absolute values.

        Returns
        -------
        Array or scalar
            Magnitudes; complex input produces real values.

        Raises
        ------
        TypeError
            If the dtype is unsupported.
        ValueError
            If the underlying operation fails.

        Examples
        --------
        >>> abs(sdnp.array([-1, 2])).to_list()
        [1, 2]
        """

    def __matmul__(self, other: Array[Scalar]) -> ArrayResult[Scalar]:
        """Matrix-multiply this array by ``other`` (``self @ other``).

        Delegates to the same path as :func:`matmul`—see that function for
        full shape, batch, dtype, and implementation detail.  Both operands
        must already be ``Array`` instances with ``ndim >= 1``; bare Python
        scalars on the right are not accepted by the dunder (use
        :func:`matmul` if you need array-like coercion from nested lists).

        Parameters
        ----------
        other : Array
            Right-hand factor with a compatible contraction axis and
            broadcastable batch prefix.

        Returns
        -------
        Array or scalar
            ``matmul(self, other)`` with 0-D unwrap when the result is a
            scalar (e.g. two 1-D vectors).

        Raises
        ------
        TypeError
            If ``other`` is not an ``Array`` or either side is 0-D.
        ValueError
            Inner dimension mismatch, non-broadcastable batch axes, or
            unsupported dtype combination (including ``bool @ bool``).

        See Also
        --------
        matmul, __rmatmul__, dot

        Examples
        --------
        >>> a = sdnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> b = sdnp.array([[2.0], [-1.0]])
        >>> (a @ b).to_list()
        [[0.0], [2.0]]
        >>> sdnp.array([1, 2, 3]) @ sdnp.array([4, 5, 6])
        32
        """

    @override
    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: object,
    ) -> ArrayResult[bool]:
        """Compare elements for equality.

        Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible comparison operand.

        Returns
        -------
        Array or bool
            Element-wise comparison.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 2]) == 2).to_list()
        [False, True]
        """

    @override
    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: object,
    ) -> ArrayResult[bool]:
        """Compare elements for inequality.

        Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible comparison operand.

        Returns
        -------
        Array or bool
            Element-wise comparison.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 2]) != 2).to_list()
        [True, False]
        """

    def __lt__(self, other: AnyOperand) -> ArrayResult[bool]:
        """Compare whether elements are less than ``other``.

        Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible comparison operand.

        Returns
        -------
        Array or bool
            Boolean comparison result.

        Raises
        ------
        TypeError
            If operands cannot be ordered.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 3]) < 2).to_list()
        [True, False]
        """

    def __le__(self, other: AnyOperand) -> ArrayResult[bool]:
        """Compare whether elements are at most ``other``.

        Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible comparison operand.

        Returns
        -------
        Array or bool
            Boolean comparison result.

        Raises
        ------
        TypeError
            If operands cannot be ordered.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 3]) <= 1).to_list()
        [True, False]
        """

    def __gt__(self, other: AnyOperand) -> ArrayResult[bool]:
        """Compare whether elements are greater than ``other``.

        Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible comparison operand.

        Returns
        -------
        Array or bool
            Boolean comparison result.

        Raises
        ------
        TypeError
            If operands cannot be ordered.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 3]) > 2).to_list()
        [False, True]
        """

    def __ge__(self, other: AnyOperand) -> ArrayResult[bool]:
        """Compare whether elements are at least ``other``.

        Operands broadcast and promote like arithmetic ufuncs, but the kernel writes ``bool`` storage.  Rank-0 comparison results unwrap to a bare Python ``bool`` rather than a 0-D array.

        Parameters
        ----------
        other : Array or scalar
            Broadcast-compatible comparison operand.

        Returns
        -------
        Array or bool
            Boolean comparison result.

        Raises
        ------
        TypeError
            If operands cannot be ordered.
        ValueError
            If shapes cannot broadcast.

        Examples
        --------
        >>> (sdnp.array([1, 3]) >= 3).to_list()
        [False, True]
        """

    def __radd__(self, other: Scalar) -> ArrayResult[Scalar]:
        """Add this array to a scalar left operand.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Left operand.

        Returns
        -------
        Array or scalar
            Promoted element-wise sum.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            If broadcasting fails.

        Examples
        --------
        >>> (1 + sdnp.array([2, 3])).to_list()
        [3, 4]
        """

    def __rsub__(self, other: Scalar) -> ArrayResult[Scalar]:
        """Subtract this array from a scalar.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Left operand.

        Returns
        -------
        Array or scalar
            Promoted element-wise difference.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            If broadcasting fails.

        Examples
        --------
        >>> (5 - sdnp.array([2, 3])).to_list()
        [3, 2]
        """

    def __rmul__(self, other: Scalar) -> ArrayResult[Scalar]:
        """Multiply this array by a scalar left operand.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Left operand.

        Returns
        -------
        Array or scalar
            Promoted element-wise product.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            If broadcasting fails.

        Examples
        --------
        >>> (2 * sdnp.array([2, 3])).to_list()
        [4, 6]
        """

    def __rtruediv__(
        self,
        other: Scalar,
    ) -> ArrayResult[float | complex]:
        """Divide a scalar by this array element by element.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Dividend.

        Returns
        -------
        Array or scalar
            Floating or complex quotient.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            For division by zero or broadcasting failure.

        Examples
        --------
        >>> (8 / sdnp.array([2, 4])).to_list()
        [4.0, 2.0]
        """

    def __rfloordiv__(self, other: Scalar) -> ArrayResult[Scalar]:
        """Floor-divide a scalar by this array.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Dividend.

        Returns
        -------
        Array or scalar
            Promoted truncating quotient.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            For division by zero or broadcasting failure.

        Examples
        --------
        >>> (8 // sdnp.array([2, 3])).to_list()
        [4, 2]
        """

    def __rmod__(self, other: Scalar) -> ArrayResult[Scalar]:
        """Compute a scalar remainder by each array element.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Dividend.

        Returns
        -------
        Array or scalar
            Promoted remainder.

        Raises
        ------
        TypeError
            If ``other`` is unsupported.
        ValueError
            For division by zero or broadcasting failure.

        Examples
        --------
        >>> (8 % sdnp.array([3, 5])).to_list()
        [2, 3]
        """

    def __rpow__(
        self,
        other: Scalar,
        modulus: None = None,
    ) -> ArrayResult[Scalar]:
        """Raise a scalar to powers from this array.

        Operands are coerced and dtype-promoted at the Python boundary, broadcast to a common shape, and evaluated in a monomorphized Rust kernel.  C-contiguous layouts use a single O(n) pass over the output; non-contiguous inputs may walk coalesced stride segments.  Rank-0 results unwrap to Python scalars.  NumPy ``out=`` and ``where=`` kwargs are not supported.

        Parameters
        ----------
        other : scalar
            Base.
        modulus : None, optional
            Modular array power is unsupported.

        Returns
        -------
        Array or scalar
            Promoted power result.

        Raises
        ------
        TypeError
            If a modulus or unsupported operand is supplied.
        ValueError
            If a power is invalid.

        Examples
        --------
        >>> (2 ** sdnp.array([1, 3])).to_list()
        [2, 8]
        """

    def __rmatmul__(self, other: Array[Scalar]) -> ArrayResult[Scalar]:
        """Matrix-multiply with this array on the right (``other @ self``).

        Delegates to the same path as :func:`matmul`—see that function for
        full shape, batch, dtype, and implementation detail.  The runtime
        evaluates ``matmul(other, self)``, so ``other`` is the left-hand factor
        and ``self`` the right-hand factor in the contraction.  Both operands
        must already be ``Array`` instances with ``ndim >= 1``; bare Python
        scalars on the left are not accepted by the dunder (use :func:`matmul`
        if you need array-like coercion from nested lists).

        Python calls this method when ``other @ self`` is evaluated and
        ``other.__matmul__(self)`` is unavailable or returns
        ``NotImplemented``.  For two plain ``sdnp.Array`` values, the left
        array's :meth:`__matmul__` usually handles ``@`` first; ``__rmatmul__``
        matters for reversed dispatch or custom subclasses.

        Parameters
        ----------
        other : Array
            Left-hand factor (written on the left of ``@``) with a compatible
            contraction axis and broadcastable batch prefix relative to
            ``self``.

        Returns
        -------
        Array or scalar
            ``matmul(other, self)`` with 0-D unwrap when the result is a
            scalar (e.g. two 1-D vectors).

        Raises
        ------
        TypeError
            If ``other`` is not an ``Array`` or either side is 0-D.
        ValueError
            Inner dimension mismatch, non-broadcastable batch axes, or
            unsupported dtype combination (including ``bool @ bool``).

        See Also
        --------
        matmul, __matmul__, dot

        Examples
        --------
        >>> left = sdnp.array([[1.0, 2.0]])
        >>> right = sdnp.array([[3.0], [4.0]])
        >>> sdnp.matmul(left, right).to_list()  # same as left @ right
        [[11.0]]
        >>> sdnp.array([4, 5, 6]) @ sdnp.array([1, 2, 3])
        32
        """

class flatiter[ScalarT: Scalar](Iterator[ScalarT]):
    """C-order flat iterator over scalar elements of one array.

    Runtime type name ``flatiter``; obtained only from :attr:`Array.flat`.
    Direct construction is not supported.

    MATERIALIZATION AT CREATION.  ``FlatIter::new`` calls ``to_vec()`` on the
    source array, producing a dense C-order ``Vec<T>`` in O(n) time and memory
    (including the cost of walking non-contiguous layouts).  The Python wrapper
    stores a typed ``IntoIter`` over that vector.  Iteration itself is O(1) per
    ``__next__`` with no further access to the parent array.

    READ-ONLY PROTOCOL.  The type implements ``__iter__`` (returning ``self``)
    and ``__next__`` only.  There is no item assignment, no reverse iteration,
    and no ``__len__``.  Each yielded value is a native Python scalar—never a
    0-D ``Array``—via the same unwrap path as reductions.

    ORDER SEMANTICS.  Elements follow logical C-order (last axis fastest),
    identical to ``ndenumerate`` / ``nditer`` traversal.  A transposed or
    sliced view therefore yields values in flattened C-order of the view, not
    in arbitrary buffer order.

    NUMPY DIFFERENCES.  NumPy ``flat`` can write through the iterator and may
    stay connected to the base array.  sdnp detaches at creation; treat the
    iterator as a snapshot of values at the time ``a.flat`` was accessed.

    Yields
    ------
    scalar
        Next element in C-order; dtype matches the source array.

    See Also
    --------
    Array.flat : factory property
    ndenumerate : index + value pairs (also materializes)

    Examples
    --------
    >>> list(sdnp.array([[1, 2], [3, 4]]).flat)
    [1, 2, 3, 4]
    >>> view = sdnp.arange(12).reshape(3, 4)[::-1, ::2].T
    >>> list(view.flat)
    [8, 4, 0, 10, 6, 2]
    """

    @override
    def __iter__(self) -> flatiter[ScalarT]:
        """Return this iterator.

        Returns
        -------
        flatiter
            ``self``.

        Examples
        --------
        >>> it = sdnp.array([1]).flat
        >>> iter(it) is it
        True
        """

    @override
    def __next__(self) -> ScalarT:
        """Return the next flat scalar.

        Returns
        -------
        scalar
            Next C-order element.

        Raises
        ------
        StopIteration
            When all elements have been consumed.

        Examples
        --------
        >>> next(sdnp.array([7]).flat)
        7
        """

class axis0iter[ScalarT: Scalar](Iterator[ArrayResult[ScalarT]]):
    """Iterator over axis-0 slices produced by ``iter(array)``.

    Runtime type name ``axis0iter``; returned from :meth:`Array.__iter__`.

    PRE-COLLECTED VIEWS.  ``Axis0Iter::new`` calls ``iter_axis0().collect()``
    on the Rust array, building a ``Vec`` of axis-0 sub-array views before the
    first ``__next__``.  Cost is O(shape[0]) view metadata plus whatever buffer
    sharing already exists—elements are not copied.  Subsequent steps pop views
    from that vector in O(1).  The parent array need not stay borrowed across
    Python iteration because handles are already materialized.

    YIELD TYPE.  For ``ndim >= 2``, each item is an ``Array`` with rank one
    less than the source (a row for 2-D, etc.).  For ``ndim == 1``, each
    axis-0 "slice" is internally rank-0 and is unwrapped to a Python scalar at
    the Python boundary—the ordinary ``for x in a`` loop on a vector therefore
    yields numbers, not length-1 arrays.  Empty ``shape[0]`` yields nothing.

    WRITABILITY.  Yielded sub-arrays are views sharing the parent ``Arc``.
    In-place mutation on a yielded row may CoW-detach that row's handle without
    affecting other rows or the parent, matching general indexing semantics.

    PROTOCOL.  Implements ``__iter__`` (``self``) and ``__next__`` only; no
    ``__len__``.

    Yields
    ------
    Array or scalar
        Next axis-0 slice; 1-D sources yield scalars.

    See Also
    --------
    Array.__iter__ : entry point
    flatiter : scalar C-order iteration with full copy

    Examples
    --------
    >>> a = sdnp.arange(6).reshape(2, 3)
    >>> next(iter(a)).to_list()
    [0, 1, 2]
    >>> list(sdnp.array([1, 2]))
    [1, 2]
    """

    @override
    def __iter__(self) -> axis0iter[ScalarT]:
        """Return this iterator.

        Returns
        -------
        axis0iter
            ``self``.

        Examples
        --------
        >>> it = iter(sdnp.array([1]))
        >>> iter(it) is it
        True
        """

    @override
    def __next__(self) -> ArrayResult[ScalarT]:
        """Return the next leading-axis slice.

        Returns
        -------
        Array or scalar
            Next slice, unwrapped if it is zero-dimensional.

        Raises
        ------
        StopIteration
            When axis zero is exhausted.

        Examples
        --------
        >>> next(iter(sdnp.array([7])))
        7
        """
