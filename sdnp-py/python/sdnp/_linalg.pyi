"""Linear algebra contractions and diagonal operations.

Operands are coerced to arrays, validated for contraction geometry, promoted
to a common numeric dtype, and dispatched to typed Rust kernels.  Internal
rank-0 contraction results are unwrapped to Python scalars.  Pure bool
contractions are rejected by the Python API.

CONTRACTIONS.  ``dot`` handles vector and matrix cases, ``matmul`` adds NumPy
batch broadcasting, ``vdot`` flattens both inputs and conjugates the left
complex operand, and ``outer`` flattens both inputs before forming every pair.
Matrix multiplication costs O(product(batch) * M * N * K) and allocates a
C-contiguous output.  Strided inputs work; the right operand may be copied when
its contraction layout prevents the optimized kernel.

DIAGONALS.  ``diagonal`` gathers selected values into owned output, while
``trace`` sums them and may unwrap to a scalar.  Negative axes are normalized,
and the two selected axes must be distinct.

NUMPY DIFFERENCES.  Advanced BLAS configuration, ``out=``, arbitrary dot ranks,
and exposed 0-D arrays are not supported.
"""

from ._array import Array, ArrayLike, ArrayResult, Scalar, ScalarT

def dot(
    left: ArrayLike[Scalar],
    right: ArrayLike[Scalar],
) -> ArrayResult[Scalar]:
    """Compute a NumPy-style dot product.

    For 1-D inputs this is an inner product that may unwrap to a scalar; for
    2-D inputs it is matrix multiplication without batch dimensions.  Higher
    ranks follow NumPy's ``dot`` stacking rules, which differ from
    :func:`matmul`'s explicit batch-broadcast semantics—check operand
    dimensionality when porting code.  Boolean dtype is rejected.  The Rust
    path dtype-promotes, then dispatches to typed contraction kernels whose
    cost is O(m * n * k) for matrix-matrix cases.

    Parameters
    ----------
    left, right : array-like
        Inputs with compatible contraction dimensions.

    Returns
    -------
    Array or scalar
        Dtype-promoted dot product.

    Raises
    ------
    TypeError
        If input coercion fails.
    ValueError
        If dimensions are incompatible or boolean dtype is selected.

    Examples
    --------
    >>> sdnp.dot(sdnp.array([1, 2, 3]), sdnp.array([4, 5, 6]))
    32
    """

def matmul(
    left: ArrayLike[Scalar],
    right: ArrayLike[Scalar],
) -> ArrayResult[Scalar]:
    """Compute NumPy-style batched matrix multiplication (``@``).

    ``matmul`` implements the same tensor-contraction rules as ``numpy.matmul``
    and the ``@`` operator: the last axis of ``left`` and the second-to-last
    axis of ``right`` (or the only axis when ``right`` is 1-D) are summed, and
    leading batch dimensions are broadcast together.  It is the right choice
    for stack-of-matrices work; use :func:`dot` only when both operands have
    rank at most 2 (NumPy ``dot`` legacy semantics).

    PIPELINE.  Each operand passes through ``coerce_array_like`` (nested lists
    become arrays; bare scalars become rejected 0-D buffers).  ``check_matmul``
    validates ranks and shapes at the Python boundary.  Operands dtype-promote
    (``bool < int < float < complex``), cast to the common storage, and enter
    a monomorphized ``sdnp::matmul`` kernel.  Results with no remaining axes
    unwrap to a Python scalar.

    CONTRACTION GEOMETRY.  Let ``K`` be the shared inner length.

    * ``(M, K) @ (K, N)`` -> ``(M, N)`` matrix multiply.
    * ``(M, K) @ (K,)`` -> ``(M,)`` matrix–vector product.
    * ``(K,) @ (K, N)`` -> ``(N,)`` vector–matrix product.
    * ``(K,) @ (K,)`` -> scalar dot product (0-D unwrap to ``int``/``float``/etc.).

    One-dimensional operands are treated as row or column vectors without
    copying memory—the planner records virtual ``(1, K)`` or ``(K, 1)`` faces
    and strips the length-1 axis from the output shape afterward.

    BATCHED INPUTS.  For ranks greater than 2, all leading axes form batch
    dimensions that must broadcast NumPy-style (size 1 repeats; otherwise sizes
    must match).  Example: ``(2, 1, 3, 4) @ (1, 5, 4, 2)`` yields shape
    ``(2, 5, 3, 2)``.  Incompatible batch prefixes raise
    ``ValueError: matmul batch dimensions are not broadcast-compatible``.

    INNER DIMENSION.  The trailing width of ``left`` must equal the contraction
    axis of ``right``—for a 2-D right operand that is ``shape[-2]``; for a 1-D
    right operand it is ``shape[0]``.  Mismatch raises
    ``ValueError: matmul inner dimensions differ: …``.

    DTYPES.  After promotion, kernels run as ``i64``, ``f64``, or ``Complex64``.
    Mixed numeric dtypes widen before the multiply-accumulate loop.  Pure
    ``bool @ bool`` is rejected at the Python layer with ``matmul dtype
    mismatch`` even though the Rust core defines a boolean semiring internally—
    combine with ``int``/``float`` if you need mask-style arithmetic.  Complex
    multiplication does not conjugate operands (contrast :func:`vdot`).

    IMPLEMENTATION AND COST.  ``MatmulPlan`` derives batch strides, ``M``, ``N``,
    and ``K`` from operand shapes without reshaping vectors.  If the right
    operand's last axis is not C-contiguous (stride ``1``), it may be copied
    once before contraction.  The ``contract`` kernel fills a fresh C-order
    output buffer, walking batch tiles with ``RunPlan`` and using an IKJ inner
    loop when the right column stride is unit.  Time is
    O(product(batch) * M * N * K)
    element-wise multiply-adds; space is O(output size) for the new buffer.
    Non-contiguous inputs are supported (strided views multiply correctly).

    EMPTY CONTRACTIONS.  When ``K == 0``, the result shape is still computed and
    filled with zeros—e.g. ``(2, 0) @ (0, 3)`` -> ``(2, 3)`` of zeros—matching
    NumPy.

    ZERO-DIMENSIONAL OPERANDS.  Scalars and 0-D arrays are rejected (
    ``matmul does not support 0-D operands``).  Use explicit ``shape=(1,)`` or
    ``shape=(1, 1)`` vectors instead of bare numbers.

    OPERATOR EQUIVALENCE.  ``a @ b`` calls the same implementation as
    ``matmul(a, b)`` when both sides are arrays.  ``__rmatmul__`` swaps order.

    Parameters
    ----------
    left, right : array-like
        Factors to multiply.  Nested sequences coerce to ``Array``; both must
        end with ``ndim >= 1``.

    Returns
    -------
    Array or scalar
        Broadcast batch shape plus ``M`` and/or ``N`` matrix axes as applicable;
        a Python scalar when both inputs are 1-D vectors.

    Raises
    ------
    TypeError
        If coercion fails or a 0-D array appears after coercion.
    ValueError
        Inner or batch shape mismatch; ``bool @ bool``; core allocation or
        contraction failure.

    See Also
    --------
    dot : rank <= 2 only, no arbitrary batch stacking
    __matmul__ : ``@`` on ``Array``
    vdot : flattened inner product with complex conjugation on the left

    Examples
    --------
    Matrix–matrix::

        >>> sdnp.matmul(sdnp.eye(2), sdnp.ones((2, 1))).shape
        [2, 1]

    Vector dot (scalar result)::

        >>> sdnp.matmul(sdnp.array([1, 2, 3]), sdnp.array([4, 5, 6]))
        32

    Batched stacks::

        >>> a = sdnp.array([[[1, 2], [3, 4]]])
        >>> b = sdnp.array([[[5, 6], [7, 8]], [[1, 0], [0, 1]]])
        >>> sdnp.matmul(a, b).shape
        [2, 2, 2]

    Empty inner dimension::

        >>> sdnp.matmul(sdnp.zeros((2, 0)), sdnp.zeros((0, 3))).shape
        [2, 3]
    """

def vdot(
    left: ArrayLike[Scalar],
    right: ArrayLike[Scalar],
) -> Scalar:
    """Compute a flattened conjugating inner product.

    ``vdot`` ignores the original shapes and traverses both operands as
    logical C-order vectors.  The vectors need not have the same rank or
    shape, but their total element counts must match.  This is a flattened
    contraction, not a broadcasting operation.

    EXECUTION.  Python sequences are first coerced to sdnp arrays.  The two
    dtypes are promoted using ``bool < int < float < complex`` and either
    operand is cast when promotion requires it.  The Rust kernel obtains a
    contiguous C-order representation of each operand, borrowing storage when
    possible and materializing a temporary for a non-contiguous view.  It then
    accumulates products through eight independent lanes before combining the
    partial sums.  Each complex value from ``left`` is conjugated immediately
    before multiplication; ``right`` is never conjugated.

    RESULT AND COST.  For ``n`` flattened elements, contraction takes O(n)
    time and uses O(1) kernel workspace.  Coercion, dtype conversion, or
    contiguous materialization can additionally require O(n) temporary
    storage.  The Rust core creates an internal 0-D result, and the Python
    boundary always unwraps it to a native scalar.  Empty inputs return the
    additive zero of the promoted dtype.

    NUMPY DIFFERENCES AND CAVEATS.  The flattening and left-side conjugation
    match ``numpy.vdot``, but sdnp supports only its four built-in dtypes and
    has no ``out=`` option.  Two boolean operands are rejected; a boolean
    operand combined with a numeric operand is promoted and accepted.
    Eight-lane accumulation may associate floating-point additions differently
    from NumPy, so the final low-order bits need not be identical.

    Parameters
    ----------
    left, right : array-like
        Inputs with equal flattened sizes. Their original dimensions do not
        participate in the contraction.

    Returns
    -------
    scalar
        Promoted sum of ``conj(left.ravel()) * right.ravel()``.

    Raises
    ------
    TypeError
        If either value cannot be coerced to a supported array.
    ValueError
        If flattened sizes differ, both operands are boolean, or the Rust
        contraction fails.

    Examples
    --------
    >>> sdnp.vdot(sdnp.array([1, 2]), sdnp.array([3, 4]))
    11

    Shapes are ignored after flattening::

        >>> sdnp.vdot([[1, 2], [3, 4]], [1, 1, 1, 1])
        10

    The left complex operand is conjugated::

        >>> sdnp.vdot([1 + 2j, 3 - 1j], [2j, 4 + 0j])
        (16+6j)
    """

def outer(
    left: ArrayLike[Scalar],
    right: ArrayLike[Scalar],
) -> Array[Scalar]:
    """Compute the outer product after flattening both operands.

    ``outer`` traverses each input in logical C order and pairs every flattened
    left value with every flattened right value.  If the flattened lengths are
    ``m`` and ``n``, element ``(i, j)`` of the result is
    ``left.ravel()[i] * right.ravel()[j]``.  Unlike ``vdot``, this operation
    performs no conjugation and requires no relationship between the input
    sizes or original shapes.

    EXECUTION.  Python sequences are coerced to sdnp arrays and their dtypes
    are promoted using ``bool < int < float < complex``.  Casting occurs before
    the typed Rust kernel runs.  Contiguous arrays can lend their storage to
    the flattening step, whereas strided or permuted views are copied into
    temporary C-order vectors.  The kernel checks ``m * n`` for shape and
    allocation overflow, reserves one output buffer, and fills it row by row.
    The output is always a fresh C-contiguous two-dimensional array and never
    aliases either operand.

    COST AND EDGE CASES.  The operation requires O(m * n) time and output
    space.  Input coercion, promotion, and contiguous materialization can add
    O(m + n) temporary space.  If either flattened input is empty, the result
    still has shape ``(m, n)`` and contains no elements; it is not unwrapped
    because its rank is always two.

    NUMPY DIFFERENCES AND CAVEATS.  These flattening and output-shape rules
    match ``numpy.outer``.  sdnp has no ``out=`` option and only supports bool,
    int, float, and complex storage.  Two boolean operands are rejected because
    contraction kernels do not define bool multiplication, while bool combined
    with a numeric dtype is promoted and accepted.  Complex values are
    multiplied directly, without conjugating either side.

    Parameters
    ----------
    left, right : array-like
        Operands of any rank. Each is flattened independently in logical
        C order.

    Returns
    -------
    Array
        Fresh array with shape ``(left.size, right.size)`` and the promoted
        dtype.

    Raises
    ------
    TypeError
        If either value cannot be coerced to a supported array.
    ValueError
        If both operands are boolean or the output size cannot be represented
        or allocated.

    Examples
    --------
    >>> sdnp.outer(sdnp.array([1, 2]), sdnp.array([3, 4])).to_list()
    [[3, 4], [6, 8]]

    Higher-dimensional inputs are flattened first::

        >>> sdnp.outer([[1, 2], [3, 4]], [10, 20]).shape
        [4, 2]

    Complex values are not conjugated::

        >>> sdnp.outer([1j], [2j]).to_list()
        [[(-2+0j)]]
    """

def diagonal(
    a: Array[ScalarT],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> Array[ScalarT]:
    """Extract a diagonal between two axes.

    The Rust core returns a zero-copy strided view over the plane spanned by
    ``axis1`` and ``axis2``.  Remaining axes retain their original order and
    the diagonal is appended as the final axis with stride
    ``stride(axis1) + stride(axis2)``.  Construction takes O(ndim) time and
    O(ndim) metadata space regardless of the number of diagonal elements.

    The view shares the input buffer.  As with other sdnp views, mutation uses
    copy-on-write: writing to the diagonal view detaches it rather than
    modifying the source array.

    Parameters
    ----------
    a : Array
        Input with at least two dimensions.
    offset : int, optional
        Offset from the main diagonal.
    axis1, axis2 : int, optional
        Distinct axes defining the diagonal plane.

    Returns
    -------
    Array
        Zero-copy diagonal view with unchanged dtype.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If axes are equal, invalid, or input has too few dimensions.

    Examples
    --------
    >>> sdnp.diagonal(sdnp.array([[1, 2], [3, 4]])).to_list()
    [1, 4]
    """

def trace(
    a: Array[ScalarT],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> ArrayResult[ScalarT | int]:
    """Sum a diagonal between two axes.

    Linear-algebra work runs in typed Rust kernels after shape and dtype validation at the Python boundary.  Boolean dtype is generally rejected.

    Parameters
    ----------
    a : Array
        Input with at least two dimensions.
    offset : int, optional
        Offset from the main diagonal.
    axis1, axis2 : int, optional
        Distinct axes defining the diagonal plane.

    Returns
    -------
    Array or scalar
        Diagonal sum; boolean input promotes to int64.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If axes are equal, invalid, or input has too few dimensions.

    Examples
    --------
    >>> sdnp.trace(sdnp.array([[1, 2], [3, 4]]))
    5
    """
