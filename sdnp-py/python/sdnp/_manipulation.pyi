"""Array joining and stacking operations.

These functions validate rank, axis, dtype, and non-concatenated dimensions,
then allocate one new C-contiguous destination buffer.  Inputs are copied in
logical order, so views and non-contiguous arrays are accepted but the result
does not alias them.

SHAPE RULES.  ``concatenate`` joins existing axes; ``stack`` inserts a new
axis; ``vstack`` and ``hstack`` apply their vector/matrix conventions before
joining.  Inputs must use the same sdnp storage dtype rather than relying on
implicit promotion.

COMPLEXITY.  Time and output space are O(total input elements).  Empty inputs
and zero-length axes retain validated geometry, while incompatible shapes or
invalid axes raise before allocation.
"""

from collections.abc import Sequence

from ._array import Array, ScalarT

def concatenate(
    arrays: Sequence[Array[ScalarT]], axis: int = 0
) -> Array[ScalarT]:
    """Join arrays along an existing axis.

    Shape or layout transformation is handled in the Rust core.  Some paths return zero-copy views (stride metadata only); others allocate a new buffer—see the summary above for the specific operation.

    Parameters
    ----------
    arrays : sequence of Array
        Nonempty inputs sharing dtype and all non-join dimensions.
    axis : int, optional
        Existing axis along which to join.

    Returns
    -------
    Array
        Concatenated result.

    Raises
    ------
    TypeError
        If an element is not an array.
    ValueError
        If shapes or dtypes differ incompatibly.

    Examples
    --------
    >>> a = sdnp.array([1, 2])
    >>> sdnp.concatenate([a, a]).to_list()
    [1, 2, 1, 2]
    """

def stack(arrays: Sequence[Array[ScalarT]], axis: int = 0) -> Array[ScalarT]:
    """Join equal-shaped arrays along a new axis.

    Shape or layout transformation is handled in the Rust core.  Some paths return zero-copy views (stride metadata only); others allocate a new buffer—see the summary above for the specific operation.

    Parameters
    ----------
    arrays : sequence of Array
        Nonempty inputs with identical shapes and dtypes.
    axis : int, optional
        Position at which to insert the new axis.

    Returns
    -------
    Array
        Result whose rank is one greater than each input.

    Raises
    ------
    TypeError
        If an element is not an array.
    ValueError
        If shapes, dtypes, or ``axis`` are incompatible.

    Examples
    --------
    >>> sdnp.stack([sdnp.array([1, 2]), sdnp.array([3, 4])]).shape
    [2, 2]
    """

def vstack(arrays: Sequence[Array[ScalarT]]) -> Array[ScalarT]:
    """Stack one- or two-dimensional arrays vertically.

    Inputs are concatenated along axis 0 into a new C-contiguous buffer.
    All arrays must share dtype and matching trailing dimensions (equal
    width for 2-D rows).  Cost is O(total elements) for the copy.

    Parameters
    ----------
    arrays : sequence of Array
        Inputs with matching widths and identical dtypes.

    Returns
    -------
    Array
        Row-wise joined result.

    Raises
    ------
    TypeError
        If an element is not an array.
    ValueError
        If ranks, widths, or dtypes are incompatible.

    Examples
    --------
    >>> sdnp.vstack([sdnp.array([1, 2]), sdnp.array([3, 4])]).to_list()
    [[1, 2], [3, 4]]
    """

def hstack(arrays: Sequence[Array[ScalarT]]) -> Array[ScalarT]:
    """Stack one- or two-dimensional arrays horizontally.

    Concatenates along the last axis (axis 1 for matrices, axis 0 for
    vectors) into a newly allocated buffer in O(total elements) time.
    Dtype and compatible leading dimensions must match across inputs.

    Parameters
    ----------
    arrays : sequence of Array
        Inputs with matching heights and identical dtypes.

    Returns
    -------
    Array
        Column-wise or one-dimensional joined result.

    Raises
    ------
    TypeError
        If an element is not an array.
    ValueError
        If ranks, heights, or dtypes are incompatible.

    Examples
    --------
    >>> sdnp.hstack([sdnp.array([1, 2]), sdnp.array([3, 4])]).to_list()
    [1, 2, 3, 4]
    """
