################################################################################
#### MIT License                                                            ####
################################################################################
#
# Copyright (c) 2026 shyunpark4862@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
################################################################################

"""Linear algebra utilities built on ``ndarray`` methods.

Extracts diagonals, traces, vector dot products, and outer products without
a separate matrix type.
"""

from __future__ import annotations

from typing import Any

from ._axis import normalize_axis
from ._dtype import promote_dtypes, require_numeric_dtype
from ._errors import axes_must_differ, requires_at_least_2d, requires_integer
from ._factory import contiguous_ndarray
from ._iter import iter_indices
from ._validation import coerce_to_ndarray
from .ndarray import ndarray


def diagonal(
    a: ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> ndarray:
    """Return specified diagonals of an array.

    Parameters
    ----------
    a : array_like
        Input array. Must have at least two dimensions.
    offset : int (default: 0)
        Offset of the diagonal from the main diagonal. Positive values
        refer to diagonals above the main diagonal; negative values refer
        to diagonals below.
    axis1 : int (default: 0)
        First axis of the two axes that form the 2-D sub-arrays from which
        diagonals are taken.
    axis2 : int (default: 1)
        Second axis of the two axes that form the 2-D sub-arrays from which
        diagonals are taken.

    Returns
    -------
    ndarray
        The diagonals. If `a` is two-dimensional, a one-dimensional array is
        returned. Otherwise, the shape is the outer shape (all axes except
        ``axis1`` and ``axis2``) followed by the diagonal length.

    Raises
    ------
    ValueError
        If `a` has fewer than two dimensions or if ``axis1 == axis2``.
    TypeError
        If `offset` is not an integer.

    Notes
    -----
    Unlike NumPy, only the five supported element dtypes are allowed. There
    is no ``out`` parameter. Negative axes are normalized using the same
    rules as :func:`normalize_axis`.

    See Also
    --------
    trace : Sum along diagonals.
    """
    a = coerce_to_ndarray(a)
    offset = _validate_diagonal_offset(offset, name="diagonal")
    axis1, axis2 = _validate_diagonal_axes(
        a.ndim,
        axis1,
        axis2,
        name="diagonal",
    )

    positions = _diagonal_positions(
        a.shape[axis1],
        a.shape[axis2],
        offset,
    )
    outer_axes = _outer_axes(a.ndim, axis1, axis2)
    outer_shape = tuple(a.shape[axis] for axis in outer_axes)
    result_shape = outer_shape + (len(positions),)
    result_buffer = []

    for outer_index in iter_indices(outer_shape):
        for position1, position2 in positions:
            source_index = _build_diagonal_index(
                a.ndim,
                outer_axes,
                outer_index,
                axis1,
                axis2,
                position1,
                position2,
            )
            result_buffer.append(a._get_value(source_index))

    return contiguous_ndarray(result_buffer, result_shape, dtype=a.dtype)


def trace(
    a: ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> Any | ndarray:
    """Return the sum along diagonals of an array.

    Parameters
    ----------
    a : array_like
        Input array. Must have at least two dimensions.
    offset : int (default: 0)
        Offset of the diagonal from the main diagonal.
    axis1 : int (default: 0)
        First axis of the diagonal.
    axis2 : int (default: 1)
        Second axis of the diagonal.

    Returns
    -------
    scalar or ndarray
        When `a` is two-dimensional, a scalar sum is returned. For
        higher-dimensional inputs, the sum is taken along diagonals for each
        position of the remaining axes, producing an array whose shape
        excludes ``axis1`` and ``axis2``.

    Raises
    ------
    ValueError
        If `a` has fewer than two dimensions or if ``axis1 == axis2``.
    TypeError
        If `offset` is not an integer.

    Notes
    -----
    Unlike NumPy, there is no ``dtype`` keyword and no ``out`` parameter.
    The return type is a Python scalar for 2-D inputs and an ``ndarray``
    otherwise.

    See Also
    --------
    diagonal : Return diagonals without summing.
    """
    a = coerce_to_ndarray(a)
    offset = _validate_diagonal_offset(offset, name="trace")
    axis1, axis2 = _validate_diagonal_axes(
        a.ndim,
        axis1,
        axis2,
        name="trace",
    )

    positions = _diagonal_positions(
        a.shape[axis1],
        a.shape[axis2],
        offset,
    )
    outer_axes = _outer_axes(a.ndim, axis1, axis2)
    outer_shape = tuple(a.shape[axis] for axis in outer_axes)

    if not outer_shape:
        # 2-D input: accumulate into a Python scalar, not a 0-D array.
        total = 0

        for position1, position2 in positions:
            source_index = _build_diagonal_index(
                a.ndim,
                outer_axes,
                (),
                axis1,
                axis2,
                position1,
                position2,
            )
            total += a._get_value(source_index)

        return total

    result_buffer = []

    for outer_index in iter_indices(outer_shape):
        total = 0

        for position1, position2 in positions:
            source_index = _build_diagonal_index(
                a.ndim,
                outer_axes,
                outer_index,
                axis1,
                axis2,
                position1,
                position2,
            )
            total += a._get_value(source_index)

        result_buffer.append(total)

    return contiguous_ndarray(result_buffer, outer_shape, dtype=a.dtype)


def vdot(
    a: ndarray,
    b: ndarray,
) -> Any:
    """Return the dot product of two vectors, conjugating the first.

    Both operands are flattened before multiplication. For complex inputs,
    the first operand is conjugated element-wise, matching the BLAS
    definition of the vector dot product.

    Parameters
    ----------
    a : array_like
        First argument. Conjugated when elements are complex.
    b : array_like
        Second argument.

    Returns
    -------
    scalar
        The dot product as a Python scalar (``int``, ``float``, or
        ``complex``).

    Raises
    ------
    ValueError
        If the flattened operands do not have the same size.
    TypeError
        If either operand does not have a numeric dtype.

    Notes
    -----
    This is a thin wrapper around :meth:`ndarray.vdot`. Unlike NumPy's
    ``numpy.vdot``, only 1-D semantics are implemented (inputs are always
    raveled). There is no broadcasting of higher-dimensional arrays.

    See Also
    --------
    ndarray.dot : General dot product with matrix support.
    outer : Outer product of two vectors.
    """
    a = coerce_to_ndarray(a)
    b = coerce_to_ndarray(b, name="b")

    return a.vdot(b)


def outer(
    a: ndarray,
    b: ndarray,
) -> ndarray:
    """Compute the outer product of two vectors.

    Parameters
    ----------
    a : array_like
        First input vector. Multi-dimensional inputs are raveled first.
    b : array_like
        Second input vector. Multi-dimensional inputs are raveled first.

    Returns
    -------
    ndarray
        Two-dimensional array of shape ``(a.size, b.size)`` with entries
        ``result[i, j] = a[i] * b[j]``.

    Raises
    ------
    TypeError
        If either operand does not have a numeric dtype.

    Notes
    -----
    Unlike NumPy, inputs are always flattened to 1-D before the product
    is formed; the original shapes are not preserved. There is no ``out``
    parameter. Only numeric dtypes are supported.
    """
    a = coerce_to_ndarray(a).ravel()
    b = coerce_to_ndarray(b, name="b").ravel()
    require_numeric_dtype(a.dtype, operation="outer")
    require_numeric_dtype(b.dtype, operation="outer")

    result_dtype = promote_dtypes(a.dtype, b.dtype)
    a_values = a._flat_buffer_slice()
    b_values = b._flat_buffer_slice()
    result_buffer = [
        left_value * right_value
        for left_value in a_values
        for right_value in b_values
    ]

    return contiguous_ndarray(
        result_buffer,
        (a.size, b.size),
        dtype=result_dtype,
    )


def _validate_diagonal_offset(
    offset: int,
    *,
    name: str,
) -> int:
    """Validate that a diagonal offset is an integer.

    Used by :func:`diagonal` and :func:`trace`.

    Parameters
    ----------
    offset : int
        Diagonal offset from the main diagonal.
    name : str
        Function name for error messages.

    Returns
    -------
    int
        The validated offset.

    Raises
    ------
    TypeError
        If ``offset`` is not an integer.
    """
    if not isinstance(offset, int):
        raise requires_integer(name=name, parameter="offset")

    return offset


def _validate_diagonal_axes(
    ndim: int,
    axis1: int,
    axis2: int,
    *,
    name: str,
) -> tuple[int, int]:
    """Normalize and validate the axis pair for diagonal/trace helpers.

    Used by :func:`diagonal` and :func:`trace`.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the input array.
    axis1, axis2 : int
        Axis indices (may be negative).
    name : str
        Function name for error messages.

    Returns
    -------
    axis1, axis2 : int
        Normalized, distinct axis indices.

    Raises
    ------
    ValueError
        If ``ndim < 2`` or ``axis1 == axis2``.
    """
    if ndim < 2:
        raise requires_at_least_2d(name=name)

    axis1 = normalize_axis(axis1, ndim)
    axis2 = normalize_axis(axis2, ndim)

    if axis1 == axis2:
        raise axes_must_differ()

    return axis1, axis2


def _diagonal_positions(
    dim1: int,
    dim2: int,
    offset: int,
) -> list[tuple[int, int]]:
    """List ``(row, col)`` pairs along a diagonal with the given offset.

    Computes all valid positions on the ``offset``-th diagonal of a
    ``dim1 × dim2`` matrix. Shared by :func:`diagonal` and :func:`trace`.

    Parameters
    ----------
    dim1, dim2 : int
        Sizes of the two axes that form the 2-D sub-array.
    offset : int
        Diagonal offset. Positive values refer to diagonals above the
        main diagonal; negative values refer to diagonals below.

    Returns
    -------
    list of (int, int)
        Ordered ``(position1, position2)`` pairs traversing the diagonal.

    Notes
    -----
    Uses the same starting row ``max(0, -offset)`` as NumPy's
    :func:`numpy.diagonal`.
    """
    row = max(0, -offset)
    positions = []

    while row < dim1 and row + offset < dim2:
        positions.append((row, row + offset))
        row += 1

    return positions


def _outer_axes(
    ndim: int,
    axis1: int,
    axis2: int,
) -> list[int]:
    """Return axes other than the two diagonal axes, in ascending order.

    Used by :func:`diagonal` and :func:`trace` to iterate over batch
    (outer) dimensions while fixing ``axis1`` and ``axis2``.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the input array.
    axis1, axis2 : int
        The two diagonal axes (already normalized).

    Returns
    -------
    list of int
        All axis indices except ``axis1`` and ``axis2``.
    """
    return [axis for axis in range(ndim) if axis not in (axis1, axis2)]


def _build_diagonal_index(
    ndim: int,
    outer_axes: list[int],
    outer_index: tuple[int, ...],
    axis1: int,
    axis2: int,
    position1: int,
    position2: int,
) -> tuple[int, ...]:
    """Assemble a full ndim index from outer coordinates and diagonal positions.

    Combines batch-axis coordinates from ``outer_index`` with the two
    diagonal positions to produce an index suitable for
    :meth:`ndarray._get_value`.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the source array.
    outer_axes : list of int
        Axes not involved in the diagonal (from :func:`_outer_axes`).
    outer_index : tuple of int
        Coordinates along the outer axes for one batch element.
    axis1, axis2 : int
        The two diagonal axes.
    position1, position2 : int
        Row and column positions on the selected diagonal.

    Returns
    -------
    tuple of int
        Full ``ndim``-length index into the source array.
    """
    index = [0] * ndim

    for axis, value in zip(outer_axes, outer_index):
        index[axis] = value

    index[axis1] = position1
    index[axis2] = position2

    return tuple(index)
