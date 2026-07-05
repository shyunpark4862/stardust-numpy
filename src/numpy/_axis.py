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

"""Axis index normalization for reductions, indexing, and shape ops.

Converts negative axis indices and validates bounds for single axes, axis
tuples, and insert-axis positions used by ``expand_dims`` and friends.
"""

from __future__ import annotations

from ._errors import (
    axes_must_be_int_or_tuple,
    axis_must_be_int,
    axis_out_of_range,
    duplicate_axis,
    insert_axis_out_of_range,
)


def normalize_axis(
    axis: int,
    ndim: int,
) -> int:
    """
    Convert an axis index to a non-negative position within ``ndim``.

    Internal helper used by reductions, shape manipulation, selection,
    linear algebra, and manipulation routines.

    Parameters
    ----------
    axis : int
        Axis index. Negative values count from the last dimension.
    ndim : int
        Number of dimensions of the array the axis refers to.

    Returns
    -------
    int
        Normalized axis in the range ``[0, ndim)``.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer.
    ValueError
        If the normalized axis is out of range for ``ndim``.

    Notes
    -----
    Unlike NumPy, only a single integer axis is accepted here; tuple axes
    are handled by :func:`normalize_axes`.
    """
    if not isinstance(axis, int):
        raise axis_must_be_int()

    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise axis_out_of_range(axis, ndim)

    return axis


def normalize_axes(
    axes: int | tuple[int, ...],
    ndim: int,
) -> tuple[int, ...]:
    """
    Normalize one or more axis indices and reject duplicates.

    Internal helper used by reduction and shape-manipulation code paths
    that accept a tuple of axes.

    Parameters
    ----------
    axes : int or tuple of int
        One or more axis indices. Negative values count from the last
        dimension.
    ndim : int
        Number of dimensions of the array the axes refer to.

    Returns
    -------
    tuple of int
        Normalized axes, each in the range ``[0, ndim)``.

    Raises
    ------
    TypeError
        If ``axes`` is neither an integer nor a tuple of integers.
    ValueError
        If any normalized axis is out of range or if duplicate axes
        appear after normalization.

    Notes
    -----
    NumPy allows ``axis=None`` to reduce over all dimensions; that
    convention is handled by callers before invoking this helper.
    """
    if isinstance(axes, int):
        return (normalize_axis(axes, ndim),)

    if not isinstance(axes, tuple):
        raise axes_must_be_int_or_tuple()

    normalized = tuple(normalize_axis(item, ndim) for item in axes)

    if len(set(normalized)) != len(normalized):
        raise duplicate_axis()

    return normalized


def normalize_insert_axis(
    axis: int,
    ndim: int,
) -> int:
    """
    Normalize an axis index for inserting a new dimension.

    Internal helper used by :func:`numpy.manipulation.expand_dims` and
    :func:`numpy.manipulation.stack`.

    Parameters
    ----------
    axis : int
        Position at which a length-1 dimension will be inserted. Negative
        values count from the end, with ``-1`` meaning after the last
        existing axis.
    ndim : int
        Number of dimensions before insertion.

    Returns
    -------
    int
        Normalized insertion position in the range ``[0, ndim]``.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer.
    ValueError
        If the normalized axis is outside ``[0, ndim]``.

    Notes
    -----
    Insertion axes permit one more valid position than reduction axes
    because a new axis may be placed at the end of the shape tuple.
    """
    if not isinstance(axis, int):
        raise axis_must_be_int()

    if axis < 0:
        axis += ndim + 1

    if axis < 0 or axis > ndim:
        raise insert_axis_out_of_range(axis, ndim)

    return axis
