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

"""Shape, stride, and flat-index geometry utilities.

Low-level helpers for C-order strides, contiguity checks, unravel/merge
index tuples, and shape validation shared across the package.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

from ._errors import (
    negative_dimension_not_allowed,
    shape_dimensions_must_be_integers,
    shape_must_be_int_or_tuple,
    slice_step_zero,
)


def c_order_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Compute C-order (row-major) strides for ``shape``.

    Internal helper used when constructing contiguous arrays and when
    mapping multi-dimensional indices to flat positions.

    Parameters
    ----------
    shape : tuple of int
        Array dimensions.

    Returns
    -------
    tuple of int
        Strides such that the last axis has stride ``1``. Returns an
        empty tuple when ``shape`` is empty.

    Examples
    --------
    >>> c_order_strides((2, 3))
    (3, 1)
    """
    if not shape:
        return ()

    strides = [1] * len(shape)

    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]

    return tuple(strides)


def flat_index(
    indices: tuple[int, ...],
    shape: tuple[int, ...],
) -> int:
    """
    Map a C-order multi-dimensional index to a flat buffer offset.

    Internal helper used by reductions, manipulation, indexing, and
    creation routines that copy contiguous memory blocks.

    Parameters
    ----------
    indices : tuple of int
        Position within ``shape``.
    shape : tuple of int
        Array dimensions used to compute strides.

    Returns
    -------
    int
        Flat index into a C-contiguous buffer of ``shape``.

    Examples
    --------
    >>> flat_index((1, 2), (2, 3))
    5
    """
    position = 0

    for index, stride in zip(indices, c_order_strides(shape)):
        position += index * stride

    return position


def size_of_shape(shape: tuple[int, ...]) -> int:
    """
    Return the number of elements described by ``shape``.

    Internal helper used throughout the library when allocating buffers
    or validating index ranges.

    Parameters
    ----------
    shape : tuple of int
        Array dimensions.

    Returns
    -------
    int
        Product of all dimensions. Returns ``1`` for an empty shape tuple.
    """
    return reduce(mul, shape, 1)


def is_c_contiguous(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> bool:
    """
    Return whether ``strides`` describe a C-contiguous layout for ``shape``.

    Internal helper used by ndarray layout checks and fast paths in
    elementwise, indexing, iteration, and linear algebra code.

    Parameters
    ----------
    shape : tuple of int
        Array dimensions.
    strides : tuple of int
        Per-axis strides to validate.

    Returns
    -------
    bool
        ``True`` if the layout is C-contiguous, otherwise ``False``.

    Notes
    -----
    Length-1 dimensions may have arbitrary strides, matching NumPy's
    treatment of broadcast axes in contiguous arrays.
    """
    expected_stride = 1

    for axis in range(len(shape) - 1, -1, -1):
        if shape[axis] == 1:
            continue

        if strides[axis] != expected_stride:
            return False

        expected_stride *= shape[axis]

    return True


def unravel_index(
    flat: int,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Convert a flat C-order index into a multi-dimensional index.

    Internal helper used by advanced indexing and slicing logic.

    Parameters
    ----------
    flat : int
        Flat index into an array of ``shape``.
    shape : tuple of int
        Array dimensions.

    Returns
    -------
    tuple of int
        C-order coordinate tuple. Returns an empty tuple when ``shape``
        is empty.

    Examples
    --------
    >>> unravel_index(5, (2, 3))
    (1, 2)
    """
    if not shape:
        return ()

    coordinates: list[int] = []
    remaining = flat

    for dimension in reversed(shape):
        coordinates.append(remaining % dimension)
        remaining //= dimension

    return tuple(reversed(coordinates))


def slice_length(
    start: int,
    stop: int,
    step: int,
) -> int:
    """
    Compute the number of elements in a slice.

    Internal helper used when resolving slice objects during indexing.

    Parameters
    ----------
    start : int
        Slice start index.
    stop : int
        Slice stop index.
    step : int
        Slice step. Must not be zero.

    Returns
    -------
    int
        Number of indices produced by ``range(start, stop, step)``.

    Raises
    ------
    ValueError
        If ``step`` is zero.

    Notes
    -----
    Follows Python slice length semantics, including zero-length results
    when the slice direction does not move toward ``stop``.
    """
    if step == 0:
        raise slice_step_zero()

    if (stop - start) * step <= 0:
        return 0

    return (stop - start + step - (1 if step > 0 else -1)) // step


def normalize_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    """
    Validate and normalize a shape specification.

    Internal helper used by creation, broadcasting, and iteration
    functions that accept ``shape`` arguments.

    Parameters
    ----------
    shape : int or tuple of int
        Desired dimensions. A scalar integer is wrapped as a
        one-dimensional shape.

    Returns
    -------
    tuple of int
        Normalized shape with non-negative integer dimensions.

    Raises
    ------
    TypeError
        If ``shape`` is not an integer or tuple of integers.
    ValueError
        If any dimension is negative.

    Notes
    -----
    NumPy also accepts array-like shapes and may allow ``-1`` placeholders
    in some APIs; those conventions are not supported here.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if not isinstance(shape, tuple):
        raise shape_must_be_int_or_tuple()

    for dimension in shape:
        if not isinstance(dimension, int):
            raise shape_dimensions_must_be_integers()

        if dimension < 0:
            raise negative_dimension_not_allowed()

    return shape


def merge_index(
    ndim: int,
    reduced_axes: set[int],
    outer_index: tuple[int, ...],
    inner_index: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Combine outer and inner index tuples into a full ``ndim`` index.

    Internal helper used by reduction kernels that iterate over
    preserved and reduced axes separately.

    Parameters
    ----------
    ndim : int
        Rank of the source array.
    reduced_axes : set of int
        Axes that take coordinates from ``inner_index``.
    outer_index : tuple of int
        Indices for axes not in ``reduced_axes``, in ascending axis order.
    inner_index : tuple of int
        Indices for axes in ``reduced_axes``, in ascending axis order.

    Returns
    -------
    tuple of int
        Full source index of length ``ndim``.
    """
    merged: list[int] = []
    outer_pos = 0
    inner_pos = 0

    for axis in range(ndim):
        if axis in reduced_axes:
            merged.append(inner_index[inner_pos])
            inner_pos += 1
        else:
            merged.append(outer_index[outer_pos])
            outer_pos += 1

    return tuple(merged)


def keepdims_shape(
    shape: tuple[int, ...],
    reduced_axes: set[int],
) -> tuple[int, ...]:
    """
    Compute the output shape for a reduction with ``keepdims=True``.

    Internal helper used by reduction implementations.

    Parameters
    ----------
    shape : tuple of int
        Input array shape.
    reduced_axes : set of int
        Axes that were reduced.

    Returns
    -------
    tuple of int
        Shape where reduced axes have length ``1`` and other axes are
        unchanged.
    """
    return tuple(
        1 if axis in reduced_axes else size for axis, size in enumerate(shape)
    )
