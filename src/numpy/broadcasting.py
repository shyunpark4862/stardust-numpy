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

"""NumPy-style broadcasting rules for shape alignment.

Computes broadcast result shapes and materializes broadcast views via
:func:`broadcast_to` and :func:`broadcast_arrays`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import broadcast_operands_incompatible
from ._shape import normalize_shape
from ._validation import coerce_to_ndarray, require_shape_tuple

if TYPE_CHECKING:
    from .ndarray import ndarray


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """
    Determine the broadcast result shape for the given shapes.

    Parameters
    ----------
    *shapes : tuple of int
        One or more shape tuples. Each may be passed as an ``int``, which
        is normalized to a length-1 shape.

    Returns
    -------
    tuple of int
        Broadcast result shape.

    Raises
    ------
    ValueError
        If the shapes are not broadcast-compatible.

    Notes
    -----
    Follows NumPy broadcasting rules: dimensions are aligned from the
    trailing axis, length-1 dimensions are stretched, and missing leading
    dimensions are treated as size ``1``.

    Examples
    --------
    >>> broadcast_shape((2, 1), (3,))
    (2, 3)
    """
    if not shapes:
        return ()

    validated_shapes = tuple(normalize_shape(shape) for shape in shapes)

    result: list[int] = []

    max_ndim = max(len(shape) for shape in validated_shapes)

    for offset in range(1, max_ndim + 1):
        dimensions = []

        for shape in validated_shapes:
            if offset <= len(shape):
                dimensions.append(shape[-offset])
            else:
                dimensions.append(1)

        target_dim = max(dimensions)

        for dimension in dimensions:
            if dimension != 1 and dimension != target_dim:
                raise broadcast_operands_incompatible(shapes=validated_shapes)

        result.append(target_dim)

    return tuple(reversed(result))


def broadcast_to(
    a: ndarray,
    shape: tuple[int, ...],
) -> ndarray:
    """
    Broadcast an array to a new shape.

    Parameters
    ----------
    a : array_like
        Input array.
    shape : int or tuple of int
        Desired shape. The array must be broadcast-compatible with this
        shape.

    Returns
    -------
    ndarray
        A read-only view when no copy is required, otherwise a
        materialized broadcast array.

    Raises
    ------
    ValueError
        If ``a`` is not broadcast-compatible with ``shape``.

    Notes
    -----
    NumPy's :func:`numpy.broadcast_to` returns a read-only view and
    supports an optional memory-layout policy. This implementation
    delegates to :meth:`ndarray.broadcast_to` and may copy when stride
    manipulation cannot express the broadcast.

    Examples
    --------
    >>> x = array([1, 2, 3])
    >>> broadcast_to(x, (2, 3))
    array([[1, 2, 3],
           [1, 2, 3]])
    """
    a = coerce_to_ndarray(a)
    shape = require_shape_tuple(shape)

    return a.broadcast_to(shape)


def broadcast_arrays(*arrays: ndarray) -> tuple[ndarray, ...]:
    """
    Broadcast any number of arrays against one another.

    Parameters
    ----------
    *arrays : array_like
        Arrays to broadcast.

    Returns
    -------
    tuple of ndarray
        Broadcast views or copies of the inputs, all sharing the same
        shape. Returns an empty tuple when called with no arguments.

    Raises
    ------
    ValueError
        If the operands are not mutually broadcast-compatible.

    Notes
    -----
    Unlike :func:`numpy.broadcast_arrays`, this function does not accept
    a ``subok`` keyword and always coerces inputs through
    :func:`coerce_to_ndarray`.

    Examples
    --------
    >>> a = array([1, 2, 3])
    >>> b = array([[1], [2], [3]])
    >>> x, y = broadcast_arrays(a, b)
    >>> x.shape, y.shape
    ((3, 3), (3, 3))
    """
    if len(arrays) == 0:
        return ()

    validated_arrays = tuple(coerce_to_ndarray(array) for array in arrays)
    result_shape = broadcast_shape(*(array.shape for array in validated_arrays))

    return tuple(array.broadcast_to(result_shape) for array in validated_arrays)
