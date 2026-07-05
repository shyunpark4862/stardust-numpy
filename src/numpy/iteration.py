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

"""Public helpers for iterating array indices and values.

Wraps internal iterators with :func:`ndindex`, :func:`ndenumerate`, and
:func:`nditer` for C-order coordinate and operand traversal.
"""

from __future__ import annotations

from typing import Any, Iterator

from ._errors import (
    nditer_invalid_operands,
    nditer_requires_operand,
    nditer_unsupported_op_flags,
)
from ._iter import NDIter, iter_indices
from ._shape import normalize_shape
from ._types import is_ndarray
from ._validation import coerce_to_ndarray


def ndindex(*shape: int | tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """
    Convert a shape into an iterator of C-order coordinate tuples.

    Parameters
    ----------
    *shape : int or tuple of int
        Dimensions of the index grid. A single tuple argument is
        accepted, so both ``ndindex(2, 3)`` and ``ndindex((2, 3))`` work.

    Returns
    -------
    iterator of tuple of int
        Coordinate tuples covering every position in the shape, in
        C-order.

    Notes
    -----
    Unlike :func:`numpy.ndindex`, this helper does not accept a shape
    tuple passed as a single non-tuple argument other than the
    ``ndindex((m, n))`` form.

    Examples
    --------
    >>> list(ndindex(2, 2))
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    if len(shape) == 1 and not isinstance(shape[0], int):
        normalized = normalize_shape(shape[0])
    else:
        normalized = normalize_shape(shape)

    return iter_indices(normalized)


def ndenumerate(a: Any) -> Iterator[tuple[tuple[int, ...], Any]]:
    """
    Enumerate an array with C-order multi-dimensional indices.

    Parameters
    ----------
    a : array_like
        Input array.

    Yields
    ------
    index : tuple of int
        C-order coordinate tuple.
    value : scalar
        Element at ``index``.

    Notes
    -----
    NumPy's :func:`numpy.ndenumerate` also accepts a ``flat`` mode for
    one-dimensional iteration; only multi-dimensional C-order enumeration
    is implemented here.

    Examples
    --------
    >>> a = array([[1, 2], [3, 4]])
    >>> list(ndenumerate(a))
    [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]
    """
    array = coerce_to_ndarray(a)

    for index in iter_indices(array.shape):
        yield index, array[index]


def nditer(
    operands: Any,
    *,
    op_flags: str = "readonly",
) -> NDIter:
    """
    Create an iterator for one or more broadcast-compatible operands.

    Parameters
    ----------
    operands : ndarray or sequence of array_like
        A single array or a non-empty sequence of arrays to iterate in
        lockstep after broadcasting.
    op_flags : str (default: "readonly")
        Operand access mode. Only ``"readonly"`` is supported.

    Returns
    -------
    NDIter
        Iterator yielding tuples of scalar elements, one per operand,
        in C-order over the broadcast shape.

    Raises
    ------
    ValueError
        If ``operands`` is empty, invalid, or not broadcast-compatible.
    NotImplementedError
        If ``op_flags`` is not ``"readonly"``.

    Notes
    -----
    NumPy's :class:`numpy.nditer` exposes extensive configuration for
    buffering, casting, and write-back. This implementation provides a
    minimal read-only iterator over broadcast operands.

    Examples
    --------
    >>> a = array([1, 2, 3])
    >>> b = array([10])
    >>> for x, y in nditer([a, b]):
    ...     print(x, y)
    1 10
    2 10
    3 10
    """
    if op_flags != "readonly":
        raise nditer_unsupported_op_flags(op_flags=op_flags)

    if is_ndarray(operands):
        ops = (operands,)
    elif isinstance(operands, (list, tuple)):
        if len(operands) == 0:
            raise nditer_requires_operand()

        ops = tuple(coerce_to_ndarray(operand) for operand in operands)
    else:
        raise nditer_invalid_operands(got=operands)

    return NDIter(ops)
