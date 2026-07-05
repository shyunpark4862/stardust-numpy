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

"""C-order iteration over array shapes and elements.

Provides :func:`iter_indices`, :class:`FlatIterator`, :class:`NDIter`, and
:class:`IterMixin` used by reductions, ufuncs, formatting, and the public
iteration module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from ._errors import iteration_over_zero_d_array, len_of_unsized

if TYPE_CHECKING:
    from .ndarray import ndarray


def iter_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Generate index tuples for an array shape in C order.

    Parameters
    ----------
    shape : tuple of int
        Array shape.

    Yields
    ------
    index : tuple of int
        Next multi-dimensional index in row-major (C) order.

    Notes
    -----
    Unlike NumPy's ``np.ndindex``, this is a module-level generator
    function rather than a class. Shapes containing a zero dimension
    yield no indices. A scalar shape ``()`` yields a single empty
    tuple ``()``.
    """
    if any(dimension == 0 for dimension in shape):
        return

    if len(shape) == 0:
        yield ()
        return

    indices = [0] * len(shape)

    while True:
        yield tuple(indices)

        for axis in range(len(shape) - 1, -1, -1):
            indices[axis] += 1

            if indices[axis] < shape[axis]:
                break

            indices[axis] = 0
        else:
            return


class FlatIterator:
    """Flat iterator object to access the array as a 1-D sequence.

    Parameters
    ----------
    array : ndarray
        Array to iterate over.

    Notes
    -----
    Unlike NumPy's ``flat`` iterator, assignment through the iterator is
    not supported. C-contiguous arrays read directly from the underlying
    buffer for efficiency.
    """

    def __init__(
        self,
        array: ndarray,
    ) -> None:
        """Store the array to iterate; iteration logic lives in ``__iter__``.

        Parameters
        ----------
        array : ndarray
            Array whose elements will be yielded in C order.
        """
        self._array = array

    def __iter__(self) -> Iterator[Any]:
        """Iterate over array elements in C order.

        Yields
        ------
        element
            Next scalar element.

        Notes
        -----
        C-contiguous arrays yield values directly from the flat buffer;
        non-contiguous arrays fall back to indexed access.
        """
        array = self._array

        if array._is_c_contiguous():
            buffer = array.buffer
            start = array.offset
            end = start + array.size

            for position in range(start, end):
                yield buffer[position]
            return

        for source_index in iter_indices(array.shape):
            yield array._get_value(source_index)


class NDIter:
    """N-dimensional iterator over broadcast-compatible operands.

    Parameters
    ----------
    operands : tuple of ndarray
        Input arrays to iterate over in lockstep after broadcasting.

    Notes
    -----
    Unlike NumPy's ``nditer``, only basic C-order iteration over
    already-broadcast operands is supported. There are no buffering,
    casting, ``op_flags``, or multi-array write-back options.
    """

    def __init__(
        self,
        operands: tuple[ndarray, ...],
    ) -> None:
        """Broadcast operands to a common shape for lockstep iteration.

        Parameters
        ----------
        operands : tuple of ndarray
            Input arrays; must be mutually broadcast-compatible.

        Notes
        -----
        Computes ``broadcast_shape`` from operand shapes, then replaces
        each operand with ``broadcast_to`` that shape. Stored as
        ``self._broadcasted``; ``__iter__`` walks ``self._shape`` in C
        order and yields scalar(s) from each broadcast view.
        """
        from .broadcasting import broadcast_shape

        self._operands = operands
        self._shape = broadcast_shape(*(operand.shape for operand in operands))
        self._broadcasted = tuple(
            operand.broadcast_to(self._shape) for operand in operands
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate over broadcast operand tuples in C order.

        Yields
        ------
        values : scalar or tuple of scalars
            Next element from each operand. A single operand yields a
            scalar; multiple operands yield a tuple.

        Notes
        -----
        Contiguous broadcast operands use direct indexed access; otherwise
        subscripting is used per element.
        """
        if all(operand._is_c_contiguous() for operand in self._broadcasted):
            for source_index in iter_indices(self._shape):
                values = tuple(
                    operand._get_value(source_index)
                    for operand in self._broadcasted
                )

                if len(values) == 1:
                    yield values[0]
                else:
                    yield values
            return

        for index in iter_indices(self._shape):
            values = tuple(operand[index] for operand in self._broadcasted)

            if len(values) == 1:
                yield values[0]
            else:
                yield values


class IterMixin:
    """Mixin providing sequence iteration over the leading axis."""

    def __iter__(self) -> Iterator[Any]:
        """Implement iter(self).

        Returns
        -------
        iterator
            Iterator yielding sub-arrays along axis 0.

        Raises
        ------
        TypeError
            If the array is 0-dimensional.

        Notes
        -----
        Unlike NumPy, 0-dimensional arrays cannot be iterated. Each
        yielded item is a view along the first axis.
        """
        if self.ndim == 0:
            raise iteration_over_zero_d_array()

        for index in range(self.shape[0]):
            yield self[index]

    def __len__(self) -> int:
        """Return len(self), the size of the first dimension.

        Returns
        -------
        int
            Length of axis 0.

        Raises
        ------
        TypeError
            If the array is 0-dimensional.

        Notes
        -----
        Unlike NumPy, 0-dimensional arrays do not support ``len()``.
        """
        if self.ndim == 0:
            raise len_of_unsized()

        return self.shape[0]

    @property
    def flat(self) -> FlatIterator:
        """A 1-D iterator over the array.

        Returns
        -------
        FlatIterator
            Iterator yielding scalar elements in C order.

        Notes
        -----
        Unlike NumPy's ``flat``, the returned iterator does not support
        assignment.
        """
        return FlatIterator(self)
