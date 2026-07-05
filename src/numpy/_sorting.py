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

"""In-place sorting and argsort for ``ndarray``.

:class:`SortingMixin` sorts along an axis or in flat order while preserving
array shape, with stable NaN-last ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._axis import normalize_axis
from ._dtype import require_orderable_dtype
from ._factory import contiguous_ndarray
from ._float_ops import order_key
from ._iter import iter_indices
from ._shape import flat_index

if TYPE_CHECKING:
    from .ndarray import ndarray


class SortingMixin:
    """Mixin providing in-place sorting and argsort for ``ndarray``."""

    def sort(
        self,
        axis: int | None = -1,
    ) -> None:
        """Sort an array in place.

        Parameters
        ----------
        axis : int or None (default: -1)
            Axis along which to sort. When ``None``, the array is flattened,
            sorted, and written back in C order while preserving the original
            shape.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the array dtype is not orderable (e.g. ``complex``).

        Notes
        -----
        Unlike :func:`sort`, this method mutates the array. When
        ``axis is None``, values are sorted in flat (C) order but the shape
        is unchanged. Sorting is stable with NaNs last. Only ``bool``,
        ``int``, ``float``, and ``str`` dtypes are orderable.

        See Also
        --------
        sort : Return a sorted copy.
        argsort : Return indices that would sort this array.
        """
        require_orderable_dtype(self.dtype, operation="sort")

        if axis is None:
            # Flatten, sort, then scatter values back in C order.
            values = list(self._iter_values())
            values.sort(key=order_key)
            self._write_flat_values(values)
            return None

        axis = normalize_axis(axis, self.ndim)
        outer_shape = self.shape[:axis] + self.shape[axis + 1 :]

        for outer_index in iter_indices(outer_shape):
            values = []

            for axis_index in range(self.shape[axis]):
                source_index = (
                    outer_index[:axis] + (axis_index,) + outer_index[axis:]
                )
                values.append(self._get_value(source_index))

            values.sort(key=order_key)

            for axis_index, value in enumerate(values):
                source_index = (
                    outer_index[:axis] + (axis_index,) + outer_index[axis:]
                )
                self._set_value(source_index, value)

    def argsort(
        self,
        axis: int | None = -1,
    ) -> ndarray:
        """Return the indices that would sort this array.

        Parameters
        ----------
        axis : int or None (default: -1)
            Axis along which to sort. When ``None``, indices refer to the
            flattened (C-order) array and the result is one-dimensional.

        Returns
        -------
        ndarray
            Integer index array of the same shape as the input (or
            one-dimensional when ``axis is None``).

        Raises
        ------
        TypeError
            If the array dtype is not orderable.

        Notes
        -----
        Sorting is stable: equal elements retain their relative order via a
        secondary key on the original index. NaN ordering matches
        :meth:`sort`. There is no ``kind`` or ``order`` keyword.

        See Also
        --------
        sort : Sort this array in place.
        """
        require_orderable_dtype(self.dtype, operation="argsort")

        if axis is None:
            # Sort (value, flat_index) pairs; return the permuted flat indices.
            pairs = [
                (value, flat_index)
                for flat_index, value in enumerate(self._iter_values())
            ]
            pairs.sort(key=lambda pair: (order_key(pair[0]), pair[1]))

            return contiguous_ndarray(
                [original_flat_index for _, original_flat_index in pairs],
                (self.size,),
                dtype=int,
            )

        axis = normalize_axis(axis, self.ndim)
        outer_shape = self.shape[:axis] + self.shape[axis + 1 :]
        result_buffer = [0] * self.size

        for outer_index in iter_indices(outer_shape):
            pairs = []

            for axis_index in range(self.shape[axis]):
                source_index = (
                    outer_index[:axis] + (axis_index,) + outer_index[axis:]
                )
                pairs.append((self._get_value(source_index), axis_index))

            pairs.sort(key=lambda pair: (order_key(pair[0]), pair[1]))

            for sorted_axis_index, (_, original_axis_index) in enumerate(pairs):
                # Map sorted position back to the original axis index.
                result_index = (
                    outer_index[:axis]
                    + (sorted_axis_index,)
                    + outer_index[axis:]
                )

                result_buffer[flat_index(result_index, self.shape)] = (
                    original_axis_index
                )

        return contiguous_ndarray(result_buffer, self.shape, dtype=int)
