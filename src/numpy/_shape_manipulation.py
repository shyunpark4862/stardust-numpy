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

"""Reshape, transpose, and dimension manipulation for ``ndarray``.

:class:`ShapeManipulationMixin` provides view-based shape ops when the
buffer is C-contiguous and copy-based fallbacks otherwise.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

from ._axis import normalize_axes, normalize_axis
from ._errors import (
    cannot_squeeze_axis,
    negative_dimension_not_allowed,
    reshape_incompatible,
    reshape_only_one_unknown,
    reshape_unknown_dimension_invalid,
    shape_dimensions_must_be_integers,
    transpose_axes_mismatch,
)
from ._shape import c_order_strides, is_c_contiguous

if TYPE_CHECKING:
    from .ndarray import ndarray


class ShapeManipulationMixin:
    """Mixin providing reshape, transpose, and related shape operations."""

    def transpose(
        self,
        *axes: int,
    ) -> ndarray:
        """Returns a view of the array with axes permuted.

        Parameters
        ----------
        *axes : int, optional
            Axis indices in the desired order. If omitted, the axes are
            reversed.

        Returns
        -------
        ndarray
            View with permuted ``shape`` and ``strides``.

        Raises
        ------
        ValueError
            If ``axes`` is not a full permutation of ``0 .. ndim-1``.

        Notes
        -----
        Unlike NumPy, ``axes`` must be passed positionally; there is no
        ``axes`` keyword argument. Returns a view sharing the underlying
        buffer.
        """
        if not axes:
            normalized_axes = tuple(reversed(range(self.ndim)))
        else:
            normalized_axes = normalize_axes(axes, self.ndim)

            if len(normalized_axes) != self.ndim:
                raise transpose_axes_mismatch(
                    ndim=self.ndim, axes=normalized_axes
                )

        return self._view(
            shape=tuple(self.shape[axis] for axis in normalized_axes),
            strides=tuple(self.strides[axis] for axis in normalized_axes),
        )

    @property
    def T(self) -> ndarray:
        """View of the array with axes reversed.

        Returns
        -------
        ndarray
            Transposed view. For 1-D arrays, returns ``self`` unchanged.

        Notes
        -----
        Equivalent to ``self.transpose()`` with no arguments. Unlike
        NumPy, 1-D arrays are not promoted to row vectors.
        """
        return self.transpose()

    def ravel(self) -> ndarray:
        """Return a contiguous flattened array.

        Returns
        -------
        ndarray
            1-D view if C-contiguous; otherwise a flattened copy.

        Notes
        -----
        Unlike NumPy's ``ravel('K')``, non-contiguous arrays are copied
        before flattening. There is no ``order`` argument.
        """
        if self._is_c_contiguous():
            return self._view(
                shape=(self.size,),
                strides=(1,),
            )

        return self.copy().ravel()

    def flatten(self) -> ndarray:
        """Return a copy of the array collapsed into one dimension.

        Returns
        -------
        ndarray
            1-D copy of the array data in C order.

        Notes
        -----
        Unlike ``ravel``, this always returns a copy. There is no
        ``order`` argument.
        """
        return self.copy().ravel()

    def reshape(
        self,
        *shape: int,
    ) -> ndarray:
        """Give a new shape to an array without changing its data.

        Parameters
        ----------
        *shape : int
            The new shape. One dimension may be ``-1`` to infer its size
            from the array size and remaining dimensions.

        Returns
        -------
        ndarray
            View if C-contiguous; otherwise a reshaped copy.

        Raises
        ------
        ValueError
            If the total size does not match or more than one ``-1`` is
            given.
        TypeError
            If shape dimensions are not integers.

        Notes
        -----
        Unlike NumPy, ``order`` and ``newshape`` keyword arguments are
        not supported; shape must be passed positionally. A tuple may be
        passed as a single argument (``reshape((2, 3))``).
        """
        new_shape = self._normalize_new_shape(shape)

        if self._is_c_contiguous():
            return self._view(
                shape=new_shape,
                strides=c_order_strides(new_shape),
            )

        return self.copy().reshape(new_shape)

    def squeeze(
        self,
        axis: int | None = None,
    ) -> ndarray:
        """Remove single-dimensional entries from the shape.

        Parameters
        ----------
        axis : int, optional
            Axis to squeeze. If omitted, all length-1 axes are removed.

        Returns
        -------
        ndarray
            View with the requested axes removed.

        Raises
        ------
        ValueError
            If ``axis`` is specified but its length is not 1.

        Notes
        -----
        Unlike NumPy, only integer axes are accepted; tuple axes are not
        supported. Returns a view, not a copy.
        """
        if axis is None:
            axes_to_remove = {
                axis for axis, size in enumerate(self.shape) if size == 1
            }
        else:
            axis = normalize_axis(axis, self.ndim)

            if self.shape[axis] != 1:
                raise cannot_squeeze_axis(axis=axis, size=self.shape[axis])

            axes_to_remove = {axis}

        return self._view(
            shape=tuple(
                size
                for axis, size in enumerate(self.shape)
                if axis not in axes_to_remove
            ),
            strides=tuple(
                stride
                for axis, stride in enumerate(self.strides)
                if axis not in axes_to_remove
            ),
        )

    def swapaxes(
        self,
        axis1: int,
        axis2: int,
    ) -> ndarray:
        """Interchange two axes of an array.

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.

        Returns
        -------
        ndarray
            View with ``axis1`` and ``axis2`` swapped.

        Notes
        -----
        Negative axis indices are normalized. Returns a view, not a copy.
        """
        axis1 = normalize_axis(axis1, self.ndim)
        axis2 = normalize_axis(axis2, self.ndim)

        axes = list(range(self.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

        return self.transpose(*axes)

    def _is_c_contiguous(self) -> bool:
        """Return whether the array buffer is laid out in C order.

        Called by :meth:`reshape`, :meth:`ravel`, and indexing helpers to
        choose between view-based and copy-based paths.
        """
        return is_c_contiguous(self.shape, self.strides)

    def _normalize_new_shape(
        self,
        shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Validate and resolve a requested reshape target shape.

        Called by :meth:`reshape` before building a view or copy. Accepts a
        single ``-1`` placeholder dimension, which is inferred from
        ``self.size`` and the product of the other dimensions.

        Parameters
        ----------
        shape : tuple of int
            Requested shape. May be passed as ``(n,)`` or ``((n, m),)`` when
            the caller used ``reshape((n, m))`` with one tuple argument.

        Returns
        -------
        tuple of int
            Fully resolved shape with no ``-1`` entries.

        Raises
        ------
        ValueError
            If more than one ``-1`` is present, if the inferred dimension
            is not an integer, if any dimension is negative, or if the
            product does not equal ``self.size``.

        Notes
        -----
        Unlike NumPy, ``order`` and ``copy`` keywords are not supported;
        non-contiguous arrays always copy during reshape.
        """
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        new_shape = list(shape)
        minus_one_count = new_shape.count(-1)

        if minus_one_count > 1:
            raise reshape_only_one_unknown()

        known_size = 1

        for dimension in new_shape:
            if dimension == -1:
                continue

            if not isinstance(dimension, int):
                raise shape_dimensions_must_be_integers()

            if dimension < 0:
                raise negative_dimension_not_allowed()

            known_size *= dimension

        if minus_one_count == 1:
            if known_size == 0 or self.size % known_size != 0:
                raise reshape_unknown_dimension_invalid(
                    size=self.size,
                    known_size=known_size,
                )

            unknown_axis = new_shape.index(-1)
            new_shape[unknown_axis] = self.size // known_size

        if self.size != reduce(mul, new_shape, 1):
            raise reshape_incompatible(size=self.size, shape=tuple(new_shape))

        return tuple(new_shape)
