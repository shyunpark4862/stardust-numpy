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

"""Core multi-dimensional array type for this package.

:class:`ndarray` stores data in a Python list with shape, strides, and dtype,
composing indexing, shape, element-wise, complex, reduction, linear algebra,
sorting, and iteration mixins into one concrete type.
"""

from __future__ import annotations

from typing import Any

from ._complex_ops import ComplexOpsMixin
from ._dtype import (
    DTypeLike,
    as_dtype,
    cast_sequence,
    cast_value,
    infer_dtype_from_values,
)
from ._elementwise_ops import ElementwiseOpsMixin
from ._errors import (
    cannot_broadcast_shape,
    cannot_broadcast_to_fewer_dimensions,
    shape_strides_length_mismatch,
)
from ._factory import contiguous_ndarray, view_ndarray
from ._format import array_repr, array_str
from ._indexing import IndexingMixin
from ._iter import IterMixin, iter_indices
from ._linalg import LinalgMixin
from ._reductions import ReductionMixin
from ._shape_manipulation import ShapeManipulationMixin
from ._sorting import SortingMixin


class ndarray(
    IndexingMixin,
    ShapeManipulationMixin,
    ElementwiseOpsMixin,
    ComplexOpsMixin,
    ReductionMixin,
    LinalgMixin,
    SortingMixin,
    IterMixin,
):
    """
    An n-dimensional array object.

    Arrays store elements in a flat Python list with shape, stride, and offset
    metadata. Most user-facing behavior is provided by mixins (indexing,
    element-wise operations, reductions, linear algebra, and so on).

    Attributes
    ----------
    buffer : list
        Flat storage for array elements.
    shape : tuple of int
        Length of each dimension.
    strides : tuple of int
        Number of buffer elements to step when advancing along each axis.
    offset : int (default: 0)
        Index into ``buffer`` of the logical origin of the array view.
    dtype : type
        Element type. One of ``bool``, ``int``, ``float``, ``complex``, or
        ``str``.
    ndim : int
        Number of dimensions.
    size : int
        Total number of elements.

    Notes
    -----
    Unlike NumPy's ``ndarray``, this implementation is pure Python, does not
    support fixed-width numeric dtypes (for example ``int32`` or ``float64``),
    object arrays, structured dtypes, or memory-mapped storage. Only the five
    Python scalar types ``bool``, ``int``, ``float``, ``complex``, and ``str``
    are supported.
    """

    def __init__(
        self,
        buffer: list[Any],
        shape: tuple[int, ...] | None = None,
        strides: tuple[int, ...] | None = None,
        offset: int = 0,
        *,
        dtype: type[Any] | None = None,
    ):
        """
        Construct an array from a flat buffer and layout metadata.

        Parameters
        ----------
        buffer : list
            Flat sequence holding array elements in logical order when the
            array is C-contiguous.
        shape : tuple of int, optional (default: ``(len(buffer),)``)
            Length of each dimension. Defaults to a one-dimensional shape
            whose length equals ``len(buffer)``.
        strides : tuple of int, optional (default: ``(1,)``)
            Stride in buffer elements along each axis. Must have the same
            length as ``shape``.
        offset : int (default: 0)
            Index into ``buffer`` of the element at index ``(0, ..., 0)``.
        dtype : type, optional (default: inferred from ``buffer``)
            Element type. When omitted and ``buffer`` is non-empty, the dtype
            is inferred from the buffer values; an empty buffer defaults to
            ``float``.

        Raises
        ------
        ValueError
            If ``shape`` and ``strides`` have different lengths.

        Notes
        -----
        End users should prefer :func:`numpy.creation.array` rather than
        calling this constructor directly.
        """
        if shape is None:
            shape = (len(buffer),)

        if strides is None:
            strides = (1,)

        if len(shape) != len(strides):
            raise shape_strides_length_mismatch(
                shape_len=len(shape),
                strides_len=len(strides),
            )

        if dtype is None:
            # Empty buffers default to float; otherwise infer from values.
            dtype = infer_dtype_from_values(buffer) if buffer else float

        self.buffer = buffer
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        from ._shape import size_of_shape

        return size_of_shape(self.shape)

    def astype(
        self,
        dtype: DTypeLike,
    ) -> ndarray:
        """
        Copy the array, casting its elements to a new dtype.

        Parameters
        ----------
        dtype : type or str
            Target element type or its string alias (``"bool"``, ``"int"``,
            ``"float"``, ``"complex"``, or ``"str"``).

        Returns
        -------
        ndarray
            A new C-contiguous array with the requested dtype.

        Notes
        -----
        When the target dtype matches the current dtype, this returns a copy
        rather than a view. NumPy may return ``self`` when no cast is
        required and ``copy=False``. Only the five supported Python scalar
        types can be used as dtypes.
        """
        target = as_dtype(dtype)

        if target is self.dtype:
            return self.copy()

        if self._is_c_contiguous():
            buffer = [
                cast_value(target, value) for value in self._flat_buffer_slice()
            ]
        else:
            # Non-contiguous layouts must be traversed in logical index order.
            buffer = [
                cast_value(target, value) for value in self._iter_values()
            ]

        return contiguous_ndarray(buffer, self.shape, dtype=target)

    def tolist(self) -> Any:
        """
        Return the array as a nested Python list.

        Returns
        -------
        list or scalar
            A nested list mirroring the array shape for ``ndim >= 1``, or a
            scalar Python value for a zero-dimensional array.

        Notes
        -----
        Elements are returned as plain Python scalars, not as array objects.
        """
        return self._tolist_recursive(0, ())

    def copy(self) -> ndarray:
        """
        Return a C-contiguous copy of the array.

        Returns
        -------
        ndarray
            A new array with the same shape, dtype, and element values.

        Notes
        -----
        Unlike NumPy, this always materializes a new buffer; there is no
        ``order`` keyword and views are never returned.
        """
        if self._is_c_contiguous():
            buffer = list(self._flat_buffer_slice())
            return contiguous_ndarray(
                buffer,
                self.shape,
                dtype=self.dtype,
                skip_cast=True,
            )

        # Gather elements in logical order when the source is a non-C view.
        buffer = [
            self._get_value(source_index)
            for source_index in iter_indices(self.shape)
        ]

        return contiguous_ndarray(buffer, self.shape, dtype=self.dtype)

    def __repr__(self) -> str:
        """Return a string representation suitable for debugging."""
        return array_repr(self)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return array_str(self)

    def broadcast_to(
        self,
        shape: tuple[int, ...],
    ) -> ndarray:
        """
        Broadcast the array to a new shape.

        Parameters
        ----------
        shape : tuple of int
            Target shape. Must be at least as many dimensions as the array and
            compatible with NumPy broadcasting rules.

        Returns
        -------
        ndarray
            A view of the same buffer with adjusted shape and strides.

        Raises
        ------
        ValueError
            If ``shape`` has fewer dimensions than the array or if the shapes
            are not broadcast-compatible.

        Notes
        -----
        This returns a view sharing ``buffer`` with the original array.
        NumPy's :func:`numpy.broadcast_to` behaves similarly, but this
        implementation does not support writing through broadcast dimensions
        with stride zero in all code paths.
        """
        if len(shape) < self.ndim:
            raise cannot_broadcast_to_fewer_dimensions(
                from_shape=self.shape,
                to_ndim=len(shape),
            )

        # Left-pad with length-1 dimensions so shapes align for broadcasting.
        padded_shape = (1,) * (len(shape) - self.ndim) + self.shape
        padded_strides = (0,) * (len(shape) - self.ndim) + self.strides
        new_strides: list[int] = []

        for from_dim, to_dim, stride in zip(
            padded_shape,
            shape,
            padded_strides,
        ):
            if from_dim == to_dim:
                new_strides.append(stride)
            elif from_dim == 1:
                # Broadcast a length-1 axis with zero stride.
                new_strides.append(0)
            else:
                raise cannot_broadcast_shape(
                    from_shape=self.shape,
                    to_shape=shape,
                )

        return self._view(
            shape=shape,
            strides=tuple(new_strides),
        )

    def _view(
        self,
        *,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> ndarray:
        """Return a view sharing ``buffer`` with updated layout metadata.

        Parameters
        ----------
        shape : tuple of int
            Shape of the returned view.
        strides : tuple of int
            Stride in buffer elements along each axis of the view. Must have
            the same length as ``shape``.

        Returns
        -------
        ndarray
            A new array object referencing the same ``buffer`` and
            ``offset`` as ``self``, with the requested ``shape`` and
            ``strides``. Element dtype is preserved.

        Notes
        -----
        Called by :meth:`broadcast_to`, :meth:`transpose`, and other shape
        manipulation methods that can represent the result without copying
        data. Unlike NumPy's ``ndarray.view``, there is no dtype reinterpret
        cast; only layout metadata changes. Mutations through the view are
        visible in the original array and vice versa.
        """
        return view_ndarray(
            self.buffer,
            shape,
            strides,
            offset=self.offset,
            dtype=self.dtype,
        )

    def _tolist_recursive(
        self,
        axis: int,
        prefix: tuple[int, ...],
    ) -> Any:
        """Build a nested Python list by recursing over array dimensions.

        Parameters
        ----------
        axis : int
            Current dimension being expanded. When ``axis == ndim``, the
            recursion base case is reached.
        prefix : tuple of int
            Partial index into ``self`` selecting the sub-array at the
            current recursion depth.

        Returns
        -------
        list or scalar
            A nested list for ``axis < ndim``, or a scalar element when
            ``axis == ndim`` (obtained via basic indexing on ``prefix``).

        Notes
        -----
        Called by :meth:`tolist`. Traversal follows C-order over shape
        indices. Each leaf value is a plain Python scalar, not an ``ndarray``.
        Zero-dimensional arrays are not supported by this implementation's
        shape model; ``tolist`` on a 1-D array returns a flat list, and on
        higher-dimensional arrays returns nested lists mirroring ``shape``.
        """
        if axis == self.ndim:
            return self[prefix]

        return [
            self._tolist_recursive(axis + 1, prefix + (i,))
            for i in range(self.shape[axis])
        ]
