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

"""Construct contiguous arrays and zero-copy views from buffers.

Internal factories used by creation, indexing, reductions, and ufuncs to
build :class:`ndarray` instances with consistent C-order strides.
"""

from __future__ import annotations

from typing import Any

from ._dtype import (
    DTypeLike,
    as_dtype,
    cast_sequence,
    default_fill_value,
    infer_dtype_from_values,
)
from ._shape import c_order_strides, size_of_shape


def contiguous_ndarray(
    buffer: list[Any],
    shape: tuple[int, ...],
    *,
    dtype: type[Any] | None = None,
    skip_cast: bool = False,
) -> Any:
    """
    Construct a C-contiguous :class:`ndarray` from a flat buffer.

    Internal factory used by creation, manipulation, reductions, linear
    algebra, indexing, and elementwise routines.

    Parameters
    ----------
    buffer : list
        Flat sequence of elements in C-order. Its length must equal the
        product of ``shape``.
    shape : tuple of int
        Array dimensions.
    dtype : type or None (default: None)
        Element dtype. When ``None``, inferred from ``buffer`` if
        non-empty, otherwise ``float``.
    skip_cast : bool (default: False)
        If ``True``, use ``buffer`` as-is without casting elements to
        ``dtype``.

    Returns
    -------
    ndarray
        New array sharing no view relationship with caller-owned buffers
        beyond the provided list reference.

    Notes
    -----
    NumPy may allocate from uninitialized memory or reuse existing
    storage; this implementation always builds from a Python ``list`` and
    computes C-order strides explicitly.
    """
    from .ndarray import ndarray

    if dtype is None:
        dtype = infer_dtype_from_values(buffer) if buffer else float

    if skip_cast:
        cast_buffer = buffer
    else:
        cast_buffer = cast_sequence(buffer, dtype)

    return ndarray(
        buffer=cast_buffer,
        shape=shape,
        strides=c_order_strides(shape),
        dtype=dtype,
    )


def view_ndarray(
    buffer: list[Any],
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int = 0,
    *,
    dtype: type[Any],
) -> Any:
    """
    Construct an :class:`ndarray` view over existing storage.

    Internal factory used by ndarray methods, indexing, and
    :func:`numpy.manipulation.expand_dims`.

    Parameters
    ----------
    buffer : list
        Shared backing storage.
    shape : tuple of int
        View shape.
    strides : tuple of int
        Per-axis strides into ``buffer``.
    offset : int (default: 0)
        Starting index into ``buffer``.
    dtype : type
        Element dtype of the view.

    Returns
    -------
    ndarray
        Array that reads and writes through the supplied buffer without
        copying data.

    Notes
    -----
    Unlike NumPy views, mutating a view here mutates the underlying
    Python list directly; there is no separate memory owner object.
    """
    from .ndarray import ndarray

    return ndarray(
        buffer=buffer,
        shape=shape,
        strides=strides,
        offset=offset,
        dtype=dtype,
    )


def placeholder_contiguous_ndarray(
    shape: tuple[int, ...],
    *,
    dtype: type[Any] = float,
) -> Any:
    """
    Allocate a C-contiguous output array filled with dtype defaults.

    Internal helper used by elementwise ufunc-style operations when an
    ``out`` argument is not supplied.

    Parameters
    ----------
    shape : tuple of int
        Output shape.
    dtype : type (default: float)
        Element dtype used for both fill values and the returned array.

    Returns
    -------
    ndarray
        New array whose elements are initialized to the default fill
        value for ``dtype``.

    Notes
    -----
    NumPy's ``empty`` leaves memory uninitialized; this helper always
    pre-fills elements so in-place writes start from a defined value.
    """
    return contiguous_ndarray(
        [default_fill_value(dtype)] * size_of_shape(shape),
        shape,
        dtype=dtype,
    )
