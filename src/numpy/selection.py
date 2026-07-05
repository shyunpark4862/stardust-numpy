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

"""Conditional selection, indexing, and clipping of array values.

Implements :func:`where`, :func:`nonzero`, :func:`take`, and :func:`clip`
with broadcasting and dtype promotion consistent with ufuncs.
"""

from __future__ import annotations

from typing import Any

from ._axis import normalize_axis
from ._dtype import (
    cast_value,
    infer_dtype_from_value,
    is_bool_dtype,
    is_int_dtype,
    promote_dtypes,
    require_numeric_dtype,
)
from ._errors import (
    take_index_out_of_range,
    take_indices_must_be_int,
    where_condition_must_be_bool,
)
from ._factory import contiguous_ndarray
from ._iter import iter_indices
from ._types import is_ndarray
from ._validation import coerce_to_ndarray, is_basic_array_like
from .broadcasting import broadcast_shape
from .creation import array
from .ndarray import ndarray


def where(
    condition: Any,
    x: Any,
    y: Any,
) -> ndarray:
    """Return elements chosen from `x` or `y` depending on `condition`.

    For each position, the output takes the value from `x` when the
    corresponding element of `condition` is True and from `y` otherwise.
    Operands are broadcast to a common shape before selection.

    Parameters
    ----------
    condition : array_like
        Boolean array. Non-boolean dtypes raise ``TypeError``.
    x : array_like or scalar
        Values selected where `condition` is True.
    y : array_like or scalar
        Values selected where `condition` is False.

    Returns
    -------
    ndarray
        Array with elements from `x` and `y` interleaved according to
        `condition`. The result dtype is determined by promoting the dtypes
        of `x` and `y`.

    Raises
    ------
    TypeError
        If `condition` is not a boolean array.
    ValueError
        If `condition`, `x`, and `y` cannot be broadcast together.

    Notes
    -----
    Unlike NumPy, only ``bool``, ``int``, ``float``, ``complex``, and ``str``
    element types are supported (no object arrays). There is no ``out``
    parameter. Array-like operands are converted to ``ndarray``; scalar
    operands are broadcast without materializing a full array.

    See Also
    --------
    nonzero : Return the indices of elements that evaluate to True.
    """
    condition = coerce_to_ndarray(condition, name="condition")

    if not is_bool_dtype(condition.dtype):
        raise where_condition_must_be_bool(got=condition.dtype)

    x_operand = _coerce_operand(x)
    y_operand = _coerce_operand(y)

    broadcast_shapes = [condition.shape]

    if is_ndarray(x_operand):
        broadcast_shapes.append(x_operand.shape)

    if is_ndarray(y_operand):
        broadcast_shapes.append(y_operand.shape)

    result_shape = (
        condition.shape
        if len(broadcast_shapes) == 1
        else broadcast_shape(*broadcast_shapes)
    )

    condition = condition.broadcast_to(result_shape)

    if is_ndarray(x_operand):
        x_dtype = x_operand.dtype
        x_operand = x_operand.broadcast_to(result_shape)
    else:
        x_dtype = infer_dtype_from_value(x_operand)

    if is_ndarray(y_operand):
        y_dtype = y_operand.dtype
        y_operand = y_operand.broadcast_to(result_shape)
    else:
        y_dtype = infer_dtype_from_value(y_operand)

    result_dtype = promote_dtypes(x_dtype, y_dtype)
    result_buffer = []

    for source_index in iter_indices(result_shape):
        # Branch per element: ndarray operands are indexed; scalars are reused.
        if condition._get_value(source_index):
            value = (
                x_operand._get_value(source_index)
                if is_ndarray(x_operand)
                else x_operand
            )
        else:
            value = (
                y_operand._get_value(source_index)
                if is_ndarray(y_operand)
                else y_operand
            )

        result_buffer.append(cast_value(result_dtype, value))

    return contiguous_ndarray(
        result_buffer,
        result_shape,
        dtype=result_dtype,
    )


def nonzero(a: Any) -> tuple[ndarray, ...]:
    """Return the indices of the elements that are non-zero (or True).

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple of ndarray
        ``a.ndim`` one-dimensional integer arrays, each holding the
        coordinates along one axis for every position where ``a`` evaluates
        to True in a boolean context.

    Notes
    -----
    Truthiness follows Python semantics (``0``, ``0.0``, and ``False`` are
    false; empty strings are false). Unlike NumPy,     index arrays always use the ``int`` dtype. There is no ``as_tuple`` keyword.

    See Also
    --------
    where : Select values from `x` and `y` according to a condition.
    """
    a = coerce_to_ndarray(a)
    coordinates = [[] for _ in range(a.ndim)]

    for source_index in iter_indices(a.shape):
        if a[source_index]:
            for axis, coordinate in enumerate(source_index):
                coordinates[axis].append(coordinate)

    return tuple(
        array(axis_coordinates, dtype=int) for axis_coordinates in coordinates
    )


def take(
    a: Any,
    indices: Any,
    axis: int | None = None,
) -> ndarray:
    """Take elements from an array along an axis.

    Parameters
    ----------
    a : array_like
        Source array.
    indices : array_like of int
        Indices of the values to extract. Negative indices count from the
        end of the axis.
    axis : int or None (default: None)
        Axis along which to select values. When ``None``, `a` is treated as
        if it had been flattened (C order) before indexing.

    Returns
    -------
    ndarray
        The indexed result. The output shape is ``indices.shape`` when
        ``axis is None``, otherwise ``a.shape[:axis] + indices.shape +
        a.shape[axis + 1:]``.

    Raises
    ------
    TypeError
        If `indices` is not an integer array.
    IndexError
        If any index is out of bounds after negative-index normalization.

    Notes
    -----
    Unlike NumPy, there is no ``mode`` argument (``'raise'`` only), no
    ``out`` parameter, and ``indices`` must be an ``ndarray`` with ``int``
    dtype after coercion. Only the five supported element dtypes are allowed.

    See Also
    --------
    ndarray.take : Equivalent method on ``ndarray``.
    """
    a = coerce_to_ndarray(a)
    indices = coerce_to_ndarray(indices, name="indices")

    if not is_int_dtype(indices.dtype):
        raise take_indices_must_be_int(got=indices.dtype)

    if axis is None:
        # Flatten `a` first; output shape matches `indices` exactly.
        source = a.ravel()
        result_buffer = [
            source[
                _normalize_take_index(
                    indices[source_index],
                    source.size,
                )
            ]
            for source_index in iter_indices(indices.shape)
        ]

        return contiguous_ndarray(
            result_buffer,
            indices.shape,
            dtype=a.dtype,
        )

    axis = normalize_axis(axis, a.ndim)
    result_shape = a.shape[:axis] + indices.shape + a.shape[axis + 1 :]
    result_buffer = []

    for result_index in iter_indices(result_shape):
        indices_index = result_index[axis : axis + indices.ndim]
        take_index = _normalize_take_index(
            indices[indices_index],
            a.shape[axis],
        )

        source_index = (
            result_index[:axis]
            + (take_index,)
            + result_index[axis + indices.ndim :]
        )
        result_buffer.append(a[source_index])

    return contiguous_ndarray(
        result_buffer,
        result_shape,
        dtype=a.dtype,
    )


def clip(
    a: Any,
    min_value: Any | None = None,
    max_value: Any | None = None,
) -> ndarray:
    """Clip (limit) the values in an array.

    Values smaller than `min_value` are set to `min_value`, and values
    larger than `max_value` are set to `max_value`.

    Parameters
    ----------
    a : array_like
        Array containing the elements to clip.
    min_value : scalar or None (default: None)
        Lower bound. When ``None``, no lower clipping is applied.
    max_value : scalar or None (default: None)
        Upper bound. When ``None``, no upper clipping is applied.

    Returns
    -------
    ndarray
        A new array with clipped values. When both bounds are ``None``, a
        copy of `a` is returned.

    Raises
    ------
    TypeError
        If `a` does not have a numeric dtype.

    Notes
    -----
    Parameter names are ``min_value`` and ``max_value`` (not NumPy's
    ``a_min`` / ``a_max``). Bounds must be scalars; array bounds are not
    broadcast. There is no ``out`` parameter. Only numeric dtypes
    (``bool``, ``int``, ``float``, ``complex``) are supported.
    """
    a = coerce_to_ndarray(a)
    require_numeric_dtype(a.dtype, operation="clip")

    if min_value is None and max_value is None:
        # No bounds supplied: preserve values but always return a new array.
        return a.copy()

    result_buffer = []

    for source_index in iter_indices(a.shape):
        value = a[source_index]

        if min_value is not None and value < min_value:
            value = min_value

        if max_value is not None and value > max_value:
            value = max_value

        result_buffer.append(value)

    return contiguous_ndarray(result_buffer, a.shape, dtype=a.dtype)


def _coerce_operand(value: Any) -> ndarray | Any:
    """Coerce array-like operands for :func:`where`; leave scalars unchanged.

    Parameters
    ----------
    value : array_like or scalar
        An ``x`` or ``y`` operand passed to :func:`where`.

    Returns
    -------
    ndarray or scalar
        An ``ndarray`` when ``value`` is array-like; otherwise the
        original scalar.

    Notes
    -----
    Unlike NumPy, only ``list``, ``tuple``, and ``range`` are accepted as
    array-like; existing ``ndarray`` instances pass through unchanged.
    """
    if is_ndarray(value):
        return value

    if is_basic_array_like(value):
        return array(value)

    return value


def _normalize_take_index(
    index: int,
    axis_size: int,
) -> int:
    """Normalize a take index with negative-offset support and bounds checks.

    Called by :func:`take` for each index value. Applies Python-style
    negative indexing, then validates the result is in ``[0, axis_size)``.

    Parameters
    ----------
    index : int
        Raw index (may be negative).
    axis_size : int
        Length of the axis being indexed.

    Returns
    -------
    int
        Non-negative index in ``[0, axis_size)``.

    Raises
    ------
    IndexError
        If the normalized index is out of bounds.

    Notes
    -----
    Unlike NumPy's ``take``, there is no ``mode='wrap'`` or
    ``mode='clip'``; out-of-range indices always raise.
    """
    if index < 0:
        index += axis_size

    if index < 0 or index >= axis_size:
        raise take_index_out_of_range(index=index, size=axis_size)

    return index
