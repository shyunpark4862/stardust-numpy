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

"""Array creation from Python sequences and built-in patterns.

Public constructors such as :func:`array`, :func:`zeros`, :func:`ones`,
:func:`full`, :func:`arange`, triangle helpers, and :func:`diag`.
"""

from __future__ import annotations

from typing import Any

from ._dtype import (
    DTypeLike,
    as_dtype,
    cast_sequence,
    cast_value,
    infer_dtype_from_values,
)
from ._errors import (
    arange_step_zero,
    array_expects_array_like,
    diag_requires_1d_or_2d,
    nested_sequence_must_be_list_or_tuple,
    nested_sequences_must_match_shape,
    requires_integer,
    requires_non_negative_int,
)
from ._factory import contiguous_ndarray
from ._iter import iter_indices
from ._shape import normalize_shape, size_of_shape
from ._validation import (
    BASIC_ARRAY_LIKE_TYPES,
    coerce_to_ndarray,
    require_number,
)
from .ndarray import ndarray


def array(
    obj: Any,
    dtype: DTypeLike | None = None,
) -> ndarray:
    """
    Create an array from array-like input.

    Parameters
    ----------
    obj : array_like
        Input data. Must be a nested or flat ``list``, ``tuple``, or
        ``range``. Nested sequences must be rectangular (all inner
        sequences have the same shape).
    dtype : DTypeLike, optional
        Data type of the returned array. If not specified, the dtype is
        inferred from the element values. Default is None.

    Returns
    -------
    ndarray
        Array interpretation of the input data.

    Notes
    -----
    Unlike NumPy, only ``list``, ``tuple``, and ``range`` are accepted as
    array-like inputs. Existing ``ndarray`` instances are not accepted;
    use :meth:`ndarray.copy` instead. Supported dtypes are ``bool``,
    ``int``, ``float``, ``complex``, and ``str`` only. Object arrays,
    structured dtypes, and the ``copy``, ``order``, and ``ndmin``
    keyword arguments are not supported.

    Examples
    --------
    >>> import numpy as np
    >>> np.array([1, 2, 3])
    array([1, 2, 3])
    >>> np.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])
    """
    if not isinstance(obj, BASIC_ARRAY_LIKE_TYPES):
        raise array_expects_array_like()

    buffer, shape = _flatten_nested_sequence(obj)

    if dtype is not None:
        target = as_dtype(dtype)
    elif buffer:
        target = infer_dtype_from_values(buffer)
    else:
        target = float

    return contiguous_ndarray(buffer, shape, dtype=target)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DTypeLike = int,
) -> ndarray:
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new array.
    dtype : DTypeLike, optional
        Desired data type for the array. Default is ``int``.

    Returns
    -------
    ndarray
        Array of zeros with the given shape and dtype.

    Notes
    -----
    Unlike NumPy, the ``order`` keyword is not supported; arrays are
    always C-contiguous. Only ``bool``, ``int``, ``float``, ``complex``,
    and ``str`` dtypes are supported.

    Examples
    --------
    >>> import numpy as np
    >>> np.zeros(3)
    array([0, 0, 0])
    >>> np.zeros((2, 2), dtype=float)
    array([[0., 0.],
           [0., 0.]])
    """
    return full(shape, 0, dtype=dtype)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DTypeLike = int,
) -> ndarray:
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new array.
    dtype : DTypeLike, optional
        Desired data type for the array. Default is ``int``.

    Returns
    -------
    ndarray
        Array of ones with the given shape and dtype.

    Notes
    -----
    Unlike NumPy, the ``order`` keyword is not supported. Only ``bool``,
    ``int``, ``float``, ``complex``, and ``str`` dtypes are supported.
    """
    return full(shape, 1, dtype=dtype)


def full(
    shape: int | tuple[int, ...],
    fill_value: Any,
    *,
    dtype: DTypeLike | None = None,
) -> ndarray:
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new array.
    fill_value : scalar or array_like
        Fill value. Cast to the target dtype.
    dtype : DTypeLike, optional
        Desired data type for the array. If not specified, inferred from
        ``fill_value``. Default is None.

    Returns
    -------
    ndarray
        Array of ``fill_value`` with the given shape and dtype.

    Notes
    -----
    Unlike NumPy, the ``order`` keyword is not supported. Only ``bool``,
    ``int``, ``float``, ``complex``, and ``str`` dtypes are supported.
    """
    shape = normalize_shape(shape)

    if dtype is None:
        target = infer_dtype_from_values([fill_value])
    else:
        target = as_dtype(dtype)

    fill_value = cast_value(target, fill_value)

    return contiguous_ndarray(
        [fill_value] * size_of_shape(shape),
        shape,
        dtype=target,
        skip_cast=True,
    )


def arange(
    start: Any,
    stop: Any | None = None,
    step: Any = 1,
    *,
    dtype: DTypeLike | None = None,
) -> ndarray:
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).

    Parameters
    ----------
    start : int or float
        Start of interval. If ``stop`` is omitted, ``start`` is treated
        as ``stop`` and the interval starts at 0.
    stop : int or float, optional
        End of interval (exclusive). Default is None.
    step : int or float, optional
        Spacing between values. Default is 1.
    dtype : DTypeLike, optional
        Data type of the returned array. If not specified, inferred from
        the generated values. Default is None.

    Returns
    -------
    ndarray
        Array of evenly spaced values.

    Notes
    -----
    Unlike NumPy, only numeric ``start``, ``stop``, and ``step`` values
    are accepted (``bool`` is not treated as an integer here). The
    ``like`` keyword and object-dtype sequences are not supported.

    Examples
    --------
    >>> import numpy as np
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(1, 5, 2)
    array([1, 3])
    """
    if stop is None:
        start, stop = 0, start

    start = require_number(start, name="start")
    stop = require_number(stop, name="stop")
    step = require_number(step, name="step")

    if step == 0:
        raise arange_step_zero()

    values = []
    current = start

    if step > 0:
        while current < stop:
            values.append(current)
            current += step
    else:
        # Descending sequence: stop is still exclusive on the lower end.
        while current > stop:
            values.append(current)
            current += step

    if dtype is None:
        if values:
            target = infer_dtype_from_values(values)
        else:
            target = float
    else:
        target = as_dtype(dtype)

    return contiguous_ndarray(
        cast_sequence(values, target),
        (len(values),),
        dtype=target,
    )


def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    *,
    dtype: DTypeLike = float,
) -> ndarray:
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of columns. Defaults to ``n``.
    k : int, optional
        Index of the diagonal. ``0`` refers to the main diagonal,
        a positive value to an upper diagonal, and a negative value to a
        lower diagonal. Default is 0.
    dtype : DTypeLike, optional
        Data type of the returned array. Default is ``float``.

    Returns
    -------
    ndarray
        An array where all elements are equal to zero, except for the
        `k`-th diagonal, whose values are equal to one.

    Notes
    -----
    Unlike NumPy, the ``order`` keyword is not supported. Only ``bool``,
    ``int``, ``float``, ``complex``, and ``str`` dtypes are supported.
    """
    n, m = _validate_triangle_dimensions(n, m, name="eye")
    k = _validate_triangle_k(k, name="eye")
    target = as_dtype(dtype)
    values = [
        cast_value(target, 1 if column == row + k else 0)
        for row in range(n)
        for column in range(m)
    ]

    return contiguous_ndarray(values, (n, m), dtype=target)


def tri(
    n: int,
    m: int | None = None,
    k: int = 0,
    *,
    dtype: DTypeLike = float,
) -> ndarray:
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of columns. Defaults to ``n``.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        ``k = 0`` is the main diagonal, ``k < 0`` is below it, and
        ``k > 0`` is above. Default is 0.
    dtype : DTypeLike, optional
        Data type of the returned array. Default is ``float``.

    Returns
    -------
    ndarray
        Array with its lower triangle filled with ones and zeros elsewhere.

    Notes
    -----
    Unlike NumPy, the ``order`` keyword is not supported. Only ``bool``,
    ``int``, ``float``, ``complex``, and ``str`` dtypes are supported.
    """
    n, m = _validate_triangle_dimensions(n, m, name="tri")
    k = _validate_triangle_k(k, name="tri")
    target = as_dtype(dtype)

    values = [
        cast_value(
            target,
            1 if _in_lower_triangle(row, column, k) else 0,
        )
        for row in range(n)
        for column in range(m)
    ]

    return contiguous_ndarray(values, (n, m), dtype=target)


def tril(
    a: Any,
    k: int = 0,
) -> ndarray:
    """
    Lower triangle of an array.

    Return a copy of the input array with elements above the `k`-th
    diagonal set to zero.

    Parameters
    ----------
    a : array_like
        Input array. One-dimensional inputs are treated as a square
        matrix before extracting the lower triangle.
    k : int, optional
        Diagonal above which to zero elements. ``k = 0`` (the default)
        is the main diagonal, ``k < 0`` is below it, and ``k > 0`` is
        above.

    Returns
    -------
    ndarray
        Lower triangle of `a`, of the same shape and dtype as `a`.

    Notes
    -----
    Unlike NumPy, only ``bool``, ``int``, ``float``, ``complex``, and
    ``str`` dtypes are supported. Object arrays are not supported.
    """
    a = coerce_to_ndarray(a)
    k = _validate_triangle_k(k, name="tril")
    source = _as_2d_triangle_source(a)
    fill_value = cast_value(source.dtype, 0)
    result_buffer = []

    for source_index in iter_indices(source.shape):
        row, column = _triangle_row_column(source_index, source.shape)

        if _in_lower_triangle(row, column, k):
            result_buffer.append(source[source_index])
        else:
            result_buffer.append(fill_value)

    return contiguous_ndarray(result_buffer, source.shape, dtype=source.dtype)


def triu(
    a: Any,
    k: int = 0,
) -> ndarray:
    """
    Upper triangle of an array.

    Return a copy of the input array with elements below the `k`-th
    diagonal set to zero.

    Parameters
    ----------
    a : array_like
        Input array. One-dimensional inputs are treated as a square
        matrix before extracting the upper triangle.
    k : int, optional
        Diagonal below which to zero elements. ``k = 0`` (the default)
        is the main diagonal, ``k < 0`` is below it, and ``k > 0`` is
        above.

    Returns
    -------
    ndarray
        Upper triangle of `a`, of the same shape and dtype as `a`.

    Notes
    -----
    Unlike NumPy, only ``bool``, ``int``, ``float``, ``complex``, and
    ``str`` dtypes are supported. Object arrays are not supported.
    """
    a = coerce_to_ndarray(a)
    k = _validate_triangle_k(k, name="triu")
    source = _as_2d_triangle_source(a)
    fill_value = cast_value(source.dtype, 0)
    result_buffer = []

    for source_index in iter_indices(source.shape):
        row, column = _triangle_row_column(source_index, source.shape)

        if _in_upper_triangle(row, column, k):
            result_buffer.append(source[source_index])
        else:
            result_buffer.append(fill_value)

    return contiguous_ndarray(result_buffer, source.shape, dtype=source.dtype)


def diag(
    v: Any,
    k: int = 0,
) -> ndarray:
    """
    Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    v : array_like
        If `v` is a 1-D array, return a 2-D array with `v` on the
        `k`-th diagonal. If `v` is a 2-D array, return a 1-D array
        containing the `k`-th diagonal.
    k : int, optional
        Diagonal in question. ``0`` refers to the main diagonal,
        a positive value to an upper diagonal, and a negative value to
        a lower diagonal. Default is 0.

    Returns
    -------
    ndarray
        The extracted diagonal or newly constructed diagonal array.

    Notes
    -----
    Unlike NumPy, only 1-D and 2-D inputs are supported. Only ``bool``,
    ``int``, ``float``, ``complex``, and ``str`` dtypes are supported.
    """
    v = coerce_to_ndarray(v)
    k = _validate_triangle_k(k, name="diag")

    if v.ndim == 1:
        return _diag_from_1d(v, k)

    if v.ndim == 2:
        return _diag_from_2d(v, k)

    raise diag_requires_1d_or_2d(ndim=v.ndim)


_NESTED_SEQUENCE_TYPES = (list, tuple)


def _flatten_nested_sequence(obj: Any) -> tuple[list[Any], tuple[int, ...]]:
    """Flatten a nested list/tuple/range into a buffer and shape tuple.

    Recursively walks nested ``list``/``tuple`` sequences to produce a flat
    element buffer and the corresponding shape. Called by :func:`array`.

    Parameters
    ----------
    obj : list, tuple, or range
        Input sequence. Top-level ``range`` objects are converted to a
        1-D buffer directly.

    Returns
    -------
    buffer : list
        Flat list of leaf elements in C (row-major) order.
    shape : tuple of int
        Dimensions inferred from nesting depth and inner sequence lengths.

    Raises
    ------
    TypeError
        If a nested level is not a ``list`` or ``tuple``.
    ValueError
        If sibling inner sequences at the same level have mismatched shapes.

    Notes
    -----
    Unlike NumPy's ``array`` constructor, ``ndarray`` instances and arbitrary
    iterables are not accepted here; callers must restrict inputs beforehand.
    """
    if isinstance(obj, range):
        values = list(obj)

        return values, (len(values),)

    if not isinstance(obj, _NESTED_SEQUENCE_TYPES):
        raise nested_sequence_must_be_list_or_tuple()

    if len(obj) == 0:
        return [], (0,)

    if not isinstance(obj[0], _NESTED_SEQUENCE_TYPES):
        # Top-level sequence is already 1-D.
        return list(obj), (len(obj),)

    first_buffer, first_shape = _flatten_nested_sequence(obj[0])
    buffer = list(first_buffer)

    for item in obj[1:]:
        if not isinstance(item, _NESTED_SEQUENCE_TYPES):
            raise nested_sequence_must_be_list_or_tuple()

        item_buffer, item_shape = _flatten_nested_sequence(item)

        if item_shape != first_shape:
            raise nested_sequences_must_match_shape()

        buffer.extend(item_buffer)

    return buffer, (len(obj),) + first_shape


def _validate_triangle_dimensions(
    n: int,
    m: int | None,
    *,
    name: str,
) -> tuple[int, int]:
    """Validate non-negative row/column counts for triangle constructors.

    Used by :func:`eye` and :func:`tri`.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int or None
        Number of columns; defaults to ``n`` when ``None``.
    name : str
        Function name for error messages.

    Returns
    -------
    n, m : int
        Validated dimensions.

    Raises
    ------
    TypeError
        If ``n`` or ``m`` is not a non-negative integer.
    """
    if not isinstance(n, int) or n < 0:
        raise requires_non_negative_int(name=name, parameter="N")

    if m is None:
        m = n
    elif not isinstance(m, int) or m < 0:
        raise requires_non_negative_int(name=name, parameter="M")

    return n, m


def _validate_triangle_k(
    k: int,
    *,
    name: str,
) -> int:
    """Validate that ``k`` is an integer diagonal offset.

    Used by :func:`eye`, :func:`tri`, :func:`tril`, :func:`triu`, and
    :func:`diag`.

    Parameters
    ----------
    k : int
        Diagonal offset (may be negative).
    name : str
        Function name for error messages.

    Returns
    -------
    int
        The validated ``k`` value.

    Raises
    ------
    TypeError
        If ``k`` is not an integer.
    """
    if not isinstance(k, int):
        raise requires_integer(name=name, parameter="k")

    return k


def _in_lower_triangle(
    row: int,
    column: int,
    k: int,
) -> bool:
    """Return True when ``(row, column)`` is on or below diagonal ``k``.

    A position belongs to the lower triangle when
    ``column <= row + k``. Used by :func:`tri` and :func:`tril`.

    Parameters
    ----------
    row, column : int
        Zero-based matrix coordinates.
    k : int
        Diagonal offset (``0`` is the main diagonal).
    """
    return column <= row + k


def _in_upper_triangle(
    row: int,
    column: int,
    k: int,
) -> bool:
    """Return True when ``(row, column)`` is on or above diagonal ``k``.

    A position belongs to the upper triangle when
    ``column >= row + k``. Used by :func:`triu`.

    Parameters
    ----------
    row, column : int
        Zero-based matrix coordinates.
    k : int
        Diagonal offset (``0`` is the main diagonal).
    """
    return column >= row + k


def _triangle_row_column(
    source_index: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[int, int]:
    """Map a C-order index to row/column coordinates for triangle ops.

    Converts a flat or multi-dimensional index into ``(row, column)``
    for :func:`tril` and :func:`triu`.

    Parameters
    ----------
    source_index : tuple of int
        Index into the source array.
    shape : tuple of int
        Shape of the source array.

    Returns
    -------
    row, column : int
        Matrix coordinates. For 1-D inputs (treated as a single row),
        ``row`` is always ``0``.

    Notes
    -----
    Matches NumPy's convention of promoting 1-D inputs to a square matrix
    before extracting triangles.
    """
    if len(shape) == 0:
        return 0, 0

    if len(shape) == 1:
        # 1-D input is treated as a single row when used with tril/triu.
        return 0, source_index[0]

    return source_index[-2], source_index[-1]


def _as_2d_triangle_source(
    a: ndarray,
) -> ndarray:
    """Promote 1-D inputs to a square matrix for triangle extraction.

    Called by :func:`tril` and :func:`triu` before zeroing off-triangle
    elements.

    Parameters
    ----------
    a : ndarray
        Input array. Multi-dimensional arrays are returned unchanged.

    Returns
    -------
    ndarray
        ``a`` itself when ``a.ndim != 1``; otherwise a square
        ``(n, n)`` matrix formed by tiling the 1-D values row-wise.

    Notes
    -----
    Follows NumPy's rule that a 1-D array of length ``n`` is treated as
    an ``n × n`` matrix whose rows are copies of the input vector.
    """
    if a.ndim != 1:
        return a

    length = a.shape[0]
    buffer = [a[column] for row in range(length) for column in range(length)]

    return contiguous_ndarray(buffer, (length, length), dtype=a.dtype)


def _diag_from_1d(
    v: ndarray,
    k: int,
) -> ndarray:
    """Construct a square matrix with ``v`` placed on diagonal ``k``.

    Called by :func:`diag` when the input is one-dimensional.

    Parameters
    ----------
    v : ndarray
        1-D source vector.
    k : int
        Diagonal offset. ``k >= 0`` places values on an upper diagonal;
        ``k < 0`` shifts the diagonal downward.

    Returns
    -------
    ndarray
        Square matrix of side ``len(v) + abs(k)``, zero-filled except on
        the requested diagonal.

    Notes
    -----
    Uses flat-buffer access when ``v`` is C-contiguous. Unlike NumPy,
    only 1-D and 2-D inputs are supported by :func:`diag`.
    """
    length = v.shape[0]
    size = length + abs(k)
    fill_value = cast_value(v.dtype, 0)
    buffer = [fill_value] * (size * size)
    v_values = (
        v._flat_buffer_slice()
        if v._is_c_contiguous()
        else list(v._iter_values())
    )

    for index, value in enumerate(v_values):
        if k >= 0:
            row = index
            column = index + k
        else:
            # Negative k shifts the diagonal down from the main diagonal.
            row = index - k
            column = index

        buffer[row * size + column] = value

    return contiguous_ndarray(
        buffer,
        (size, size),
        dtype=v.dtype,
        skip_cast=True,
    )


def _diag_from_2d(
    v: ndarray,
    k: int,
) -> ndarray:
    """Extract diagonal ``k`` from a 2-D array as a 1-D array.

    Called by :func:`diag` when the input is two-dimensional.

    Parameters
    ----------
    v : ndarray
        2-D source matrix.
    k : int
        Diagonal offset. Positive values select upper diagonals;
        negative values select lower diagonals.

    Returns
    -------
    ndarray
        1-D array containing ``v[row, row + k]`` for all valid ``row``
        positions.

    Notes
    -----
    The extracted length is ``min(rows - max(0, -k), columns - max(0, k))``,
    matching NumPy's diagonal extraction semantics.
    """
    rows, columns = v.shape
    values = []
    row = max(0, -k)

    while row < rows and row + k < columns:
        values.append(v[row, row + k])
        row += 1

    return contiguous_ndarray(values, (len(values),), dtype=v.dtype)
