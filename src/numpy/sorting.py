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

"""Return-sorted copies and unique value extraction.

Functional wrappers around :meth:`ndarray.sort` and a pure-Python
:func:`unique` with optional index, inverse, and count outputs.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from ._float_ops import order_key
from ._shape import flat_index, size_of_shape
from ._validation import coerce_to_ndarray
from .creation import array
from .ndarray import ndarray


def sort(
    a: ndarray,
    axis: int | None = -1,
) -> ndarray:
    """Return a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None (default: -1)
        Axis along which to sort. When ``None``, the array is flattened
        before sorting and a one-dimensional result is returned.

    Returns
    -------
    ndarray
        Sorted copy of `a`. The original array is not modified.

    Raises
    ------
    TypeError
        If `a` has a dtype that is not orderable (e.g. ``complex``).

    Notes
    -----
    Unlike :meth:`ndarray.sort`, this function never sorts in place.
    When ``axis is None``, the result is always one-dimensional even if
    `a` was multi-dimensional. Sorting uses a stable ordering with NaNs
    placed last and infinities ordered before finite values. Only
    ``bool``, ``int``, ``float``, and ``str`` dtypes are orderable.

    See Also
    --------
    ndarray.sort : Sort an array in place.
    ndarray.argsort : Return indices that would sort an array.
    """
    a = coerce_to_ndarray(a)

    if axis is None:
        # Functional API returns a new 1-D array; does not reshape `a` in place.
        values = list(a._iter_values())
        values.sort(key=order_key)
        return array(values, dtype=a.dtype)

    result = a.copy()
    result.sort(axis=axis)
    return result


def unique(
    a: ndarray,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> ndarray | tuple[Any, ...]:
    """Find the unique elements of an array.

    Parameters
    ----------
    a : array_like
        Input array. The array is flattened before finding unique values.
    return_index : bool (default: False)
        If True, also return the indices of the first occurrence of each
        unique value in the flattened input.
    return_inverse : bool (default: False)
        If True, also return the indices to reconstruct the flattened input
        from the unique values.
    return_counts : bool (default: False)
        If True, also return the number of times each unique value appears
        in the flattened input.

    Returns
    -------
    ndarray or tuple of ndarray
        The unique values in sorted order. When any ``return_*`` flag is
        True, a tuple is returned with the unique values first, followed
        by the requested auxiliary arrays in the order
        ``(unique, index, inverse, counts)``.

    Notes
    -----
    Unlike NumPy, there is no ``axis`` keyword and no ``equal_nan`` option.
    NaN values (real or complex) are treated as equal to each other and
    sorted after all finite values. Only the five supported element dtypes
    are allowed. Optional outputs are keyword-only.

    See Also
    --------
    sort : Return a sorted copy of an array.
    """
    a = coerce_to_ndarray(a)

    indexed_values = [
        (flat_index, value) for flat_index, value in enumerate(a._iter_values())
    ]
    sorted_values = sorted(
        indexed_values,
        key=lambda pair: _unique_sort_key(pair[1]),
    )

    groups: list[tuple[int, Any, list[int]]] = []

    for flat_index, value in sorted_values:
        # Merge consecutive equal values (NaNs compare equal here).
        if groups and _values_equal(groups[-1][1], value):
            groups[-1][2].append(flat_index)
            continue

        groups.append((flat_index, value, [flat_index]))

    unique_values = [group[1] for group in groups]
    first_indices = [group[0] for group in groups]
    counts = [len(group[2]) for group in groups]

    result = array(unique_values, dtype=a.dtype)

    if not (return_index or return_inverse or return_counts):
        return result

    outputs: list[Any] = [result]

    if return_index:
        outputs.append(array(first_indices, dtype=int))

    if return_inverse:
        flat_to_group: dict[int, int] = {}

        for group_index, group in enumerate(groups):
            for flat_index in group[2]:
                flat_to_group[flat_index] = group_index

        inverse = [
            flat_to_group[flat_index] for flat_index, _ in indexed_values
        ]

        outputs.append(array(inverse, dtype=int))

    if return_counts:
        outputs.append(array(counts, dtype=int))

    return tuple(outputs)


def _values_equal(
    left: Any,
    right: Any,
) -> bool:
    """Return whether two scalar values should be merged by :func:`unique`.

    Determines equality for grouping consecutive sorted elements. NaN
    values of the same floating type are treated as equal even though
    ``float('nan') == float('nan')`` is False in Python.

    Parameters
    ----------
    left, right : scalar
        Two element values to compare.

    Returns
    -------
    bool
        True if the values belong to the same unique group.

    Notes
    -----
    Unlike NumPy's ``equal_nan=False`` default, NaNs are always merged.
    Uses plain ``==`` for all non-NaN comparisons.
    """
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True

    if isinstance(left, complex) and isinstance(right, complex):
        if cmath.isnan(left) and cmath.isnan(right):
            return True

    return left == right


def _unique_sort_key(value: Any) -> tuple[Any, ...]:
    """Sort key placing NaNs after finite values for :func:`unique`.

    Produces a tuple suitable for Python's ``sorted`` that orders finite
    values before NaNs and compares complex numbers by ``(real, imag)``.

    Parameters
    ----------
    value : scalar
        Element value from the flattened input.

    Returns
    -------
    tuple
        ``(0, ...)`` prefix for finite values (with complex unpacked
        as ``(0, real, imag)``), or ``(1,)`` for NaN values so they
        sort after all finite entries.

    Notes
    -----
    Matches the NaN-last ordering used by :func:`sort` via
    :func:`numpy._float_ops.order_key`, but is specialized for the
    grouping pass in :func:`unique`.
    """
    if isinstance(value, float) and math.isnan(value):
        return (1,)

    if isinstance(value, complex):
        if cmath.isnan(value):
            return (1,)

        return (0, value.real, value.imag)

    return (0, value)
