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

"""Join and expand array dimensions along new or existing axes.

Provides :func:`expand_dims`, :func:`concatenate`, and :func:`stack` for
combining arrays with compatible shapes.
"""

from typing import Any, Iterable

from ._axis import normalize_axis, normalize_insert_axis
from ._dtype import cast_value, promote_dtypes
from ._errors import (
    at_least_one_array_required,
    concatenate_dimension_mismatch,
    concatenate_ndim_mismatch,
    concatenate_zero_dimensional,
    stack_shape_mismatch,
)
from ._factory import contiguous_ndarray, view_ndarray
from ._iter import iter_indices
from ._shape import flat_index, size_of_shape
from ._validation import coerce_to_ndarray, coerce_to_ndarray_iterable
from .ndarray import ndarray


def expand_dims(
    array: ndarray,
    axis: int,
) -> ndarray:
    """
    Expand the shape of an array by inserting a new axis.

    Parameters
    ----------
    array : array_like
        Input array.
    axis : int
        Position of the new length-1 axis. Negative values count from the
        end, and values in ``[-ndim - 1, ndim]`` are supported after
        normalization.

    Returns
    -------
    ndarray
        View of ``array`` with ``array.shape[axis]`` equal to ``1`` and
        stride ``0`` along the inserted axis.

    Raises
    ------
    ValueError
        If ``axis`` is out of range for insertion.

    Notes
    -----
    Returns a view rather than a copy, matching NumPy behavior for this
    operation.

    Examples
    --------
    >>> x = array([1, 2, 3])
    >>> expand_dims(x, 0).shape
    (1, 3)
    >>> expand_dims(x, 1).shape
    (3, 1)
    """
    array = coerce_to_ndarray(array)
    axis = normalize_insert_axis(axis, array.ndim)

    new_shape = list(array.shape)
    new_strides = list(array.strides)

    new_shape.insert(axis, 1)
    new_strides.insert(axis, 0)

    return view_ndarray(
        array.buffer,
        tuple(new_shape),
        tuple(new_strides),
        offset=array.offset,
        dtype=array.dtype,
    )


def concatenate(
    arrays: Iterable[ndarray],
    axis: int = 0,
) -> ndarray:
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays : sequence of array_like
        Arrays to join. All arrays must have the same shape except along
        ``axis``.
    axis : int (default: 0)
        Axis along which arrays will be joined.

    Returns
    -------
    ndarray
        The concatenated array with dtype promoted from the inputs.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, contains a zero-dimensional array, has
        mismatched ranks, or has incompatible shapes on non-concatenation
        axes.

    Notes
    -----
    Uses a buffer-copy fast path when every input is C-contiguous;
    otherwise falls back to element-wise gathering. NumPy additionally
    supports ``out`` and custom casting policies.

    Examples
    --------
    >>> a = array([[1, 2], [3, 4]])
    >>> b = array([[5, 6]])
    >>> concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    arrays = coerce_to_ndarray_iterable(arrays)

    if len(arrays) == 0:
        raise at_least_one_array_required(operation="concatenate")

    first = arrays[0]

    if first.ndim == 0:
        raise concatenate_zero_dimensional()

    axis = normalize_axis(axis, first.ndim)

    for array in arrays[1:]:
        if array.ndim != first.ndim:
            raise concatenate_ndim_mismatch(
                left_ndim=first.ndim,
                right_ndim=array.ndim,
            )

        for current_axis in range(first.ndim):
            if current_axis == axis:
                continue

            if array.shape[current_axis] != first.shape[current_axis]:
                raise concatenate_dimension_mismatch(
                    concat_axis=axis,
                    mismatch_axis=current_axis,
                    expected=first.shape[current_axis],
                    got=array.shape[current_axis],
                )

    result_shape = list(first.shape)
    result_shape[axis] = sum(array.shape[axis] for array in arrays)
    result_shape = tuple(result_shape)
    result_dtype = promote_dtypes(*(array.dtype for array in arrays))

    outer_shape = first.shape[:axis]
    inner_shape = first.shape[axis + 1 :]

    if all(array._is_c_contiguous() for array in arrays):
        inner_count = size_of_shape(inner_shape) if inner_shape else 1
        skip_cast = all(array.dtype is result_dtype for array in arrays)
        result_buffer: list[Any] = []

        for outer_index in iter_indices(outer_shape):
            for array in arrays:
                axis_size = array.shape[axis]

                if inner_shape:
                    start = array.offset + flat_index(
                        outer_index + (0,) + (0,) * len(inner_shape),
                        array.shape,
                    )
                else:
                    start = array.offset + flat_index(outer_index, array.shape)

                block_size = axis_size * inner_count
                chunk = array.buffer[start : start + block_size]

                if skip_cast:
                    result_buffer.extend(chunk)
                else:
                    result_buffer.extend(
                        cast_value(result_dtype, value) for value in chunk
                    )

        return contiguous_ndarray(
            result_buffer,
            result_shape,
            dtype=result_dtype,
            skip_cast=skip_cast,
        )

    # Non-contiguous inputs: gather element-by-element in concat order.
    result_buffer = []

    for outer_index in iter_indices(outer_shape):
        for array in arrays:
            for axis_index in range(array.shape[axis]):
                for inner_index in iter_indices(inner_shape):
                    source_index = outer_index + (axis_index,) + inner_index
                    result_buffer.append(
                        cast_value(result_dtype, array._get_value(source_index))
                    )

    return contiguous_ndarray(
        result_buffer,
        result_shape,
        dtype=result_dtype,
    )


def stack(
    arrays: Iterable[ndarray],
    axis: int = 0,
) -> ndarray:
    """
    Stack a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays : sequence of array_like
        Arrays to stack. All inputs must have the same shape.
    axis : int (default: 0)
        Index of the new axis in the result.

    Returns
    -------
    ndarray
        The stacked array. The ``i``-th slice along ``axis`` corresponds
        to ``arrays[i]``.

    Raises
    ------
    ValueError
        If ``arrays`` is empty or if any input shape differs from the
        first array.

    Notes
    -----
    Implemented via :func:`expand_dims` followed by :func:`concatenate`,
    matching the conceptual NumPy algorithm.

    Examples
    --------
    >>> a = array([1, 2, 3])
    >>> b = array([4, 5, 6])
    >>> stack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    arrays = coerce_to_ndarray_iterable(arrays)

    if len(arrays) == 0:
        raise at_least_one_array_required(operation="stack")

    first = arrays[0]
    axis = normalize_insert_axis(axis, first.ndim)

    for array in arrays[1:]:
        if array.shape != first.shape:
            raise stack_shape_mismatch(
                reference=first.shape,
                other=array.shape,
            )

    expanded = [expand_dims(array, axis) for array in arrays]

    return concatenate(expanded, axis=axis)
