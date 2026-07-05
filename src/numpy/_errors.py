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

"""Centralized error message helpers for consistent, contextual exceptions.

Each factory function builds a typed exception with a formatted message so
call sites share wording and context (shapes, dtypes, axes, and so on).
"""

from __future__ import annotations

from typing import Any

SUPPORTED_DTYPES = "bool, int, float, complex, str"


def repr_shape(shape: tuple[int, ...]) -> str:
    """Return a string representation of an array shape."""
    return str(shape)


def repr_dtype(dtype: type[Any]) -> str:
    """Return the name of a dtype type."""
    return dtype.__name__


def repr_type(value: Any) -> str:
    """Return the name of a value's type."""
    return type(value).__name__


def axis_must_be_int() -> TypeError:
    """Raise TypeError when axis is not an integer."""
    return TypeError("axis must be an integer")


def axis_out_of_range(axis: int, ndim: int) -> ValueError:
    """Raise ValueError when axis is out of bounds for the array ndim."""
    if ndim == 0:
        return ValueError(
            f"axis {axis} is out of bounds for a 0-dimensional array"
        )

    return ValueError(
        f"axis {axis} is out of bounds for array with ndim {ndim} "
        f"(valid axes: 0..{ndim - 1})"
    )


def insert_axis_out_of_range(axis: int, ndim: int) -> ValueError:
    """Raise ValueError when insert axis is out of bounds."""
    return ValueError(
        f"axis {axis} is out of bounds for inserting into array with ndim {ndim} "
        f"(valid axes: 0..{ndim})"
    )


def axes_must_be_int_or_tuple() -> TypeError:
    """Raise TypeError when axes is not an int or tuple of ints."""
    return TypeError("axes must be an int or tuple of ints")


def duplicate_axis() -> ValueError:
    """Raise ValueError when axes contain duplicates."""
    return ValueError("axes must not contain duplicates")


def transpose_axes_mismatch(
    *,
    ndim: int,
    axes: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when transpose axes are not a full permutation."""
    return ValueError(
        f"transpose() requires a permutation of all {ndim} axes; got {len(axes)} axes {axes}"
    )


def shape_must_be_int_or_tuple(*, name: str = "shape") -> TypeError:
    """Raise TypeError when shape is not an int or tuple of ints."""
    return TypeError(f"{name} must be an int or tuple of ints")


def shape_dimensions_must_be_integers() -> TypeError:
    """Raise TypeError when shape dimensions are not integers."""
    return TypeError("shape dimensions must be integers")


def negative_dimension_not_allowed() -> ValueError:
    """Raise ValueError when a shape dimension is negative."""
    return ValueError("shape dimensions must be non-negative")


def slice_step_zero() -> ValueError:
    """Raise ValueError when a slice step is zero."""
    return ValueError("slice step cannot be zero")


def shape_strides_length_mismatch(
    *,
    shape_len: int,
    strides_len: int,
) -> ValueError:
    """Raise ValueError when shape and strides lengths differ."""
    return ValueError(
        f"shape and strides must have the same length; "
        f"got shape with {shape_len} dimension(s) and strides with {strides_len}"
    )


def reshape_incompatible(
    *,
    size: int,
    shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when reshape target size does not match array size."""
    requested = 1

    for dimension in shape:
        requested *= dimension

    return ValueError(
        f"cannot reshape array of size {size} into shape {repr_shape(shape)} "
        f"(requested {requested} element(s))"
    )


def reshape_unknown_dimension_invalid(
    *, size: int, known_size: int
) -> ValueError:
    """Raise ValueError when an inferred reshape dimension is invalid."""
    return ValueError(
        f"cannot reshape array of size {size} with inferred dimension "
        f"(known dimensions multiply to {known_size})"
    )


def reshape_only_one_unknown() -> ValueError:
    """Raise ValueError when reshape has more than one unknown dimension."""
    return ValueError("reshape() allows at most one unknown dimension (-1)")


def cannot_squeeze_axis(*, axis: int, size: int) -> ValueError:
    """Raise ValueError when squeeze axis size is not 1."""
    return ValueError(
        f"cannot squeeze axis {axis} with size {size} (size must be 1)"
    )


def cannot_broadcast_to_fewer_dimensions(
    *,
    from_shape: tuple[int, ...],
    to_ndim: int,
) -> ValueError:
    """Raise ValueError when broadcasting to fewer dimensions."""
    return ValueError(
        f"cannot broadcast array with shape {repr_shape(from_shape)} "
        f"to fewer than {len(from_shape)} dimension(s)"
    )


def cannot_broadcast_shape(
    *,
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when broadcast shapes are incompatible."""
    return ValueError(
        f"cannot broadcast shape {repr_shape(from_shape)} to {repr_shape(to_shape)}"
    )


def broadcast_operands_incompatible(
    *,
    shapes: tuple[tuple[int, ...], ...],
) -> ValueError:
    """Raise ValueError when operands cannot be broadcast together."""
    formatted = ", ".join(repr_shape(shape) for shape in shapes)

    return ValueError(
        f"operands could not be broadcast together with shapes ({formatted})"
    )


def dot_shape_mismatch(
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    left_cols: int,
    right_rows: int,
) -> ValueError:
    """Raise ValueError when dot inner dimensions do not match."""
    return ValueError(
        f"dot: incompatible shapes {repr_shape(left_shape)} and "
        f"{repr_shape(right_shape)} — inner dimensions {left_cols} and {right_rows} "
        f"do not match"
    )


def matmul_shape_mismatch(
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    left_cols: int,
    right_rows: int,
) -> ValueError:
    """Raise ValueError when matmul inner dimensions do not match."""
    return ValueError(
        f"matmul: incompatible shapes {repr_shape(left_shape)} @ "
        f"{repr_shape(right_shape)} — inner dimensions {left_cols} and {right_rows} "
        f"do not match"
    )


def vdot_size_mismatch(*, left_size: int, right_size: int) -> ValueError:
    """Raise ValueError when vdot operand sizes do not match."""
    return ValueError(
        f"vdot: size mismatch — left has {left_size} element(s), "
        f"right has {right_size} element(s)"
    )


def matmul_scalar_operand() -> ValueError:
    """Raise ValueError when matmul receives a scalar operand."""
    return ValueError("matmul() does not support scalar operands")


def dot_not_implemented(*, ndim: int) -> NotImplementedError:
    """Raise NotImplementedError when dot ndim exceeds supported rank."""
    return NotImplementedError(
        f"dot() supports arrays with ndim at most 2; got ndim {ndim}"
    )


def index_must_be_int() -> TypeError:
    """Raise TypeError when an array index is not an integer."""
    return TypeError("array index must be an integer")


def index_out_of_bounds(*, index: int, axis: int, size: int) -> IndexError:
    """Raise IndexError when an index is out of bounds for an axis."""
    last = size - 1

    return IndexError(
        f"index {index} is out of bounds for axis {axis} with size {size} "
        f"(valid range: 0..{last})"
    )


def invalid_fancy_index(*, got: Any) -> TypeError:
    """Raise TypeError when a fancy index has an invalid type."""
    return TypeError(
        f"index must be int, slice, ndarray, None, or Ellipsis; got {repr_type(got)}"
    )


def single_ellipsis_allowed() -> IndexError:
    """Raise IndexError when an index contains more than one ellipsis."""
    return IndexError(
        "an index expression can contain at most one ellipsis (...)"
    )


def too_many_indices(*, ndim: int, index_count: int) -> IndexError:
    """Raise IndexError when too many indices are provided for ndim."""
    return IndexError(
        f"too many indices for array with ndim {ndim}: got {index_count} index object(s)"
    )


def fancy_index_must_be_int_or_bool() -> TypeError:
    """Raise TypeError when a fancy index is not int or bool."""
    return TypeError("array index must contain only integers or booleans")


def boolean_mask_must_be_bool(*, got: type[Any]) -> TypeError:
    """Raise TypeError when a boolean mask dtype is not bool."""
    return TypeError(
        f"boolean mask must have dtype bool; got {repr_dtype(got)}"
    )


def boolean_index_shape_mismatch(
    *,
    got: tuple[int, ...],
    expected: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when a boolean index shape does not match."""
    return ValueError(
        f"boolean index shape {repr_shape(got)} does not match "
        f"expected shape {repr_shape(expected)}"
    )


def index_must_be_int_slice_or_ellipsis() -> TypeError:
    """Raise TypeError when an index type is not int, slice, None, or Ellipsis."""
    return TypeError(
        "array indices must be integers, slices, None, or Ellipsis"
    )


def invalid_assignment_target() -> TypeError:
    """Raise TypeError when assignment targets a scalar element."""
    return TypeError(
        "assignment target must be an array slice, not a scalar element"
    )


def could_not_broadcast_assignment(
    *,
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when an assignment value cannot be broadcast."""
    return ValueError(
        f"could not broadcast assignment value from shape {repr_shape(from_shape)} "
        f"into shape {repr_shape(to_shape)}"
    )


def invalid_dtype(value: Any) -> TypeError:
    """Raise TypeError when a value is not a supported dtype."""
    return TypeError(
        f"{value!r} is not a supported dtype; expected one of {SUPPORTED_DTYPES}"
    )


def unsupported_dtype_for_operation(
    *, operation: str, dtype: type[Any]
) -> TypeError:
    """Raise TypeError when an operation does not support the dtype."""
    return TypeError(
        f"operation {operation!r} is not supported for dtype {repr_dtype(dtype)}"
    )


def incompatible_operand_dtypes(
    *,
    operation: str,
    left: type[Any],
    right: type[Any],
) -> TypeError:
    """Raise TypeError when operand dtypes are incompatible for an operation."""
    return TypeError(
        f"operation {operation!r} does not support operands with dtypes "
        f"{repr_dtype(left)} and {repr_dtype(right)}"
    )


def cannot_cast(*, value: Any, target: type[Any]) -> TypeError:
    """Raise TypeError when a value cannot be cast to the target dtype."""
    return TypeError(
        f"cannot cast value {value!r} ({repr_type(value)}) to {repr_dtype(target)}"
    )


def unsupported_dtype(*, dtype: type[Any]) -> TypeError:
    """Raise TypeError for an unsupported dtype."""
    return TypeError(f"unsupported dtype {repr_dtype(dtype)!r}")


def unsupported_element_type(*, value: Any) -> TypeError:
    """Raise TypeError when an array element type is not supported."""
    return TypeError(
        f"unsupported array element type {repr_type(value)}; "
        f"expected bool, int, float, complex, or str"
    )


def mixed_dtypes(*, dtypes: set[type[Any]]) -> ValueError:
    """Raise ValueError when array creation mixes incompatible dtypes."""
    formatted = ", ".join(sorted(repr_dtype(dtype) for dtype in dtypes))

    return ValueError(
        f"cannot create homogeneous array from mixed dtypes: {formatted}"
    )


def must_be_ndarray_or_array_like(*, name: str) -> TypeError:
    """Raise TypeError when an argument is not ndarray or array-like."""
    return TypeError(
        f"{name} must be an ndarray or array-like (list, tuple, range)"
    )


def scalar_operands_not_supported(*, name: str) -> ValueError:
    """Raise ValueError when an operation does not support scalar operands."""
    return ValueError(f"{name}() does not support scalar operands")


def must_be_number(*, name: str) -> TypeError:
    """Raise TypeError when an argument is not a number."""
    return TypeError(f"{name} must be a number (int, float, or complex)")


def at_least_one_array_required(*, operation: str) -> ValueError:
    """Raise ValueError when an operation requires at least one array."""
    return ValueError(f"{operation}() requires at least one array")


def concatenate_zero_dimensional() -> ValueError:
    """Raise ValueError when concatenate receives a 0-dimensional array."""
    return ValueError("concatenate() does not support 0-dimensional arrays")


def concatenate_ndim_mismatch(*, left_ndim: int, right_ndim: int) -> ValueError:
    """Raise ValueError when concatenate operands have different ndim."""
    return ValueError(
        f"concatenate() requires arrays with the same number of dimensions; "
        f"got ndim {left_ndim} and {right_ndim}"
    )


def concatenate_shape_mismatch(
    *,
    axis: int,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when concatenate shapes mismatch along an axis."""
    return ValueError(
        f"concatenate() shape mismatch along axis {axis}: "
        f"{repr_shape(left_shape)} and {repr_shape(right_shape)}"
    )


def stack_shape_mismatch(
    *,
    reference: tuple[int, ...],
    other: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when stack operands have different shapes."""
    return ValueError(
        f"stack() requires arrays with the same shape; "
        f"got {repr_shape(reference)} and {repr_shape(other)}"
    )


def cannot_reduce_empty_axis(
    *,
    operation: str,
    axis: int,
    shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when reducing an axis of size zero."""
    return ValueError(
        f"{operation}: cannot reduce axis {axis} with size 0 "
        f"in array with shape {repr_shape(shape)}"
    )


def empty_reduction(*, operation: str) -> ValueError:
    """Raise ValueError when reducing an empty array."""
    return ValueError(f"{operation} of empty array")


def cannot_cumulate_empty_axis(
    *,
    operation: str,
    axis: int,
    shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when cumulating an axis of size zero."""
    return ValueError(
        f"{operation}: cannot cumulate axis {axis} with size 0 "
        f"in array with shape {repr_shape(shape)}"
    )


def where_must_be_bool(*, got: Any | None = None) -> TypeError:
    """Raise TypeError when where is not bool or a bool array."""
    if got is None:
        return TypeError("where must be bool or an array with dtype bool")

    if isinstance(got, bool):
        return TypeError("where must be bool or an array with dtype bool")

    return TypeError(
        f"where must be bool or an array with dtype bool; got {repr_type(got)}"
    )


def out_dtype_mismatch(
    *, out_dtype: type[Any], result_dtype: type[Any]
) -> TypeError:
    """Raise TypeError when out dtype does not match the result dtype."""
    return TypeError(
        f"out has dtype {repr_dtype(out_dtype)}, "
        f"but result has dtype {repr_dtype(result_dtype)}"
    )


def out_shape_mismatch(
    *,
    out_shape: tuple[int, ...],
    result_shape: tuple[int, ...],
) -> ValueError:
    """Raise ValueError when out shape does not match the result shape."""
    return ValueError(
        f"out has shape {repr_shape(out_shape)}, "
        f"but result has shape {repr_shape(result_shape)}"
    )


def iteration_over_zero_d_array() -> TypeError:
    """Raise TypeError when iterating over a 0-dimensional array."""
    return TypeError("iteration over a 0-d array")


def len_of_unsized() -> TypeError:
    """Raise TypeError when len() is called on an unsized object."""
    return TypeError("len() of unsized object")


def array_expects_array_like() -> TypeError:
    """Raise TypeError when array() receives a non-array-like argument."""
    return TypeError(
        "array() expects an array-like sequence (list, tuple, range)"
    )


def nested_sequence_must_be_list_or_tuple() -> TypeError:
    """Raise TypeError when nested array input is not a list or tuple."""
    return TypeError("array() expects an array-like nested list or tuple")


def nested_sequences_must_match_shape() -> ValueError:
    """Raise ValueError when nested sequences have mismatched shapes."""
    return ValueError("all nested sequences must have the same shape")


def arange_step_zero() -> ValueError:
    """Raise ValueError when arange step is zero."""
    return ValueError("arange() step must not be zero")


def requires_non_negative_int(*, name: str, parameter: str) -> TypeError:
    """Raise TypeError when a parameter is not a non-negative integer."""
    return TypeError(f"{name}() requires a non-negative integer {parameter}")


def requires_integer(*, name: str, parameter: str) -> TypeError:
    """Raise TypeError when a parameter is not an integer."""
    return TypeError(f"{name}() requires an integer {parameter}")


def requires_at_least_2d(*, name: str) -> ValueError:
    """Raise ValueError when an array has fewer than two dimensions."""
    return ValueError(f"{name}() requires an array with at least 2 dimensions")


def axes_must_differ() -> ValueError:
    """Raise ValueError when axis1 and axis2 are the same."""
    return ValueError("axis1 and axis2 must be different")


def diag_requires_1d_or_2d(*, ndim: int) -> ValueError:
    """Raise ValueError when diag receives an array that is not 1-D or 2-D."""
    return ValueError(f"diag() requires a 1-D or 2-D array; got ndim {ndim}")


def take_index_out_of_range(*, index: int, size: int) -> IndexError:
    """Raise IndexError when a take index is out of bounds."""
    last = size - 1

    return IndexError(
        f"take index {index} is out of bounds for axis of size {size} "
        f"(valid range: 0..{last})"
    )


def take_indices_must_be_int(*, got: type[Any]) -> TypeError:
    """Raise TypeError when take indices dtype is not int."""
    return TypeError(f"take indices must have dtype int; got {repr_dtype(got)}")


def where_condition_must_be_bool(*, got: type[Any]) -> TypeError:
    """Raise TypeError when a where condition dtype is not bool."""
    return TypeError(
        f"where condition must have dtype bool; got {repr_dtype(got)}"
    )


def concatenate_dimension_mismatch(
    *,
    concat_axis: int,
    mismatch_axis: int,
    expected: int,
    got: int,
) -> ValueError:
    """Raise ValueError when concatenate array dimensions do not match."""
    return ValueError(
        f"concatenate(): dimension {mismatch_axis} must match for all arrays "
        f"(axis {concat_axis} is the concatenate axis); got {expected} and {got}"
    )


def integer_fancy_index_required() -> ValueError:
    """Raise ValueError when an integer fancy index is required."""
    return ValueError("integer fancy index is required")


def nditer_unsupported_op_flags(*, op_flags: str) -> ValueError:
    """Raise ValueError when nditer op_flags are unsupported."""
    return ValueError(
        f"unsupported op_flags {op_flags!r}; only 'readonly' is supported"
    )


def nditer_requires_operand() -> ValueError:
    """Raise ValueError when nditer has no operands."""
    return ValueError("nditer() requires at least one operand")


def nditer_invalid_operands(*, got: Any) -> TypeError:
    """Raise TypeError when nditer operands are invalid."""
    return TypeError(
        f"nditer operands must be an ndarray or a sequence of ndarrays; "
        f"got {repr_type(got)}"
    )
