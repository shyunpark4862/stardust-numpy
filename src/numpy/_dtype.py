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

"""Dtype predicates, casting, and promotion for supported scalar types."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, TypeAlias

from ._errors import (
    cannot_cast,
    incompatible_operand_dtypes,
    invalid_dtype,
    mixed_dtypes,
    unsupported_dtype,
    unsupported_dtype_for_operation,
    unsupported_element_type,
)

DTypeLike: TypeAlias = (
    type[bool] | type[int] | type[float] | type[complex] | type[str] | str
)
"""Union of supported dtype types and their string aliases."""

NUMERIC_DTYPES = frozenset({bool, int, float, complex})

REAL_NUMERIC_DTYPES = frozenset({bool, int, float})

ORDERABLE_DTYPES = frozenset({bool, int, float, str})

LOGICAL_DTYPES = frozenset({bool, int, float, complex})


def is_bool_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` is ``bool``."""
    return dtype is bool


def is_int_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` is ``int``."""
    return dtype is int


def is_float_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` is ``float``."""
    return dtype is float


def is_complex_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` is ``complex``."""
    return dtype is complex


def is_str_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` is ``str``."""
    return dtype is str


def is_numeric_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` is a numeric type (``bool``, ``int``, ``float``, or ``complex``)."""
    return dtype in NUMERIC_DTYPES


def is_orderable_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` supports ordering comparisons."""
    return dtype in ORDERABLE_DTYPES


def is_logical_dtype(dtype: type[Any]) -> bool:
    """Return ``True`` if ``dtype`` can participate in logical ufuncs."""
    return dtype in LOGICAL_DTYPES


def as_dtype(value: DTypeLike) -> type[Any]:
    """
    Normalize a dtype specifier to a supported Python scalar type.

    Parameters
    ----------
    value : type or str
        A dtype type (``bool``, ``int``, ``float``, ``complex``, or ``str``)
        or a case-insensitive string alias (for example ``"float"``).

    Returns
    -------
    type
        The corresponding scalar type.

    Raises
    ------
    TypeError
        If ``value`` is not a supported dtype or alias.

    Notes
    -----
    NumPy accepts ``numpy.dtype`` objects, size-qualified names such as
    ``"float64"``, and many other specifiers. This implementation only
    recognizes the five Python scalar types and their simple string names.
    """
    if isinstance(value, str):
        normalized = value.lower()

        if normalized in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[normalized]

        raise invalid_dtype(value)

    # Accept dtype types directly when they are among the supported set.
    if value in _DTYPE_ALIASES.values():
        return value

    raise invalid_dtype(value)


def require_numeric_dtype(
    dtype: type[Any],
    *,
    operation: str,
) -> None:
    """
    Require that ``dtype`` is numeric, raising otherwise.

    Parameters
    ----------
    dtype : type
        Dtype to validate.
    operation : str
        Name of the operation requesting the check; included in the error
        message.

    Raises
    ------
    TypeError
        If ``dtype`` is not numeric.
    """
    if not is_numeric_dtype(dtype):
        raise unsupported_dtype_for_operation(operation=operation, dtype=dtype)


def require_orderable_dtype(
    dtype: type[Any],
    *,
    operation: str,
) -> None:
    """
    Require that ``dtype`` supports ordering, raising otherwise.

    Parameters
    ----------
    dtype : type
        Dtype to validate.
    operation : str
        Name of the operation requesting the check; included in the error
        message.

    Raises
    ------
    TypeError
        If ``dtype`` is not orderable (for example ``complex``).
    """
    if not is_orderable_dtype(dtype):
        raise unsupported_dtype_for_operation(operation=operation, dtype=dtype)


def result_dtype_for_absolute(dtype: type[Any]) -> type[Any]:
    """
    Return the result dtype of :func:`numpy.absolute` for ``dtype``.

    Parameters
    ----------
    dtype : type
        Input element type.

    Returns
    -------
    type
        ``float`` for complex inputs; otherwise ``dtype`` unchanged.

    Notes
    -----
    NumPy's ``absolute`` returns an unsigned integer dtype for signed integer
    inputs. This implementation preserves the input dtype for real types.
    """
    if is_complex_dtype(dtype):
        return float

    return dtype


def require_logical_dtype(
    dtype: type[Any],
    *,
    operation: str,
) -> None:
    """
    Require that ``dtype`` can be used in logical ufuncs, raising otherwise.

    Parameters
    ----------
    dtype : type
        Dtype to validate.
    operation : str
        Name of the operation requesting the check; included in the error
        message.

    Raises
    ------
    TypeError
        If ``dtype`` is not logical (for example ``str``).
    """
    if not is_logical_dtype(dtype):
        raise unsupported_dtype_for_operation(operation=operation, dtype=dtype)


def require_comparable_dtypes(
    left: type[Any],
    right: type[Any],
    *,
    operation: str,
) -> None:
    """
    Require that two dtypes can be promoted together, raising otherwise.

    Parameters
    ----------
    left : type
        Dtype of the left operand.
    right : type
        Dtype of the right operand.
    operation : str
        Name of the operation requesting the check; included in the error
        message.

    Raises
    ------
    TypeError
        If the dtypes cannot be promoted (for example ``int`` with ``str``).
    """
    try:
        promote_dtypes(left, right)
    except ValueError as error:
        raise incompatible_operand_dtypes(
            operation=operation,
            left=left,
            right=right,
        ) from error


def cast_value(dtype: type[Any], value: Any) -> Any:
    """
    Cast a scalar value to ``dtype``.

    Parameters
    ----------
    dtype : type
        Target element type.
    value : Any
        Value to cast.

    Returns
    -------
    Any
        ``value`` converted to ``dtype``.

    Raises
    ------
    TypeError
        If the cast is not supported.

    Notes
    -----
    Casting follows Python scalar conversion rules rather than NumPy's
    fixed-width overflow and rounding semantics. For example, ``bool`` is
    not a subclass of ``int`` here, so explicit branches handle each
    conversion path.
    """
    if dtype is bool:
        if isinstance(value, bool):
            return value

        return bool(value)

    if dtype is int:
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        raise cannot_cast(value=value, target=int)

    if dtype is float:
        if isinstance(value, bool):
            return float(int(value))

        if isinstance(value, (int, float)):
            return float(value)

        raise cannot_cast(value=value, target=float)

    if dtype is complex:
        if isinstance(value, bool):
            return complex(int(value))

        if isinstance(value, (int, float)):
            return complex(value)

        if isinstance(value, complex):
            return value

        raise cannot_cast(value=value, target=complex)

    if dtype is str:
        if isinstance(value, str):
            return value

        return str(value)

    raise unsupported_dtype(dtype=dtype)


def can_cast(from_dtype: type[Any], to_dtype: type[Any]) -> bool:
    """
    Return whether values of ``from_dtype`` can be cast to ``to_dtype``.

    Parameters
    ----------
    from_dtype : type
        Source element type.
    to_dtype : type
        Target element type.

    Returns
    -------
    bool
        ``True`` if the cast is allowed by the promotion hierarchy.

    Notes
    -----
    NumPy's :func:`numpy.can_cast` also considers casting rules between
    fixed-width dtypes and the ``casting`` mode. This helper implements a
    simplified hierarchy over the five supported scalar types only.
    """
    if from_dtype is to_dtype:
        return True

    if to_dtype is bool:
        return from_dtype is bool

    if to_dtype is int:
        return from_dtype in (bool, int)

    if to_dtype is float:
        return from_dtype in (bool, int, float)

    if to_dtype is complex:
        return from_dtype in (bool, int, float, complex)

    if to_dtype is str:
        # Any supported dtype can be stringified.
        return True

    return False


def default_fill_value(dtype: type[Any]) -> Any:
    """
    Return the default fill value for ``dtype``.

    Parameters
    ----------
    dtype : type
        Element type.

    Returns
    -------
    Any
        ``False`` for ``bool``, ``0`` for ``int``, ``0.0`` for ``float``,
        ``0j`` for ``complex``, and ``""`` for ``str``.

    Raises
    ------
    TypeError
        If ``dtype`` is not supported.
    """
    if dtype is bool:
        return False

    if dtype is int:
        return 0

    if dtype is float:
        return 0.0

    if dtype is complex:
        return 0j

    if dtype is str:
        return ""

    raise unsupported_dtype(dtype=dtype)


def promote_dtypes(*dtypes: type[Any]) -> type[Any]:
    """
    Return the common result dtype for a set of dtypes.

    Parameters
    ----------
    *dtypes : type
        Two or more dtypes to promote.

    Returns
    -------
    type
        The promoted dtype following the hierarchy ``bool`` < ``int`` <
        ``float`` < ``complex``.

    Raises
    ------
    TypeError
        If the dtypes cannot be promoted together (for example ``int`` and
        ``str``).

    Notes
    -----
    NumPy uses a richer type hierarchy with many fixed-width dtypes. This
    function only considers the five supported Python scalar types.
    """
    unique = set(dtypes)

    if len(unique) == 1:
        return next(iter(unique))

    # bool mixed with int promotes to int.
    if unique <= {bool, int}:
        return int

    # Any complex operand forces a complex result.
    if complex in unique and unique <= NUMERIC_DTYPES:
        return complex

    # Remaining numeric mixes promote to float.
    if unique <= {bool, int, float}:
        return float

    raise mixed_dtypes(dtypes=unique)


def infer_dtype_from_value(value: Any) -> type[Any]:
    """
    Infer a supported dtype from a single Python scalar.

    Parameters
    ----------
    value : Any
        Scalar value whose type will be mapped to a supported dtype.

    Returns
    -------
    type
        ``bool``, ``int``, ``float``, ``complex``, or ``str``.

    Raises
    ------
    TypeError
        If ``value`` is not one of the supported scalar types.

    Notes
    -----
    Because ``bool`` is a subclass of ``int`` in Python, ``bool`` values are
    detected before ``int``. NumPy distinguishes ``numpy.bool_`` from
    integer dtypes at the type level.
    """
    if isinstance(value, bool):
        return bool

    # bool is a subclass of int; check it first.
    if isinstance(value, int):
        return int

    if isinstance(value, float):
        return float

    if isinstance(value, complex):
        return complex

    if isinstance(value, str):
        return str

    raise unsupported_element_type(value=value)


def infer_dtype_from_values(values: Sequence[Any]) -> type[Any]:
    """
    Infer a common dtype from a sequence of values.

    Parameters
    ----------
    values : sequence
        Non-empty sequence of scalar values.

    Returns
    -------
    type
        Promoted dtype of all elements, or ``float`` when ``values`` is
        empty.

    Notes
    -----
    An empty sequence defaults to ``float``, matching the convention used by
    :func:`numpy.creation.array` for empty inputs in this package.
    """
    if not values:
        # Match array() convention for empty inputs.
        return float

    return promote_dtypes(*(infer_dtype_from_value(value) for value in values))


def cast_sequence(
    values: Iterable[Any],
    target: type[Any],
) -> list[Any]:
    """
    Cast each element of ``values`` to ``target``.

    Parameters
    ----------
    values : iterable
        Values to cast.
    target : type
        Target element type.

    Returns
    -------
    list
        New list of cast values.
    """
    return [cast_value(target, value) for value in values]


def result_dtype_for_arithmetic(
    left: type[Any],
    right: type[Any],
) -> type[Any]:
    """
    Return the result dtype for binary arithmetic on ``left`` and ``right``.

    Parameters
    ----------
    left : type
        Dtype of the left operand.
    right : type
        Dtype of the right operand.

    Returns
    -------
    type
        Promoted dtype of the operands.
    """
    return promote_dtypes(left, right)


def result_dtype_for_truediv(
    left: type[Any],
    right: type[Any],
) -> type[Any]:
    """
    Return the result dtype for true division.

    Parameters
    ----------
    left : type
        Dtype of the left operand.
    right : type
        Dtype of the right operand.

    Returns
    -------
    type
        ``float`` for real operands; ``complex`` when either operand is
        complex.

    Notes
    -----
    True division always yields a floating-point result for real inputs,
    matching NumPy's behavior for integer operands.
    """
    promoted = promote_dtypes(left, right)

    if promoted is complex:
        return complex

    return float


def result_dtype_for_comparison(
    left: type[Any],
    right: type[Any],
) -> type[Any]:
    """
    Return the result dtype for a comparison operation.

    Parameters
    ----------
    left : type
        Dtype of the left operand.
    right : type
        Dtype of the right operand.

    Returns
    -------
    type
        Always ``bool`` after validating that the operands are comparable.

    Raises
    ------
    TypeError
        If the operand dtypes cannot be promoted.
    """
    promote_dtypes(left, right)
    return bool


def result_dtype_for_logical(
    left: type[Any],
    right: type[Any],
) -> type[Any]:
    """
    Return the result dtype for a logical binary operation.

    Parameters
    ----------
    left : type
        Dtype of the left operand.
    right : type
        Dtype of the right operand.

    Returns
    -------
    type
        Always ``bool`` after validating that the operands are logical.

    Raises
    ------
    TypeError
        If either operand dtype is not logical.
    """
    promote_dtypes(left, right)
    return bool


def result_dtype_for_mean(dtype: type[Any]) -> type[Any]:
    """
    Return the result dtype of :func:`numpy.mean` for ``dtype``.

    Parameters
    ----------
    dtype : type
        Input array dtype.

    Returns
    -------
    type
        ``float`` for real dtypes; ``complex`` when the input is complex.
    """
    if dtype is complex:
        return complex

    return float


# String aliases accepted by :func:`as_dtype` (e.g. ``dtype="float"``).
_DTYPE_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
}
