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

"""Coerce and validate user inputs before array operations.

Converts array-like values to :class:`ndarray`, rejects unsupported scalars
where required, and normalizes shape tuples for broadcasting helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from ._errors import (
    must_be_ndarray_or_array_like,
    must_be_number,
    scalar_operands_not_supported,
)
from ._types import is_ndarray

if TYPE_CHECKING:
    from .ndarray import ndarray

BASIC_ARRAY_LIKE_TYPES = (list, tuple, range)


def is_basic_array_like(value: Any) -> bool:
    """
    Return whether ``value`` is an ndarray or a supported nested sequence.

    Internal helper used before coercion to decide whether a value can be
    converted with :func:`numpy.array`.

    Parameters
    ----------
    value : any
        Value to inspect.

    Returns
    -------
    bool
        ``True`` if ``value`` is an :class:`ndarray` or an instance of
        ``list``, ``tuple``, or ``range``.

    Notes
    -----
    NumPy accepts a wider range of array-likes (including scalars with
    ``ndmin``, buffer objects, and duck arrays). This helper only covers
    the nested-sequence types supported by this implementation.
    """
    return is_ndarray(value) or isinstance(value, BASIC_ARRAY_LIKE_TYPES)


def coerce_to_ndarray(
    value: Any,
    *,
    name: str = "array",
    reject_scalars: bool = False,
) -> ndarray:
    """
    Convert a value to an :class:`ndarray` or raise a descriptive error.

    Internal helper used at the boundary of most public functions that
    accept array-like operands.

    Parameters
    ----------
    value : any
        Value to convert. Existing ndarrays are returned unchanged.
    name : str (default: "array")
        Parameter name included in error messages.
    reject_scalars : bool (default: False)
        If ``True``, reject Python scalars instead of treating them as
        invalid array-likes.

    Returns
    -------
    ndarray
        ``value`` unchanged when it is already an ndarray, otherwise a
        newly constructed array from nested sequences.

    Raises
    ------
    TypeError
        If ``value`` is not an ndarray or supported array-like, or if
        ``reject_scalars`` is ``True`` and ``value`` is a scalar.

    Notes
    -----
    Unlike NumPy's ``np.asarray``, this helper eagerly wraps ``list``,
    ``tuple``, and ``range`` via :func:`numpy.array` and does not accept
    arbitrary duck-array protocols.
    """
    if is_ndarray(value):
        return value

    if isinstance(value, BASIC_ARRAY_LIKE_TYPES):
        from .creation import array

        return array(value)

    if reject_scalars:
        raise scalar_operands_not_supported(name=name)

    raise must_be_ndarray_or_array_like(name=name)


def coerce_to_ndarray_iterable(
    values: Iterable[Any],
    *,
    name: str = "arrays",
) -> tuple[ndarray, ...]:
    """
    Convert an iterable of values to a tuple of ndarrays.

    Internal helper used by :func:`numpy.manipulation.concatenate` and
    :func:`numpy.manipulation.stack`.

    Parameters
    ----------
    values : iterable of any
        Operands to convert.
    name : str (default: "arrays")
        Base parameter name used in per-element error messages.

    Returns
    -------
    tuple of ndarray
        Coerced arrays in iteration order.

    Raises
    ------
    TypeError
        If any element cannot be converted to an ndarray.
    """
    arrays = tuple(values)

    return tuple(
        coerce_to_ndarray(value, name=f"{name}[{index}]")
        for index, value in enumerate(arrays)
    )


def require_number(
    value: Any,
    *,
    name: str,
) -> int | float | complex:
    """
    Require that ``value`` is a numeric scalar.

    Internal helper used by creation routines such as :func:`numpy.arange`
    and :func:`numpy.linspace`.

    Parameters
    ----------
    value : any
        Scalar to validate.
    name : str
        Parameter name included in error messages.

    Returns
    -------
    int, float, or complex
        ``value`` when it is a valid numeric scalar.

    Raises
    ------
    TypeError
        If ``value`` is a boolean or not an ``int``, ``float``, or
        ``complex`` instance.

    Notes
    -----
    Python ``bool`` is a subclass of ``int`` and is explicitly rejected,
    matching NumPy's scalar typing rules.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise must_be_number(name=name)

    return value


def require_shape_tuple(
    shape: Any,
    *,
    name: str = "shape",
) -> tuple[int, ...]:
    """
    Validate and normalize a shape argument.

    Internal helper used by broadcasting and array-creation functions.

    Parameters
    ----------
    shape : int or tuple of int
        Desired output shape.
    name : str (default: "shape")
        Parameter name included in error messages. Currently unused but
        reserved for future error reporting.

    Returns
    -------
    tuple of int
        Normalized shape with non-negative integer dimensions.

    Raises
    ------
    TypeError
        If ``shape`` is not an integer or tuple of integers.
    ValueError
        If any dimension is negative.

    See Also
    --------
    normalize_shape : Lower-level shape normalization in ``_shape``.
    """
    from ._shape import normalize_shape

    return normalize_shape(shape)
