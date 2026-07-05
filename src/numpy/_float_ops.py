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

"""Scalar floating-point kernels for division and ordering.

Pure-Python implementations of true divide, floor divide, remainder, real
math helpers, and sort keys with IEEE-style special-case handling.
"""

from __future__ import annotations

import cmath
import math
from typing import Any


def true_divide(left: Any, right: Any) -> Any:
    """Divide two scalars with NumPy-like floating-point semantics.

    Parameters
    ----------
    left : scalar
        Dividend.
    right : scalar
        Divisor.

    Returns
    -------
    scalar
        Quotient. ``0/0`` yields NaN; finite/non-zero divided by zero
        yields a signed infinity.

    Notes
    -----
    Used as the scalar kernel for element-wise true division. Complex
    ``0/0`` returns ``complex(nan, nan)``.
    """
    if _is_zero(right):
        # 0/0 -> NaN; nonzero/0 -> signed infinity.
        if _is_zero(left):
            if isinstance(left, complex) or isinstance(right, complex):
                return complex(float("nan"), float("nan"))

            return float("nan")

        return _signed_infinity(left)

    return left / right


def floor_divide(left: Any, right: Any) -> Any:
    """Floor-divide two scalars with NumPy-like semantics.

    Parameters
    ----------
    left : scalar
        Dividend.
    right : scalar
        Divisor.

    Returns
    -------
    scalar
        Floor of the quotient. Integer ``0//0`` returns ``0``; float
        ``0.0//0.0`` returns NaN.

    Notes
    -----
    Used as the scalar kernel for element-wise floor division.
    """
    if _is_zero(right):
        if _is_zero(left):
            if isinstance(left, float):
                return float("nan")

            return 0

        return _signed_infinity(left)

    try:
        return left // right
    except ZeroDivisionError:
        return _signed_infinity(left)


def remainder(left: Any, right: Any) -> Any:
    """Return the remainder of scalar division with NumPy-like semantics.

    Parameters
    ----------
    left : scalar
        Dividend.
    right : scalar
        Divisor.

    Returns
    -------
    scalar
        ``left % right``. Division by zero yields NaN.

    Notes
    -----
    Used as the scalar kernel for element-wise remainder (modulo).
    """
    if _is_zero(right):
        if isinstance(left, complex) or isinstance(right, complex):
            return complex(float("nan"), float("nan"))

        return float("nan")

    try:
        return left % right
    except ZeroDivisionError:
        return float("nan")


def real_sqrt(value: Any) -> Any:
    """Square root for real scalar inputs.

    Parameters
    ----------
    value : scalar
        Input value.

    Returns
    -------
    scalar
        ``math.sqrt(value)`` for valid inputs. Negative finite floats and
        NaN inputs return NaN.

    Notes
    -----
    Unlike NumPy's ``sqrt``, negative reals do not produce complex results.
    Non-float inputs use Python's built-in square root without the negative
    check.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return float("nan")

        if value < 0:
            return float("nan")

    return math.sqrt(value)


def real_log(value: Any) -> Any:
    """Natural logarithm for real scalar inputs.

    Parameters
    ----------
    value : scalar
        Input value.

    Returns
    -------
    scalar
        ``math.log(value)`` for positive inputs. NaN and negative floats
        return NaN; ``log(0)`` returns ``-inf``.

    Notes
    -----
    Unlike NumPy's ``log``, negative reals do not produce complex results.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return float("nan")

        if value < 0:
            return float("nan")

        if value == 0.0:
            return float("-inf")

    return math.log(value)


def real_floor(value: Any) -> Any:
    """Floor of a scalar, preserving NaN and infinity.

    Parameters
    ----------
    value : scalar
        Input value.

    Returns
    -------
    scalar
        ``math.floor(value)``, except NaN and infinity are returned unchanged
        for float inputs.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value

    return math.floor(value)


def real_ceil(value: Any) -> Any:
    """Ceiling of a scalar, preserving NaN and infinity.

    Parameters
    ----------
    value : scalar
        Input value.

    Returns
    -------
    scalar
        ``math.ceil(value)``, except NaN and infinity are returned unchanged
        for float inputs.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value

    return math.ceil(value)


def real_round(value: Any) -> Any:
    """Round a scalar to the nearest integer.

    Parameters
    ----------
    value : scalar
        Input value.

    Returns
    -------
    scalar
        ``round(value)``. NaN float inputs return NaN.
    """
    if isinstance(value, float) and math.isnan(value):
        return float("nan")

    return round(value)


def numeric_power(left: Any, right: Any) -> Any:
    """Exponentiation with NumPy-like error and dtype semantics.

    Parameters
    ----------
    left : scalar
        Base.
    right : scalar
        Exponent.

    Returns
    -------
    scalar
        ``left ** right``. Invalid operations return NaN. If the mathematical
        result is complex but neither operand is complex, NaN is returned.

    Notes
    -----
    Used as the scalar kernel for element-wise power. This blocks silent
    promotion to complex for real operands.
    """
    try:
        result = left**right
    except (ValueError, ZeroDivisionError, OverflowError):
        if isinstance(left, complex) or isinstance(right, complex):
            return complex(float("nan"), float("nan"))

        return float("nan")

    if isinstance(result, complex) and not isinstance(left, complex):
        # Real operands must not silently produce complex results.
        return float("nan")

    return result


def order_key(value: Any) -> tuple[Any, ...]:
    """Return a sort key with NaNs last and infinities before finite values.

    Parameters
    ----------
    value : scalar
        Element to rank.

    Returns
    -------
    tuple
        Comparable key used by :func:`sort`, :meth:`ndarray.sort`, and
        :func:`unique`.

    Notes
    -----
    NaNs sort after all finite values. Positive infinity sorts before
    negative infinity among non-NaN values. Complex numbers are ordered by
    ``(real, imag)`` after the NaN/infinity prefix.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return (1, 0.0)

        if math.isinf(value):
            return (0, 1 if value > 0 else -1)

        return (0, 0, value)

    if isinstance(value, complex):
        if cmath.isnan(value):
            return (1, 0.0, 0.0)

        if cmath.isinf(value):
            return (0, 1 if value.real >= 0 else -1, 0.0, 0.0)

        return (0, 0, value.real, value.imag)

    return (0, 0, value)


def _is_zero(value: Any) -> bool:
    """Return whether a scalar divisor should be treated as zero.

    Used by :func:`true_divide`, :func:`floor_divide`, and :func:`remainder`
    before applying IEEE-style division rules. Matches both ``0`` and
    ``0.0``; does not treat very small floats as zero.

    Parameters
    ----------
    value : Any
        Scalar divisor or dividend.

    Returns
    -------
    bool
        ``True`` when ``value == 0`` or ``value == 0.0``.
    """
    return value == 0 or value == 0.0


def _signed_infinity(value: Any) -> Any:
    """Return a signed infinity (or NaN) derived from the dividend.

    Used when division by zero occurs in :func:`true_divide` and
    :func:`floor_divide` after :func:`_is_zero` detects a zero divisor.
    Preserves the sign of the dividend for real values; for complex,
    maps the first quadrant to ``(+inf, +inf)`` and other directions
    to ``(-inf, -inf)``.

    Parameters
    ----------
    value : Any
        Dividend scalar (the numerator in ``value / 0``).

    Returns
    -------
    float or complex
        ``float('inf')``, ``float('-inf')``, ``float('nan')``, or the
        complex analog with both components set to the same signed infinity
        or NaN.

    Notes
    -----
        NaN dividends propagate to NaN. Complex ``0+0j`` maps to complex
        NaN rather than infinity, matching common NumPy scalar behavior.
    """
    if isinstance(value, float) and math.isnan(value):
        return float("nan")

    if isinstance(value, complex) and cmath.isnan(value):
        return complex(float("nan"), float("nan"))

    if isinstance(value, complex):
        if value.real == 0 and value.imag == 0:
            return complex(float("nan"), float("nan"))

        if value.real > 0 or (value.real == 0 and value.imag > 0):
            return complex(float("inf"), float("inf"))

        return complex(float("-inf"), float("-inf"))

    if value > 0:
        return float("inf")

    if value < 0:
        return float("-inf")

    return float("nan")
