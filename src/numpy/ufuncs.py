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

"""Universal functions for element-wise array math.

Factory-built ufuncs (``add``, ``exp``, ``sqrt``, logical ops, etc.) delegate
to :class:`ndarray` binary and unary operation helpers. Complex-aware math
ufuncs dispatch separately for real and complex dtypes.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Callable

from ._dtype import (
    is_complex_dtype,
    require_logical_dtype,
    require_numeric_dtype,
    result_dtype_for_absolute,
)
from ._errors import where_must_be_bool
from ._float_ops import (
    floor_divide as _floor_divide,
)
from ._float_ops import (
    numeric_power as _numeric_power,
)
from ._float_ops import (
    real_ceil,
    real_floor,
    real_log,
    real_round,
    real_sqrt,
)
from ._float_ops import (
    remainder as _remainder,
)
from ._float_ops import (
    true_divide as _true_divide,
)
from ._types import is_ndarray
from ._validation import coerce_to_ndarray
from .ndarray import ndarray


def conjugate(a: ndarray) -> ndarray:
    """
    Return the complex conjugate, element-wise.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        The complex conjugate of `a`, element-wise.

    Notes
    -----
    Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex``
    dtypes are supported. Object arrays are not supported.
    """
    a = coerce_to_ndarray(a)
    return a.conjugate()


conj = conjugate
conj.__doc__ = conjugate.__doc__


def real(a: ndarray) -> ndarray:
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        The real part of `a`. For non-complex inputs, values are cast to
        ``float``.

    Notes
    -----
    Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex``
    dtypes are supported. Object arrays are not supported.
    """
    a = coerce_to_ndarray(a)
    return a.real


def imag(a: ndarray) -> ndarray:
    """
    Return the imaginary part of the complex argument.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        The imaginary part of `a`. For non-complex inputs, an array of
        zeros with ``float`` dtype is returned.

    Notes
    -----
    Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex``
    dtypes are supported. Object arrays are not supported.
    """
    a = coerce_to_ndarray(a)
    return a.imag


def _make_binary_ufunc(
    op: Callable[[Any, Any], Any],
    *,
    kind: str = "arithmetic",
) -> Callable[..., ndarray]:
    """Build a binary ufunc with optional ``out`` and ``where`` support.

    Factory used at module level to create ufuncs such as :func:`add`,
    :func:`divide`, and :func:`logical_and`. The returned callable
    coerces inputs and delegates to :meth:`ndarray._binary_op`.

    Parameters
    ----------
    op : callable
        Binary scalar function applied element-wise.
    kind : str, optional
        Operation category forwarded to :meth:`ndarray._binary_op`.
        Default is ``'arithmetic'``.

    Returns
    -------
    callable
        Ufunc accepting ``(a, b, out=None, where=True)``.

    Notes
    -----
    Unlike NumPy ufuncs, there is no casting, signature, or extobj
    machinery. ``where`` must be a ``bool`` or ``ndarray``; other types
    raise immediately.
    """

    def ufunc(
        a: ndarray,
        b: Any,
        out: ndarray | None = None,
        where: bool | ndarray = True,
    ) -> ndarray:
        a = coerce_to_ndarray(a)

        if out is not None:
            out = coerce_to_ndarray(out, name="out")

        if not isinstance(where, bool) and not is_ndarray(where):
            raise where_must_be_bool(got=where)

        return a._binary_op(
            b,
            op,
            out=out,
            where=where,
            kind=kind,
        )

    return ufunc


def _make_unary_ufunc(
    op: Callable[[Any], Any],
    *,
    result_dtype: type[Any] | None = None,
    operation: str = "ufunc",
    require_logical: bool = False,
    require_numeric: bool = False,
) -> Callable[[ndarray], ndarray]:
    """Build a unary ufunc with dtype validation and optional result dtype.

    Factory for ufuncs such as :func:`square`, :func:`floor`, and
    :func:`logical_not`. Coerces the input and delegates to
    :meth:`ndarray._unary_op`.

    Parameters
    ----------
    op : callable
        Unary scalar function.
    result_dtype : type, optional
        Explicit output dtype passed to :meth:`ndarray._unary_op`.
    operation : str, optional
        Label for dtype validation error messages.
    require_logical : bool, optional
        Require boolean input dtype.
    require_numeric : bool, optional
        Require numeric input dtype.

    Returns
    -------
    callable
        Ufunc accepting a single array argument.

    Notes
    -----
    Unlike NumPy ufuncs, there is no ``out`` or ``where`` parameter.
    """

    def ufunc(a: ndarray) -> ndarray:
        a = coerce_to_ndarray(a)
        return a._unary_op(
            op,
            result_dtype=result_dtype,
            operation=operation,
            require_logical=require_logical,
            require_numeric=require_numeric,
        )

    return ufunc


def _make_math_ufunc(
    real_op: Callable[[Any], Any],
    complex_op: Callable[[Any], Any],
    *,
    operation: str,
) -> Callable[[ndarray], ndarray]:
    """Build a unary math ufunc that dispatches on real vs complex dtype.

    Factory for transcendental ufuncs such as :func:`sqrt`, :func:`exp`,
    and :func:`sin`. Selects ``real_op`` for real dtypes (promoting the
    result to ``float``) and ``complex_op`` for complex dtypes
    (result ``complex``).

    Parameters
    ----------
    real_op : callable
        Scalar function for real-valued inputs.
    complex_op : callable
        Scalar function for complex-valued inputs (typically from
        ``cmath``).
    operation : str
        Label for dtype validation error messages.

    Returns
    -------
    callable
        Ufunc accepting a single array argument.

    Notes
    -----
    Unlike NumPy, integer inputs are accepted but the result is always
    ``float`` for real dtypes. There is no ``dtype`` keyword to override
    the result type.
    """

    def ufunc(a: ndarray) -> ndarray:
        a = coerce_to_ndarray(a)
        require_numeric_dtype(a.dtype, operation=operation)

        if is_complex_dtype(a.dtype):
            return a._unary_op(
                complex_op,
                result_dtype=complex,
                operation=operation,
                require_numeric=True,
            )

        return a._unary_op(
            real_op,
            result_dtype=float,
            operation=operation,
            require_numeric=True,
        )

    return ufunc


def _absolute(a: ndarray) -> ndarray:
    """Element-wise absolute value with dtype-aware result promotion.

    Implementation backing :func:`absolute`. Uses
    :func:`result_dtype_for_absolute` so complex inputs produce a
    floating magnitude while real inputs preserve sign information in
    the promoted dtype.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        Absolute value of each element.

    Notes
    -----
    Also assigned to the public name :data:`absolute`. Unlike NumPy,
    there is no ``out`` parameter.
    """
    a = coerce_to_ndarray(a)
    return a._unary_op(
        abs,
        result_dtype=result_dtype_for_absolute(a.dtype),
        operation="absolute",
        require_numeric=True,
    )


add = _make_binary_ufunc(lambda x, y: x + y)
add.__doc__ = """
Add arguments element-wise.

Parameters
----------
x1 : array_like
    First input array.
x2 : array_like or scalar
    Second input array or scalar. Scalars are broadcast to the shape of
    ``x1``.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Where True, the ufunc result
    is stored; elsewhere, the ``out`` array retains its original value
    (or the result is left undefined when ``out`` is None). Default is
    True.

Returns
-------
ndarray
    The sum of ``x1`` and ``x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported for arithmetic. Object arrays, structured dtypes, and
string operands are not supported for this operation.

Examples
--------
>>> import numpy as np
>>> np.add([1, 2], [3, 4])
array([4, 6])
>>> np.add(1, [2, 3])
array([3, 4])
"""

subtract = _make_binary_ufunc(lambda x, y: x - y)
subtract.__doc__ = """
Subtract arguments element-wise.

Parameters
----------
x1 : array_like
    First input array (minuend).
x2 : array_like or scalar
    Second input array or scalar (subtrahend).
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    The difference of ``x1`` and ``x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported for arithmetic. Object arrays are not supported.
"""

multiply = _make_binary_ufunc(lambda x, y: x * y)
multiply.__doc__ = """
Multiply arguments element-wise.

Parameters
----------
x1 : array_like
    First input array.
x2 : array_like or scalar
    Second input array or scalar.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    The product of ``x1`` and ``x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported for arithmetic. Object arrays are not supported.
"""

divide = _make_binary_ufunc(
    _true_divide,
    kind="truediv",
)
divide.__doc__ = """
Divide arguments element-wise.

Parameters
----------
x1 : array_like
    Dividend array.
x2 : array_like or scalar
    Divisor array or scalar.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    The quotient ``x1 / x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Integer division promotes to ``float`` or ``complex`` as
needed. Object arrays are not supported.
"""

floor_divide = _make_binary_ufunc(_floor_divide)
floor_divide.__doc__ = """
Return the largest integer smaller or equal to the division of the inputs.

Parameters
----------
x1 : array_like
    Dividend array.
x2 : array_like or scalar
    Divisor array or scalar.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    The floor of the quotient ``x1 // x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

mod = _make_binary_ufunc(_remainder)
mod.__doc__ = """
Return element-wise remainder of division.

Parameters
----------
x1 : array_like
    Dividend array.
x2 : array_like or scalar
    Divisor array or scalar.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    The remainder ``x1 % x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

power = _make_binary_ufunc(_numeric_power)
power.__doc__ = """
First array elements raised to powers from second array, element-wise.

Parameters
----------
x1 : array_like
    Base array.
x2 : array_like or scalar
    Exponent array or scalar.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    ``x1`` raised to the power ``x2``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

absolute = _absolute
absolute.__doc__ = """
Calculate the absolute value element-wise.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    Absolute value of ``x``. For complex inputs, returns the magnitude.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

square = _make_unary_ufunc(
    lambda x: x * x,
    operation="square",
    require_numeric=True,
)
square.__doc__ = """
Return the element-wise square of the input.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    ``x * x``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

sqrt = _make_math_ufunc(real_sqrt, cmath.sqrt, operation="sqrt")
sqrt.__doc__ = """
Return the non-negative square-root of an array, element-wise.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    The square root of ``x``. For complex inputs, the principal value
    is returned.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

exp = _make_math_ufunc(math.exp, cmath.exp, operation="exp")
exp.__doc__ = """
Calculate the exponential of all elements in the input array.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    The exponential of ``x``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

log = _make_math_ufunc(real_log, cmath.log, operation="log")
log.__doc__ = """
Natural logarithm, element-wise.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    The natural logarithm of ``x``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

sin = _make_math_ufunc(math.sin, cmath.sin, operation="sin")
sin.__doc__ = """
Trigonometric sine, element-wise.

Parameters
----------
x : array_like
    Angle in radians.

Returns
-------
ndarray
    The sine of ``x``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

cos = _make_math_ufunc(math.cos, cmath.cos, operation="cos")
cos.__doc__ = """
Trigonometric cosine, element-wise.

Parameters
----------
x : array_like
    Angle in radians.

Returns
-------
ndarray
    The cosine of ``x``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

tan = _make_math_ufunc(math.tan, cmath.tan, operation="tan")
tan.__doc__ = """
Compute tangent element-wise.

Parameters
----------
x : array_like
    Angle in radians.

Returns
-------
ndarray
    The tangent of ``x``, element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

floor = _make_unary_ufunc(
    real_floor,
    operation="floor",
    require_numeric=True,
)
floor.__doc__ = """
Return the floor of the input, element-wise.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    The floor of ``x``. For complex inputs, the floor is applied
    independently to real and imaginary parts.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

ceil = _make_unary_ufunc(
    real_ceil,
    operation="ceil",
    require_numeric=True,
)
ceil.__doc__ = """
Return the ceiling of the input, element-wise.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    The ceiling of ``x``. For complex inputs, the ceiling is applied
    independently to real and imaginary parts.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

logical_not = _make_unary_ufunc(
    lambda x: not bool(x),
    result_dtype=bool,
    operation="logical_not",
    require_logical=True,
)
logical_not.__doc__ = """
Compute the truth value of NOT ``x`` element-wise.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    Boolean array with the logical NOT of ``x``.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are accepted as logical operands. Object arrays are not supported.
"""

isnan = _make_unary_ufunc(
    lambda x: cmath.isnan(x) if isinstance(x, complex) else math.isnan(x),
    result_dtype=bool,
    operation="isnan",
    require_numeric=True,
)
isnan.__doc__ = """
Test element-wise for NaN and return result as a boolean array.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    Boolean array that is True where ``x`` is NaN.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

isinf = _make_unary_ufunc(
    lambda x: cmath.isinf(x) if isinstance(x, complex) else math.isinf(x),
    result_dtype=bool,
    operation="isinf",
    require_numeric=True,
)
isinf.__doc__ = """
Test element-wise for positive or negative infinity.

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    Boolean array that is True where ``x`` is positive or negative
    infinity.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""

isfinite = _make_unary_ufunc(
    lambda x: cmath.isfinite(x) if isinstance(x, complex) else math.isfinite(x),
    result_dtype=bool,
    operation="isfinite",
    require_numeric=True,
)
isfinite.__doc__ = """
Test element-wise for finiteness (not NaN and not infinity).

Parameters
----------
x : array_like
    Input array.

Returns
-------
ndarray
    Boolean array that is True where ``x`` is finite.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. Object arrays are not supported.
"""


_py_round = round
round = _make_unary_ufunc(
    lambda value: (
        complex(_py_round(value.real), _py_round(value.imag))
        if isinstance(value, complex)
        else real_round(value)
    ),
    operation="round",
    require_numeric=True,
)
round.__doc__ = """
Evenly round to the given number of decimals.

Parameters
----------
a : array_like
    Input array.

Returns
-------
ndarray
    Rounded values. For complex inputs, real and imaginary parts are
    rounded independently.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are supported. The ``decimals`` and ``out`` keyword arguments are not
supported. Object arrays are not supported.
"""

logical_and = _make_binary_ufunc(
    lambda x, y: bool(x) and bool(y),
    kind="logical",
)
logical_and.__doc__ = """
Compute the truth value of ``x1 AND x2`` element-wise.

Parameters
----------
x1, x2 : array_like
    Input arrays. Broadcast together.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    Boolean result of the logical AND operation applied element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are accepted as logical operands. Object arrays are not supported.
"""

logical_or = _make_binary_ufunc(
    lambda x, y: bool(x) or bool(y),
    kind="logical",
)
logical_or.__doc__ = """
Compute the truth value of ``x1 OR x2`` element-wise.

Parameters
----------
x1, x2 : array_like
    Input arrays. Broadcast together.
out : ndarray, optional
    A location into which the result is stored. Default is None.
where : bool or array_like of bool, optional
    Condition broadcast over the inputs. Default is True.

Returns
-------
ndarray
    Boolean result of the logical OR operation applied element-wise.

Notes
-----
Unlike NumPy, only ``bool``, ``int``, ``float``, and ``complex`` dtypes
are accepted as logical operands. Object arrays are not supported.
"""


from ._complex_ops import angle  # noqa: E402
