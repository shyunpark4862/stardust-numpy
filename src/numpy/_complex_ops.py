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

"""Complex-number attributes and unary operations for ``ndarray``.

Provides :class:`ComplexOpsMixin` (``conjugate``, ``real``, ``imag``) and the
module-level :func:`angle` helper used by ufuncs.
"""

from __future__ import annotations

import cmath
from typing import TYPE_CHECKING, Any

from ._dtype import is_complex_dtype, require_numeric_dtype
from ._factory import contiguous_ndarray

if TYPE_CHECKING:
    from .ndarray import ndarray


class ComplexOpsMixin:
    """Mixin providing complex-number views and conjugation for ``ndarray``."""

    def conjugate(self) -> ndarray:
        """Return the complex conjugate, element-wise.

        Returns
        -------
        ndarray
            Conjugate of each element. Real-valued inputs are returned
            unchanged.

        Raises
        ------
        TypeError
            If the array dtype is not numeric.

        Notes
        -----
        Unlike NumPy, only the five supported numeric dtypes are allowed.
        There is no ``out`` parameter.

        See Also
        --------
        real : Real part of each element.
        imag : Imaginary part of each element.
        """
        require_numeric_dtype(self.dtype, operation="conjugate")

        return self._unary_op(
            _conjugate_value,
            operation="conjugate",
            require_numeric=True,
        )

    conj = conjugate

    @property
    def real(self) -> ndarray:
        """Real part of each element.

        Returns
        -------
        ndarray
            Array of ``float`` values. For non-complex inputs, values are
            cast to ``float``.

        Raises
        ------
        TypeError
            If the array dtype is not numeric.

        Notes
        -----
        Unlike NumPy, the result dtype is always ``float`` (not the input
        floating type). There is no ``out`` parameter.
        """
        require_numeric_dtype(self.dtype, operation="real")

        if is_complex_dtype(self.dtype):
            return self._unary_op(
                lambda value: value.real,
                result_dtype=float,
                operation="real",
                require_numeric=True,
            )

        return self.astype(float)

    @property
    def imag(self) -> ndarray:
        """Imaginary part of each element.

        Returns
        -------
        ndarray
            Array of ``float`` values. Non-complex inputs yield zeros.

        Raises
        ------
        TypeError
            If the array dtype is not numeric.

        Notes
        -----
        For real-valued arrays, an array of zeros with ``float`` dtype is
        returned rather than an integer array.
        """
        require_numeric_dtype(self.dtype, operation="imag")

        if is_complex_dtype(self.dtype):
            return self._unary_op(
                lambda value: value.imag,
                result_dtype=float,
                operation="imag",
                require_numeric=True,
            )

        # Real dtypes have zero imaginary part everywhere.
        return contiguous_ndarray(
            [0.0] * self.size,
            self.shape,
            dtype=float,
        )


def angle(a: ndarray) -> ndarray:
    """Return the angle of the complex argument, element-wise.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        Array of ``float`` angles in radians, in the interval ``[-pi, pi]``.

    Raises
    ------
    TypeError
        If `a` does not have a numeric dtype.

    Notes
    -----
    Implemented via :func:`cmath.phase`. Unlike NumPy's ``numpy.angle``,
    there is no ``deg`` keyword and the result dtype is always ``float``.
    """
    from ._validation import coerce_to_ndarray

    a = coerce_to_ndarray(a)
    require_numeric_dtype(a.dtype, operation="angle")

    return a._unary_op(
        cmath.phase,
        result_dtype=float,
        operation="angle",
        require_numeric=True,
    )


def _conjugate_value(value: Any) -> Any:
    """Apply complex conjugation to a single scalar element.

    Called by :meth:`ComplexOpsMixin.conjugate` via :meth:`_unary_op`.
    Real and integer values pass through unchanged because they have zero
    imaginary part in this implementation.

    Parameters
    ----------
    value : Any
        Scalar array element.

    Returns
    -------
    Any
        ``value.conjugate()`` for complex; otherwise ``value`` unchanged.
    """
    if isinstance(value, complex):
        return value.conjugate()

    return value
