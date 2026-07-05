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

"""Element-wise binary and unary operations for ``ndarray``.

:class:`ElementwiseOpsMixin` implements arithmetic, comparison, and logical
dunder operators by delegating to shared broadcasting and dtype promotion
helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ._dtype import (
    cast_value,
    infer_dtype_from_value,
    is_bool_dtype,
    promote_dtypes,
    require_comparable_dtypes,
    require_logical_dtype,
    require_numeric_dtype,
    result_dtype_for_absolute,
    result_dtype_for_arithmetic,
    result_dtype_for_comparison,
    result_dtype_for_logical,
    result_dtype_for_truediv,
)
from ._errors import (
    out_dtype_mismatch,
    out_shape_mismatch,
    where_must_be_bool,
)
from ._factory import contiguous_ndarray, placeholder_contiguous_ndarray
from ._float_ops import (
    floor_divide as _floor_divide,
)
from ._float_ops import (
    numeric_power as _numeric_power,
)
from ._float_ops import (
    remainder as _remainder,
)
from ._float_ops import (
    true_divide as _true_divide,
)
from ._iter import iter_indices
from ._types import is_ndarray
from ._validation import coerce_to_ndarray, is_basic_array_like
from .broadcasting import broadcast_shape

if TYPE_CHECKING:
    from .ndarray import ndarray


class ElementwiseOpsMixin:
    """Mixin providing element-wise arithmetic, comparison, and logical ops."""

    def __add__(
        self,
        other: Any,
    ) -> ndarray:
        """Add arguments, element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Sum of ``self`` and ``other``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators. Only ``bool``, ``int``, ``float``, ``complex``, and
        ``str`` dtypes are supported.
        """
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(
        self,
        other: Any,
    ) -> ndarray:
        """Subtract arguments, element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Difference ``self - other``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators. Only ``bool``, ``int``, ``float``, ``complex``, and
        ``str`` dtypes are supported.
        """
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(
        self,
        other: Any,
    ) -> ndarray:
        """Multiply arguments, element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Product of ``self`` and ``other``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators. Only ``bool``, ``int``, ``float``, ``complex``, and
        ``str`` dtypes are supported.
        """
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(
        self,
        other: Any,
    ) -> ndarray:
        """Divide arguments, element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Divisor.

        Returns
        -------
        ndarray
            Quotient of ``self`` and ``other``, with broadcasting. The
            result dtype follows NumPy-like promotion rules for true
            division.

        Notes
        -----
        Uses NumPy-like scalar semantics for ``0/0`` and division by
        zero. Unlike NumPy, ``out`` and ``where`` are not exposed on
        dunder operators.
        """
        return self._binary_op(
            other,
            _true_divide,
            kind="truediv",
        )

    def __floordiv__(
        self,
        other: Any,
    ) -> ndarray:
        """Floor-divide arguments, element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Divisor.

        Returns
        -------
        ndarray
            Floor of the quotient, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(other, _floor_divide)

    def __mod__(
        self,
        other: Any,
    ) -> ndarray:
        """Return the remainder of division, element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Divisor.

        Returns
        -------
        ndarray
            Remainder of ``self`` divided by ``other``, with
            broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(other, _remainder)

    def __pow__(
        self,
        other: Any,
    ) -> ndarray:
        """First array elements raised to powers from second array,
        element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Exponents.

        Returns
        -------
        ndarray
            ``self`` raised to the power ``other``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators. Negative exponents on integers follow Python
        semantics.
        """
        return self._binary_op(other, _numeric_power)

    def __neg__(self) -> ndarray:
        """Numerical negative, element-wise.

        Returns
        -------
        ndarray
            Negated array.

        Raises
        ------
        TypeError
            If the array dtype is not numeric.

        Notes
        -----
        Unlike NumPy, ``out`` is not supported.
        """
        return self._unary_op(
            lambda x: -x,
            operation="negation",
            require_numeric=True,
        )

    def __abs__(self) -> ndarray:
        """Calculate the absolute value, element-wise.

        Returns
        -------
        ndarray
            Absolute value of each element.

        Raises
        ------
        TypeError
            If the array dtype is not numeric.

        Notes
        -----
        Unlike NumPy, ``out`` is not supported.
        """
        return self._unary_op(
            abs,
            result_dtype=result_dtype_for_absolute(self.dtype),
            operation="absolute value",
            require_numeric=True,
        )

    def __radd__(
        self,
        other: Any,
    ) -> ndarray:
        """Add arguments, element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Left operand when ``other + self`` is evaluated.

        Returns
        -------
        ndarray
            Sum of ``other`` and ``self``, with broadcasting.

        Notes
        -----
        Equivalent to ``self + other``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self + other

    def __rsub__(
        self,
        other: Any,
    ) -> ndarray:
        """Subtract arguments, element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Minuend when ``other - self`` is evaluated.

        Returns
        -------
        ndarray
            Difference ``other - self``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(other, lambda a, b: b - a)

    def __rmul__(
        self,
        other: Any,
    ) -> ndarray:
        """Multiply arguments, element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Left operand when ``other * self`` is evaluated.

        Returns
        -------
        ndarray
            Product of ``other`` and ``self``, with broadcasting.

        Notes
        -----
        Equivalent to ``self * other``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self * other

    def __rtruediv__(
        self,
        other: Any,
    ) -> ndarray:
        """Divide arguments, element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Dividend when ``other / self`` is evaluated.

        Returns
        -------
        ndarray
            Quotient ``other / self``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(
            other,
            lambda a, b: _true_divide(b, a),
            kind="truediv",
        )

    def __rfloordiv__(
        self,
        other: Any,
    ) -> ndarray:
        """Floor-divide arguments, element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Dividend when ``other // self`` is evaluated.

        Returns
        -------
        ndarray
            Floor of ``other // self``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(other, lambda a, b: _floor_divide(b, a))

    def __rmod__(
        self,
        other: Any,
    ) -> ndarray:
        """Return the remainder of division, element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Dividend when ``other % self`` is evaluated.

        Returns
        -------
        ndarray
            Remainder of ``other`` divided by ``self``, with
            broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(other, lambda a, b: _remainder(b, a))

    def __rpow__(
        self,
        other: Any,
    ) -> ndarray:
        """Raise second argument to the power of first argument,
        element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Base when ``other ** self`` is evaluated.

        Returns
        -------
        ndarray
            ``other`` raised to the power ``self``, with broadcasting.

        Notes
        -----
        Unlike NumPy, ``out`` and ``where`` are not exposed on dunder
        operators.
        """
        return self._binary_op(other, lambda a, b: _numeric_power(b, a))

    def __eq__(
        self,
        other: Any,
    ) -> ndarray:
        """Return (self == other) element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array of comparison results, with broadcasting.

        Notes
        -----
        The result dtype is always ``bool``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: a == b,
            kind="comparison",
        )

    def __ne__(
        self,
        other: Any,
    ) -> ndarray:
        """Return (self != other) element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array of comparison results, with broadcasting.

        Notes
        -----
        The result dtype is always ``bool``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: a != b,
            kind="comparison",
        )

    def __lt__(
        self,
        other: Any,
    ) -> ndarray:
        """Return (self < other) element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array of comparison results, with broadcasting.

        Notes
        -----
        The result dtype is always ``bool``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: a < b,
            kind="comparison",
        )

    def __le__(
        self,
        other: Any,
    ) -> ndarray:
        """Return (self <= other) element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array of comparison results, with broadcasting.

        Notes
        -----
        The result dtype is always ``bool``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: a <= b,
            kind="comparison",
        )

    def __gt__(
        self,
        other: Any,
    ) -> ndarray:
        """Return (self > other) element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array of comparison results, with broadcasting.

        Notes
        -----
        The result dtype is always ``bool``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: a > b,
            kind="comparison",
        )

    def __ge__(
        self,
        other: Any,
    ) -> ndarray:
        """Return (self >= other) element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array of comparison results, with broadcasting.

        Notes
        -----
        The result dtype is always ``bool``. Unlike NumPy, ``out`` and
        ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: a >= b,
            kind="comparison",
        )

    def __and__(
        self,
        other: Any,
    ) -> ndarray:
        """Return the logical AND of two arrays element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array, with broadcasting.

        Notes
        -----
        Operands are evaluated via Python ``bool()``. Unlike NumPy,
        ``out`` and ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: bool(a) and bool(b),
            kind="logical",
        )

    def __or__(
        self,
        other: Any,
    ) -> ndarray:
        """Return the logical OR of two arrays element-wise.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.

        Returns
        -------
        ndarray
            Boolean array, with broadcasting.

        Notes
        -----
        Operands are evaluated via Python ``bool()``. Unlike NumPy,
        ``out`` and ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: bool(a) or bool(b),
            kind="logical",
        )

    def __invert__(self) -> ndarray:
        """Compute bit-wise inversion, element-wise.

        Returns
        -------
        ndarray
            Boolean array with each element logically negated.

        Raises
        ------
        TypeError
            If the array dtype is not boolean.

        Notes
        -----
        Implemented as ``not bool(x)`` per element. Unlike NumPy,
        only boolean arrays are accepted; integer bit inversion is not
        supported.
        """
        return self._unary_op(
            lambda x: not bool(x),
            result_dtype=bool,
            operation="logical not",
            require_logical=True,
        )

    def __rand__(
        self,
        other: Any,
    ) -> ndarray:
        """Return the logical AND of two arrays element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Left operand when ``other & self`` is evaluated.

        Returns
        -------
        ndarray
            Boolean array, with broadcasting.

        Notes
        -----
        Operands are evaluated via Python ``bool()``. Unlike NumPy,
        ``out`` and ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: bool(b) and bool(a),
            kind="logical",
        )

    def __ror__(
        self,
        other: Any,
    ) -> ndarray:
        """Return the logical OR of two arrays element-wise (reflected).

        Parameters
        ----------
        other : array_like or scalar
            Left operand when ``other | self`` is evaluated.

        Returns
        -------
        ndarray
            Boolean array, with broadcasting.

        Notes
        -----
        Operands are evaluated via Python ``bool()``. Unlike NumPy,
        ``out`` and ``where`` are not exposed on dunder operators.
        """
        return self._binary_op(
            other,
            lambda a, b: bool(b) or bool(a),
            kind="logical",
        )

    def _validate_binary_op(
        self,
        other: Any,
        *,
        kind: str,
    ) -> None:
        """Validate operand dtypes before a binary element-wise operation.

        Called by :meth:`_binary_op` after coercion. Raises
        ``TypeError`` when dtypes are incompatible with ``kind``.

        Parameters
        ----------
        other : ndarray or scalar
            Second operand (already coerced when array-like).
        kind : {'arithmetic', 'comparison', 'logical', 'truediv'}
            Operation category. ``'truediv'`` is validated as arithmetic.
        """
        if kind == "arithmetic":
            require_numeric_dtype(self.dtype, operation="arithmetic")

            if is_ndarray(other):
                require_numeric_dtype(other.dtype, operation="arithmetic")
                promote_dtypes(self.dtype, other.dtype)
            else:
                require_numeric_dtype(
                    infer_dtype_from_value(other),
                    operation="arithmetic",
                )
            return

        if kind == "comparison":
            if is_ndarray(other):
                require_comparable_dtypes(
                    self.dtype,
                    other.dtype,
                    operation="comparison",
                )
            else:
                require_comparable_dtypes(
                    self.dtype,
                    infer_dtype_from_value(other),
                    operation="comparison",
                )
            return

        require_logical_dtype(self.dtype, operation="logical")

        if is_ndarray(other):
            require_logical_dtype(other.dtype, operation="logical")
        else:
            require_logical_dtype(
                infer_dtype_from_value(other),
                operation="logical",
            )

    def _resolve_binary_result_dtype(
        self,
        other: Any,
        *,
        kind: str,
    ) -> type[Any]:
        """Determine the result dtype for a binary element-wise operation.

        Called by :meth:`_binary_op` when ``result_dtype`` is not
        supplied explicitly.

        Parameters
        ----------
        other : ndarray or scalar
            Second operand.
        kind : {'arithmetic', 'comparison', 'logical', 'truediv'}
            Operation category. Comparisons and logical ops always
            resolve to ``bool`` (or a promoted logical dtype).

        Returns
        -------
        type
            Target dtype for casting operation results.

        Notes
        -----
        For ``kind='truediv'``, applies :func:`result_dtype_for_truediv`
        so integer inputs promote to ``float`` or ``complex`` as in
        NumPy. Unlike NumPy, only the five supported element dtypes are
        considered.
        """
        if kind == "comparison":
            if is_ndarray(other):
                return result_dtype_for_comparison(self.dtype, other.dtype)

            return bool

        if kind == "logical":
            if is_ndarray(other):
                return result_dtype_for_logical(self.dtype, other.dtype)

            return bool

        if is_ndarray(other):
            result = result_dtype_for_arithmetic(self.dtype, other.dtype)
        else:
            result = promote_dtypes(
                self.dtype,
                infer_dtype_from_value(other),
            )

        if kind == "truediv":
            if is_ndarray(other):
                return result_dtype_for_truediv(self.dtype, other.dtype)

            return result_dtype_for_truediv(
                self.dtype,
                infer_dtype_from_value(other),
            )

        return result

    def _binary_op(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        out: ndarray | None = None,
        where: bool | ndarray = True,
        *,
        result_dtype: type[Any] | None = None,
        kind: str = "arithmetic",
    ) -> ndarray:
        """Execute a binary element-wise operation with optional masking.

        Central dispatcher for dunder operators (``__add__``, ``__eq__``,
        etc.) and module-level ufuncs created by
        :func:`numpy.ufuncs._make_binary_ufunc`. Handles coercion,
        broadcasting, dtype validation, and optional ``out``/``where``
        semantics.

        Parameters
        ----------
        other : array_like or scalar
            Second operand.
        op : callable
            Binary scalar function applied element-wise.
        out : ndarray, optional
            Pre-allocated output array. When ``None`` and ``where`` is
            fully True, a new contiguous array is returned.
        where : bool or ndarray, optional
            Boolean mask broadcast to the result shape. Positions where
            the mask is False are skipped (leaving ``out`` unchanged when
            provided).
        result_dtype : type, optional
            Explicit result dtype. When ``None``, resolved via
            :meth:`_resolve_binary_result_dtype`.
        kind : str, optional
            Operation category passed to validation and dtype resolution.
            Default is ``'arithmetic'``.

        Returns
        -------
        ndarray
            Element-wise result, either newly allocated or written into
            ``out``.

        Raises
        ------
        TypeError
            If operand dtypes are invalid for ``kind``.
        ValueError
            If ``out`` shape or dtype does not match the computed result.

        Notes
        -----
        Delegates unmasked computation to :meth:`_binary_op_unmasked` or
        :meth:`_binary_op_write_unmasked` when possible. Unlike NumPy
        ufuncs, masked positions with ``out=None`` leave placeholder
        values in a newly allocated array rather than undefined memory.
        """
        if is_basic_array_like(other):
            other = coerce_to_ndarray(other, name="operand")

        self._validate_binary_op(other, kind=kind)

        if result_dtype is None:
            result_dtype = self._resolve_binary_result_dtype(other, kind=kind)

        if is_ndarray(other):
            result_shape = broadcast_shape(self.shape, other.shape)
            left = self.broadcast_to(result_shape)
            right = other.broadcast_to(result_shape)
        else:
            result_shape = self.shape
            left = self
            right = other

        if is_ndarray(where):
            if not is_bool_dtype(where.dtype):
                raise where_must_be_bool(got=where)

            mask = where.broadcast_to(result_shape)
        elif isinstance(where, bool):
            mask = where
        else:
            raise where_must_be_bool(got=where)

        if out is None:
            if mask is True:
                return self._binary_op_unmasked(
                    left,
                    right,
                    op,
                    result_shape=result_shape,
                    result_dtype=result_dtype,
                )

            out = placeholder_contiguous_ndarray(
                result_shape,
                dtype=result_dtype,
            )
        elif out.dtype is not result_dtype:
            raise out_dtype_mismatch(
                out_dtype=out.dtype,
                result_dtype=result_dtype,
            )

        if out.shape != result_shape:
            raise out_shape_mismatch(
                out_shape=out.shape,
                result_shape=result_shape,
            )

        # Fast path: write directly into a pre-allocated contiguous out buffer.
        if mask is True and out._is_c_contiguous():
            self._binary_op_write_unmasked(
                left,
                right,
                op,
                out,
                result_dtype=result_dtype,
            )
            return out

        for source_index in iter_indices(out.shape):
            should_calculate = (
                mask if isinstance(mask, bool) else mask[source_index]
            )

            if should_calculate:
                if is_ndarray(right):
                    left_value = left._get_value(source_index)
                    right_value = right._get_value(source_index)
                else:
                    left_value = left._get_value(source_index)
                    right_value = right

                out._set_value(
                    source_index,
                    cast_value(result_dtype, op(left_value, right_value)),
                )

        return out

    def _binary_op_unmasked(
        self,
        left: ndarray,
        right: Any,
        op: Callable[[Any, Any], Any],
        *,
        result_shape: tuple[int, ...],
        result_dtype: type[Any],
    ) -> ndarray:
        """Compute an unmasked binary op into a new contiguous array.

        Called by :meth:`_binary_op` when ``where`` is True and no
        pre-allocated ``out`` buffer is supplied.

        Parameters
        ----------
        left : ndarray
            Left operand, already broadcast to ``result_shape``.
        right : ndarray or scalar
            Right operand; arrays are broadcast to ``result_shape``.
        op : callable
            Binary scalar function.
        result_shape : tuple of int
            Shape of the output array.
        result_dtype : type
            Target dtype for the result.

        Returns
        -------
        ndarray
            New C-contiguous array containing the element-wise results.

        Notes
        -----
        Uses flat-buffer zip iteration when both array operands are
        C-contiguous with matching shapes; otherwise falls back to
        index-based :meth:`ndarray._get_value`. Skips redundant casting
        via ``skip_cast`` when result dtype matches operand dtypes.
        """
        if is_ndarray(right):
            if (
                left.shape == right.shape == result_shape
                and left._is_c_contiguous()
                and right._is_c_contiguous()
            ):
                left_flat = left._flat_buffer_slice()
                right_flat = right._flat_buffer_slice()
                need_cast = result_dtype not in (left.dtype, right.dtype)

                if need_cast:
                    result = [
                        cast_value(result_dtype, op(left_value, right_value))
                        for left_value, right_value in zip(
                            left_flat,
                            right_flat,
                        )
                    ]
                else:
                    result = [
                        op(left_value, right_value)
                        for left_value, right_value in zip(
                            left_flat,
                            right_flat,
                        )
                    ]

                return contiguous_ndarray(
                    result,
                    result_shape,
                    dtype=result_dtype,
                    skip_cast=not need_cast,
                )

            result = []

            for source_index in iter_indices(result_shape):
                result.append(
                    cast_value(
                        result_dtype,
                        op(
                            left._get_value(source_index),
                            right._get_value(source_index),
                        ),
                    )
                )

            return contiguous_ndarray(result, result_shape, dtype=result_dtype)

        if left.shape == result_shape and left._is_c_contiguous():
            left_flat = left._flat_buffer_slice()
            need_cast = result_dtype is not left.dtype

            if need_cast:
                result = [
                    cast_value(result_dtype, op(left_value, right))
                    for left_value in left_flat
                ]
            else:
                result = [op(left_value, right) for left_value in left_flat]

            return contiguous_ndarray(
                result,
                result_shape,
                dtype=result_dtype,
                skip_cast=not need_cast,
            )

        result = [
            cast_value(result_dtype, op(left._get_value(source_index), right))
            for source_index in iter_indices(result_shape)
        ]

        return contiguous_ndarray(result, result_shape, dtype=result_dtype)

    def _binary_op_write_unmasked(
        self,
        left: ndarray,
        right: Any,
        op: Callable[[Any, Any], Any],
        out: ndarray,
        *,
        result_dtype: type[Any],
    ) -> None:
        """Write an unmasked binary op in-place into ``out``.

        Called by :meth:`_binary_op` when ``where`` is True, ``out`` is
        provided, and ``out`` is C-contiguous.

        Parameters
        ----------
        left : ndarray
            Left operand, already broadcast to ``out.shape``.
        right : ndarray or scalar
            Right operand.
        op : callable
            Binary scalar function.
        out : ndarray
            Pre-allocated C-contiguous output buffer.
        result_dtype : type
            Target dtype for casting each computed value.

        Notes
        -----
        Writes directly into ``out.buffer`` at ``out.offset`` when the
        left operand (and right, if an array) is C-contiguous. Unlike
        NumPy, there is no check that ``out`` is writeable beyond dtype
        and shape validation performed by the caller.
        """
        if is_ndarray(right):
            if left._is_c_contiguous() and right._is_c_contiguous():
                left_flat = left._flat_buffer_slice()
                right_flat = right._flat_buffer_slice()
                need_cast = result_dtype is not out.dtype
                buffer = out.buffer
                start = out.offset

                for offset, (left_value, right_value) in enumerate(
                    zip(left_flat, right_flat)
                ):
                    value = op(left_value, right_value)

                    if need_cast:
                        value = cast_value(result_dtype, value)

                    buffer[start + offset] = value

                return

            for source_index in iter_indices(out.shape):
                out._set_value(
                    source_index,
                    cast_value(
                        result_dtype,
                        op(
                            left._get_value(source_index),
                            right._get_value(source_index),
                        ),
                    ),
                )

            return

        if left._is_c_contiguous():
            left_flat = left._flat_buffer_slice()
            need_cast = result_dtype is not out.dtype
            buffer = out.buffer
            start = out.offset

            for offset, left_value in enumerate(left_flat):
                value = op(left_value, right)

                if need_cast:
                    value = cast_value(result_dtype, value)

                buffer[start + offset] = value

            return

        for source_index in iter_indices(out.shape):
            out._set_value(
                source_index,
                cast_value(
                    result_dtype,
                    op(left._get_value(source_index), right),
                ),
            )

    def _unary_op(
        self,
        op: Callable[[Any], Any],
        *,
        result_dtype: type[Any] | None = None,
        operation: str = "unary operation",
        require_logical: bool = False,
        require_numeric: bool = False,
    ) -> ndarray:
        """Apply a unary operation element-wise into a new array.

        Called by dunder methods (``__neg__``, ``__abs__``, ``__invert__``)
        and unary ufuncs from :func:`numpy.ufuncs._make_unary_ufunc`.

        Parameters
        ----------
        op : callable
            Unary scalar function applied to each element.
        result_dtype : type, optional
            Target dtype. Defaults to ``self.dtype`` when ``None``.
        operation : str, optional
            Label used in error messages for dtype validation.
        require_logical : bool, optional
            When True, require a boolean dtype.
        require_numeric : bool, optional
            When True, require a numeric dtype.

        Returns
        -------
        ndarray
            New C-contiguous array with the same shape as ``self``.

        Notes
        -----
        Uses flat-buffer iteration for C-contiguous inputs. Unlike NumPy
        ufuncs, there is no ``out`` or ``where`` parameter on this path.
        """
        if require_logical:
            require_logical_dtype(self.dtype, operation=operation)
        elif require_numeric:
            require_numeric_dtype(self.dtype, operation=operation)

        if result_dtype is None:
            result_dtype = self.dtype

        need_cast = result_dtype is not self.dtype

        if self._is_c_contiguous():
            flat_values = self._flat_buffer_slice()

            if need_cast:
                result = [
                    cast_value(result_dtype, op(value)) for value in flat_values
                ]
            else:
                result = [op(value) for value in flat_values]

            return contiguous_ndarray(
                result,
                self.shape,
                dtype=result_dtype,
                skip_cast=not need_cast,
            )

        result = [
            cast_value(result_dtype, op(value)) if need_cast else op(value)
            for value in self._iter_values()
        ]

        return contiguous_ndarray(result, self.shape, dtype=result_dtype)
