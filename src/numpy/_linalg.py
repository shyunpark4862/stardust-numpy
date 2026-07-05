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

"""Dot products and matrix multiplication on ``ndarray``.

:class:`LinalgMixin` adds :meth:`dot`, :meth:`vdot`, and :meth:`matmul`
with optional batched broadcasting for rank > 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._dtype import promote_dtypes, require_numeric_dtype
from ._errors import (
    dot_not_implemented,
    dot_shape_mismatch,
    matmul_scalar_operand,
    matmul_shape_mismatch,
    vdot_size_mismatch,
)
from ._factory import contiguous_ndarray
from ._iter import iter_indices
from ._validation import coerce_to_ndarray
from .broadcasting import broadcast_shape

if TYPE_CHECKING:
    from .ndarray import ndarray


class LinalgMixin:
    """Mixin providing dot products and matrix multiplication for ``ndarray``."""

    def dot(
        self,
        other: ndarray,
    ) -> Any | ndarray:
        """
        Dot product of two arrays.

        For 1-D arrays this is the inner product; for 2-D arrays it is matrix
        multiplication. Higher-dimensional ``dot`` is not supported.

        Parameters
        ----------
        other : ndarray
            Second operand. Both operands must have numeric dtypes.

        Returns
        -------
        scalar or ndarray
            Inner product (1-D), matrix-vector product (2-D × 1-D), vector-
            matrix product (1-D × 2-D), or matrix product (2-D × 2-D).

        Raises
        ------
        ValueError
            If inner dimensions are incompatible or either operand has more
            than two dimensions.

        Notes
        -----
        Unlike NumPy's ``dot``, N-dimensional tensor contraction is not
        implemented. Use ``matmul`` for batched matrix multiplication when
        either operand has more than two dimensions.
        """
        require_numeric_dtype(self.dtype, operation="dot")
        other = coerce_to_ndarray(other)
        require_numeric_dtype(other.dtype, operation="dot")

        if self.ndim == 1 and other.ndim == 1:
            return self._dot_vectors(other)

        if self.ndim == 2 and other.ndim == 1:
            return self._dot_matrix_vector(other)

        if self.ndim == 1 and other.ndim == 2:
            return self._dot_vector_matrix(other)

        if self.ndim == 2 and other.ndim == 2:
            return self._dot_matrices(other)

        raise dot_not_implemented(ndim=max(self.ndim, other.ndim))

    def vdot(
        self,
        other: ndarray,
    ) -> Any:
        """
        Return the dot product of two vectors after flattening.

        The first argument is conjugated when its elements are complex.

        Parameters
        ----------
        other : ndarray
            Second operand. Both operands must have numeric dtypes and the
            same total size after raveling.

        Returns
        -------
        scalar
            Scalar dot product of the flattened operands.

        Notes
        -----
        Unlike NumPy, operands are always flattened; there is no
        matrix-conjugate behavior for higher-dimensional inputs.
        """
        require_numeric_dtype(self.dtype, operation="vdot")
        other = coerce_to_ndarray(other)
        require_numeric_dtype(other.dtype, operation="vdot")

        left = self.ravel()
        right = other.ravel()

        if left.size != right.size:
            raise vdot_size_mismatch(left_size=left.size, right_size=right.size)

        left_values = left._flat_buffer_slice()
        right_values = right._flat_buffer_slice()
        total = 0

        for left_value, right_value in zip(left_values, right_values):
            if isinstance(left_value, complex):
                left_value = left_value.conjugate()

            total += left_value * right_value

        return total

    def matmul(
        self,
        other: ndarray,
    ) -> Any | ndarray:
        """
        Matrix product of two arrays.

        Parameters
        ----------
        other : ndarray
            Second operand. Scalars (0-D arrays) are rejected.

        Returns
        -------
        scalar or ndarray
            Matrix product. 1-D operands are promoted to row or column vectors
            for the multiply and the extra length-1 axis is removed from the
            result when the corresponding operand was a vector.

        Raises
        ------
        ValueError
            If either operand is 0-D, or trailing matrix dimensions are
            incompatible after batch broadcasting.

        Notes
        -----
        For operands with at most two dimensions, ``matmul`` delegates to
        ``dot``. For batched inputs, batch leading dimensions are broadcast
        together and each matrix pair is multiplied independently. There is no
        ``out`` argument and no support for NumPy's stack-of-matrices
        ``tensordot``-style contraction beyond batch matmul.
        """
        other = coerce_to_ndarray(other, reject_scalars=True)

        if self.ndim == 0 or other.ndim == 0:
            raise matmul_scalar_operand()

        if self.ndim <= 2 and other.ndim <= 2:
            return self.dot(other)

        left_was_vector = self.ndim == 1
        right_was_vector = other.ndim == 1

        # Promote 1-D operands to 2-D for batched matrix multiply.
        left = self.reshape((1, self.shape[0])) if left_was_vector else self
        right = (
            other.reshape((other.shape[0], 1)) if right_was_vector else other
        )

        left_rows, left_columns = left.shape[-2:]
        right_rows, right_columns = right.shape[-2:]

        if left_columns != right_rows:
            raise matmul_shape_mismatch(
                left_shape=left.shape,
                right_shape=right.shape,
                left_cols=left_columns,
                right_rows=right_rows,
            )

        batch_shape = broadcast_shape(left.shape[:-2], right.shape[:-2])

        left = left.broadcast_to(batch_shape + (left_rows, left_columns))
        right = right.broadcast_to(batch_shape + (right_rows, right_columns))

        result_buffer = []

        for batch_index in iter_indices(batch_shape):
            for row in range(left_rows):
                for column in range(right_columns):
                    total = 0

                    for shared_axis in range(left_columns):
                        total += (
                            left[batch_index + (row, shared_axis)]
                            * right[batch_index + (shared_axis, column)]
                        )

                    result_buffer.append(total)

        result_shape = batch_shape

        if not left_was_vector:
            result_shape += (left_rows,)

        if not right_was_vector:
            result_shape += (right_columns,)

        result_dtype = promote_dtypes(left.dtype, right.dtype)

        return contiguous_ndarray(
            result_buffer,
            result_shape,
            dtype=result_dtype,
        )

    def __matmul__(
        self,
        other: ndarray,
    ) -> Any | ndarray:
        """Operator alias for :meth:`matmul`."""
        return self.matmul(other)

    def _dot_vectors(
        self,
        other: ndarray,
    ) -> Any:
        """Compute the inner product of two 1-D arrays.

        Called by :meth:`dot` when both operands are one-dimensional.
        Unlike :meth:`vdot`, neither operand is conjugated.

        Parameters
        ----------
        other : ndarray
            Second 1-D operand. Must have the same length as ``self``.

        Returns
        -------
        scalar
            Python scalar (``int``, ``float``, or ``complex``) equal to
            ``sum(self[i] * other[i])``.

        Raises
        ------
        ValueError
            If the operands have different lengths.

        Notes
        -----
        Uses flat buffer slices when available; no BLAS or SIMD
        acceleration. Unlike NumPy's ``numpy.dot`` for 1-D inputs, the
        result is always a Python scalar rather than a 0-D ``ndarray``.
        """
        if self.shape[0] != other.shape[0]:
            raise dot_shape_mismatch(
                left_shape=self.shape,
                right_shape=other.shape,
                left_cols=self.shape[0],
                right_rows=other.shape[0],
            )

        left_values = self._flat_buffer_slice()
        right_values = other._flat_buffer_slice()
        total = 0

        for left_value, right_value in zip(left_values, right_values):
            total += left_value * right_value

        return total

    def _dot_matrix_vector(
        self,
        other: ndarray,
    ) -> ndarray:
        """Compute the matrix-vector product for 2-D × 1-D operands.

        Called by :meth:`dot` when ``self`` is two-dimensional and
        ``other`` is one-dimensional. Computes ``self @ other`` as a
        1-D result of length ``self.shape[0]``.

        Parameters
        ----------
        other : ndarray
            1-D vector whose length must equal ``self.shape[1]``.

        Returns
        -------
        ndarray
            1-D array of shape ``(rows,)`` with dtype promoted from both
            operands.

        Raises
        ------
        ValueError
            If ``self.shape[1] != other.shape[0]``.

        Notes
        -----
        When both operands are C-contiguous, uses direct flat-buffer
        indexing to avoid per-element stride arithmetic. Otherwise falls
        back to :meth:`ndarray._get_value`. Unlike NumPy, there is no
        ``out`` parameter and no support for N-dimensional contraction.
        """
        rows, columns = self.shape

        if columns != other.shape[0]:
            raise dot_shape_mismatch(
                left_shape=self.shape,
                right_shape=other.shape,
                left_cols=columns,
                right_rows=other.shape[0],
            )

        result_dtype = promote_dtypes(self.dtype, other.dtype)

        if self._is_c_contiguous() and other._is_c_contiguous():
            # Fast path: direct buffer indexing without per-element stride math.
            left_buffer = self._flat_buffer_slice()
            right_buffer = other._flat_buffer_slice()
            result_buffer = []

            for row in range(rows):
                row_start = row * columns
                total = 0

                for column in range(columns):
                    total += (
                        left_buffer[row_start + column] * right_buffer[column]
                    )

                result_buffer.append(total)

            return contiguous_ndarray(
                result_buffer,
                (rows,),
                dtype=result_dtype,
            )

        result_buffer = []

        for row in range(rows):
            total = 0

            for column in range(columns):
                total += self._get_value((row, column)) * other._get_value(
                    (column,)
                )

            result_buffer.append(total)

        return contiguous_ndarray(
            result_buffer,
            (rows,),
            dtype=result_dtype,
        )

    def _dot_vector_matrix(
        self,
        other: ndarray,
    ) -> ndarray:
        """Compute the vector-matrix product for 1-D × 2-D operands.

        Called by :meth:`dot` when ``self`` is one-dimensional and
        ``other`` is two-dimensional. Computes ``self @ other`` as a
        1-D result of length ``other.shape[1]``.

        Parameters
        ----------
        other : ndarray
            2-D matrix whose first dimension must equal ``self.shape[0]``.

        Returns
        -------
        ndarray
            1-D array of shape ``(columns,)`` with dtype promoted from both
            operands.

        Raises
        ------
        ValueError
            If ``self.shape[0] != other.shape[0]``.

        Notes
        -----
        Uses a C-contiguous fast path when both operands allow flat-buffer
        access. Unlike NumPy's general ``dot``, higher-dimensional tensor
        products are not supported.
        """
        rows, columns = other.shape

        if self.shape[0] != rows:
            raise dot_shape_mismatch(
                left_shape=self.shape,
                right_shape=other.shape,
                left_cols=self.shape[0],
                right_rows=rows,
            )

        result_dtype = promote_dtypes(self.dtype, other.dtype)

        if self._is_c_contiguous() and other._is_c_contiguous():
            left_buffer = self._flat_buffer_slice()
            right_buffer = other._flat_buffer_slice()
            result_buffer = []

            for column in range(columns):
                total = 0

                for row in range(rows):
                    total += (
                        left_buffer[row] * right_buffer[row * columns + column]
                    )

                result_buffer.append(total)

            return contiguous_ndarray(
                result_buffer,
                (columns,),
                dtype=result_dtype,
            )

        result_buffer = []

        for column in range(columns):
            total = 0

            for row in range(rows):
                total += self._get_value((row,)) * other._get_value(
                    (row, column)
                )

            result_buffer.append(total)

        return contiguous_ndarray(
            result_buffer,
            (columns,),
            dtype=result_dtype,
        )

    def _dot_matrices(
        self,
        other: ndarray,
    ) -> ndarray:
        """Compute the matrix-matrix product for 2-D × 2-D operands.

        Called by :meth:`dot` when both operands are two-dimensional.
        Implements standard matrix multiplication with inner dimension
        ``self.shape[1] == other.shape[0]``.

        Parameters
        ----------
        other : ndarray
            Right-hand 2-D matrix.

        Returns
        -------
        ndarray
            2-D array of shape ``(self.shape[0], other.shape[1])`` with
            promoted dtype.

        Raises
        ------
        ValueError
            If inner dimensions are incompatible.

        Notes
        -----
        Uses triple-nested loops with an optional C-contiguous fast path.
        Unlike NumPy, there is no ``out`` argument, no BLAS backend, and
        no N-dimensional ``tensordot``-style contraction via ``dot``.
        """
        left_rows, left_columns = self.shape
        right_rows, right_columns = other.shape

        if left_columns != right_rows:
            raise dot_shape_mismatch(
                left_shape=self.shape,
                right_shape=other.shape,
                left_cols=left_columns,
                right_rows=right_rows,
            )

        result_dtype = promote_dtypes(self.dtype, other.dtype)

        if self._is_c_contiguous() and other._is_c_contiguous():
            left_buffer = self._flat_buffer_slice()
            right_buffer = other._flat_buffer_slice()
            result_buffer = []

            for row in range(left_rows):
                row_start = row * left_columns

                for column in range(right_columns):
                    total = 0

                    for shared_axis in range(left_columns):
                        total += (
                            left_buffer[row_start + shared_axis]
                            * right_buffer[shared_axis * right_columns + column]
                        )

                    result_buffer.append(total)

            return contiguous_ndarray(
                result_buffer,
                (left_rows, right_columns),
                dtype=result_dtype,
            )

        result_buffer = []

        for row in range(left_rows):
            for column in range(right_columns):
                total = 0

                for shared_axis in range(left_columns):
                    total += self._get_value(
                        (row, shared_axis)
                    ) * other._get_value((shared_axis, column))

                result_buffer.append(total)

        return contiguous_ndarray(
            result_buffer,
            (left_rows, right_columns),
            dtype=result_dtype,
        )
