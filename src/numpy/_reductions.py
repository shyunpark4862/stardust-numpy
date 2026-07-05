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

"""Axis reductions and cumulative operations for ``ndarray``.

:class:`ReductionMixin` implements ``sum``, ``mean``, ``var``, ``min``,
``max``, ``argmin``, ``argmax``, ``cumsum``, ``cumprod``, and related
helpers over selected axes or the full array.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, Any, Callable

from ._axis import normalize_axes, normalize_axis
from ._dtype import (
    cast_value,
    default_fill_value,
    require_logical_dtype,
    require_numeric_dtype,
    require_orderable_dtype,
    result_dtype_for_mean,
)
from ._errors import (
    cannot_cumulate_empty_axis,
    cannot_reduce_empty_axis,
    empty_reduction,
)
from ._factory import contiguous_ndarray
from ._iter import iter_indices
from ._shape import flat_index, keepdims_shape, merge_index, size_of_shape

if TYPE_CHECKING:
    from .ndarray import ndarray


class ReductionMixin:
    """Mixin providing axis reductions and cumulative operations for ``ndarray``."""

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Sum of array elements over the given axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to sum. ``None`` sums all elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            The sum. Returns a scalar when ``axis is None`` and
            ``keepdims is False``.

        Notes
        -----
        Requires a numeric dtype. Empty axes raise ``ValueError`` when
        ``axis`` is specified.
        """
        self._require_numeric("sum")
        reducer = lambda total, value: total + value

        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_all(reducer, 0),
                keepdims,
            )

        axes = self._normalize_reduction_axes(axis)
        return self._reduce_axes(axes, reducer, 0, keepdims)

    def prod(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Product of array elements over the given axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to multiply. ``None`` multiplies all
            elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            The product.
        """
        self._require_numeric("prod")
        reducer = lambda total, value: total * value

        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_all(reducer, 1),
                keepdims,
            )

        axes = self._normalize_reduction_axes(axis)
        return self._reduce_axes(axes, reducer, 1, keepdims)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Return the minimum of array elements over the given axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to reduce. ``None`` reduces over all
            elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            The minimum value. NaN values propagate: if any element is NaN,
            the result is NaN.

        Notes
        -----
        Requires an orderable dtype (numeric or string).
        """
        self._require_orderable("min")
        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_extremum_all(
                    operation="min",
                    is_better=lambda candidate, best: candidate < best,
                ),
                keepdims,
            )

        axes = self._normalize_reduction_axes(axis)
        self._require_non_empty_axes(axes, operation="min")
        return self._reduce_extremum_axes(
            axes,
            lambda candidate, best: candidate < best,
            keepdims,
        )

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Return the maximum of array elements over the given axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to reduce. ``None`` reduces over all
            elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            The maximum value. NaN values propagate: if any element is NaN,
            the result is NaN.
        """
        self._require_orderable("max")
        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_extremum_all(
                    operation="max",
                    is_better=lambda candidate, best: candidate > best,
                ),
                keepdims,
            )

        axes = self._normalize_reduction_axes(axis)
        self._require_non_empty_axes(axes, operation="max")
        return self._reduce_extremum_axes(
            axes,
            lambda candidate, best: candidate > best,
            keepdims,
        )

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Compute the arithmetic mean along the specified axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to compute the mean. ``None`` uses all
            elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            The mean. Integer inputs promote to ``float`` in the result.

        Notes
        -----
        Unlike NumPy, there is no ``dtype`` or ``where`` argument. Reducing
        over an empty array raises ``ValueError``.
        """
        self._require_numeric("mean")
        if axis is None:
            if self.size == 0:
                raise empty_reduction(operation="mean")

            return self._finalize_reduce_all(
                self.sum() / self.size,
                keepdims,
                dtype=result_dtype_for_mean(self.dtype),
            )

        axes = self._normalize_reduction_axes(axis)
        self._require_non_empty_axes(axes, operation="mean")
        count = self._axis_element_count(axes)

        return self.sum(axis=axes, keepdims=keepdims) / count

    def any(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Test whether any array element evaluates to ``True``.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to reduce. ``None`` tests all elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        bool or ndarray of bool
            ``True`` if any element is truthy along the reduced axes.
        """
        self._require_logical("any")
        reducer = lambda total, value: total or bool(value)

        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_all(reducer, False),
                keepdims,
                dtype=bool,
            )

        axes = self._normalize_reduction_axes(axis)
        return self._reduce_axes(
            axes,
            reducer,
            False,
            keepdims=keepdims,
            dtype=bool,
        )

    def all(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Test whether all array elements evaluate to ``True``.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to reduce. ``None`` tests all elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        bool or ndarray of bool
            ``True`` if every element is truthy along the reduced axes.
        """
        self._require_logical("all")
        reducer = lambda total, value: total and bool(value)

        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_all(reducer, True),
                keepdims,
                dtype=bool,
            )

        axes = self._normalize_reduction_axes(axis)
        return self._reduce_axes(
            axes,
            reducer,
            True,
            keepdims=keepdims,
            dtype=bool,
        )

    def count_nonzero(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Count the number of non-zero / ``True`` elements.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to count. ``None`` counts over all
            elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        int or ndarray of int
            Count of truthy elements along the reduced axes.
        """
        self._require_logical("count_nonzero")
        reducer = lambda total, value: total + int(bool(value))

        if axis is None:
            return self._finalize_reduce_all(
                self._reduce_all(reducer, 0),
                keepdims,
                dtype=int,
            )

        axes = self._normalize_reduction_axes(axis)
        return self._reduce_axes(
            axes,
            reducer,
            0,
            keepdims=keepdims,
            dtype=int,
        )

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Compute the variance along the specified axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to compute variance. ``None`` uses all
            elements.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            Population variance (``ddof=0``). Integer inputs promote to
            ``float`` in the result.

        Notes
        -----
        Unlike NumPy, there is no ``ddof`` parameter; variance always divides
        by the element count ``N``, not ``N - ddof``.
        """
        self._require_numeric("var")
        if axis is None:
            if self.size == 0:
                raise empty_reduction(operation="var")

            return self._finalize_reduce_all(
                self._welford_variance_all(),
                keepdims,
                dtype=result_dtype_for_mean(self.dtype),
            )

        axes = self._normalize_reduction_axes(axis)
        self._require_non_empty_axes(axes, operation="var")

        return self._welford_variance_axes(axes, keepdims)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Any | ndarray:
        """
        Compute the standard deviation along the specified axes.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional (default: None)
            Axis or axes along which to compute standard deviation.
        keepdims : bool, optional (default: False)
            If ``True``, reduced axes are kept as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            Square root of ``var``. Uses population variance (``ddof=0``).

        Notes
        -----
        Implemented as ``var ** 0.5``; no separate Welford pass.
        """
        return self.var(axis=axis, keepdims=keepdims) ** 0.5

    def argmin(
        self,
        axis: int | None = None,
    ) -> int | ndarray:
        """
        Return indices of the minimum values along an axis.

        Parameters
        ----------
        axis : int or None, optional (default: None)
            Axis along which to find minima. ``None`` flattens the array and
            returns a single linear index.

        Returns
        -------
        int or ndarray of int
            Index (or indices) of the minimum. Ties favor the first
            occurrence. NaN values propagate: the first NaN index is returned.

        Notes
        -----
        Unlike reductions with ``axis=None``, ``argmin`` with ``axis=None``
        returns a flat index, not an array shaped like the input with
        ``keepdims``.
        """
        self._require_orderable("argmin")
        if axis is None:
            return self._arg_extremum_all(
                operation="argmin",
                is_better=lambda candidate, best: candidate < best,
            )

        axis = normalize_axis(axis, self.ndim)

        if self.shape[axis] == 0:
            raise cannot_reduce_empty_axis(
                operation="argmin",
                axis=axis,
                shape=self.shape,
            )

        return self._arg_reduce_axis(
            axis,
            lambda candidate, best: candidate < best,
        )

    def argmax(
        self,
        axis: int | None = None,
    ) -> int | ndarray:
        """
        Return indices of the maximum values along an axis.

        Parameters
        ----------
        axis : int or None, optional (default: None)
            Axis along which to find maxima. ``None`` flattens the array and
            returns a single linear index.

        Returns
        -------
        int or ndarray of int
            Index (or indices) of the maximum. Ties favor the first
            occurrence. NaN values propagate.
        """
        self._require_orderable("argmax")
        if axis is None:
            return self._arg_extremum_all(
                operation="argmax",
                is_better=lambda candidate, best: candidate > best,
            )

        axis = normalize_axis(axis, self.ndim)

        if self.shape[axis] == 0:
            raise cannot_reduce_empty_axis(
                operation="argmax",
                axis=axis,
                shape=self.shape,
            )

        return self._arg_reduce_axis(
            axis,
            lambda candidate, best: candidate > best,
        )

    def cumsum(
        self,
        axis: int | None = None,
    ) -> ndarray:
        """
        Return the cumulative sum of elements along a given axis.

        Parameters
        ----------
        axis : int or None, optional (default: None)
            Axis along which to cumulate. ``None`` flattens the array first
            and returns a 1-D result of length ``size``.

        Returns
        -------
        ndarray
            Cumulative sum with the same shape as the input (or shape
            ``(size,)`` when ``axis is None``).

        Notes
        -----
        An axis of length zero raises ``ValueError``.
        """
        self._require_numeric("cumsum")
        reducer = lambda total, value: total + value

        if axis is None:
            return self._cumulate_flat(reducer, 0)

        axis = normalize_axis(axis, self.ndim)

        if self.shape[axis] == 0:
            raise cannot_cumulate_empty_axis(
                operation="cumsum",
                axis=axis,
                shape=self.shape,
            )

        return self._cumulate_axis(axis, reducer)

    def cumprod(
        self,
        axis: int | None = None,
    ) -> ndarray:
        """
        Return the cumulative product of elements along a given axis.

        Parameters
        ----------
        axis : int or None, optional (default: None)
            Axis along which to cumulate. ``None`` flattens the array first
            and returns a 1-D result of length ``size``.

        Returns
        -------
        ndarray
            Cumulative product with the same shape as the input (or shape
            ``(size,)`` when ``axis is None``).
        """
        self._require_numeric("cumprod")
        reducer = lambda total, value: total * value

        if axis is None:
            return self._cumulate_flat(reducer, 1)

        axis = normalize_axis(axis, self.ndim)

        if self.shape[axis] == 0:
            raise cannot_cumulate_empty_axis(
                operation="cumprod",
                axis=axis,
                shape=self.shape,
            )

        return self._cumulate_axis(axis, reducer)

    def _require_numeric(self, operation: str) -> None:
        """Raise if ``self.dtype`` is not numeric; called at the start of sum/prod/mean/var."""
        require_numeric_dtype(self.dtype, operation=operation)

    def _require_orderable(self, operation: str) -> None:
        """Raise if ``self.dtype`` is not orderable; used by min/max/argmin/argmax."""
        require_orderable_dtype(self.dtype, operation=operation)

    def _require_logical(self, operation: str) -> None:
        """Raise if ``self.dtype`` is not bool; used by any/all/count_nonzero."""
        require_logical_dtype(self.dtype, operation=operation)

    def _normalize_reduction_axes(
        self,
        axis: int | tuple[int, ...],
    ) -> tuple[int, ...]:
        """Wrap ``normalize_axes`` with ``self.ndim`` for public ``axis`` arguments."""
        return normalize_axes(axis, self.ndim)

    def _require_non_empty_axes(
        self,
        axes: tuple[int, ...],
        *,
        operation: str,
    ) -> None:
        """Raise ``cannot_reduce_empty_axis`` if any axis in ``axes`` has length zero."""
        for axis in axes:
            if self.shape[axis] == 0:
                raise cannot_reduce_empty_axis(
                    operation=operation,
                    axis=axis,
                    shape=self.shape,
                )

    def _finalize_reduce_all(
        self,
        result: Any,
        keepdims: bool,
        *,
        dtype: Any | None = None,
    ) -> Any | ndarray:
        """Return a global-reduction scalar, or a length-1 array when ``keepdims``."""
        if not keepdims:
            return result

        target = dtype if dtype is not None else self.dtype

        return contiguous_ndarray(
            [cast_value(target, result)],
            (1,) * self.ndim,
            dtype=target,
        )

    def _make_reduction_result(
        self,
        result_buffer: list[Any],
        result_shape: tuple[int, ...],
        *,
        dtype: Any | None = None,
    ) -> Any | ndarray:
        """Unpack a flat ``result_buffer`` into a scalar or ``contiguous_ndarray``."""
        if result_shape == ():
            return result_buffer[0]

        target = dtype if dtype is not None else self.dtype

        return contiguous_ndarray(
            result_buffer,
            result_shape,
            dtype=target,
        )

    def _reduce_axes(
        self,
        axes: tuple[int, ...],
        reducer: Callable[[Any, Any], Any],
        initial: Any,
        keepdims: bool = False,
        *,
        dtype: Any | None = None,
    ) -> Any | ndarray:
        """Reduce over one or more axes with an associative binary ``reducer``.

        Parameters
        ----------
        axes : tuple of int
            Axes to collapse (must be normalized and non-empty when required).
        reducer : callable
            Binary function ``(accumulator, value) -> new_accumulator``.
        initial : Any
            Starting accumulator for each output slice.
        keepdims : bool, optional
            If ``True``, reduced axes appear as length-1 dimensions.
        dtype : type, optional
            Output dtype override passed to ``_make_reduction_result``.

        Returns
        -------
        scalar or ndarray
            One result per slice orthogonal to ``axes``.

        Notes
        -----
        Iterates outer indices over non-reduced dimensions and inner indices
        over the product of reduced axis lengths. ``merge_index`` maps each
        inner/outer pair to a source index for ``_get_value``. Used by
        ``sum``, ``prod``, ``any``, ``all``, and ``count_nonzero``.
        """
        axes_set = set(axes)

        result_shape = tuple(
            size for axis, size in enumerate(self.shape) if axis not in axes_set
        )
        reduced_shape = tuple(self.shape[axis] for axis in axes)
        result_buffer: list[Any] = []

        for outer_index in iter_indices(result_shape):
            total = initial

            for inner_index in iter_indices(reduced_shape):
                source_index = merge_index(
                    self.ndim,
                    axes_set,
                    outer_index,
                    inner_index,
                )
                total = reducer(total, self._get_value(source_index))

            result_buffer.append(total)

        if keepdims:
            result_shape = keepdims_shape(self.shape, axes_set)

        return self._make_reduction_result(
            result_buffer,
            result_shape,
            dtype=dtype,
        )

    def _reduce_extremum_axes(
        self,
        axes: tuple[int, ...],
        is_better: Callable[[Any, Any], bool],
        keepdims: bool = False,
    ) -> Any | ndarray:
        """Min/max reduction over axes with NaN propagation.

        Parameters
        ----------
        axes : tuple of int
            Axes to collapse.
        is_better : callable
            Comparison ``(candidate, best) -> bool``; ``<`` for min, ``>`` for max.
        keepdims : bool, optional
            If ``True``, reduced axes appear as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            Extremum per slice orthogonal to ``axes``.

        Notes
        -----
        For each outer slice the first inner element seeds ``best``. If it is
        NaN (via ``_is_nan_value``), that NaN is stored immediately. Otherwise
        remaining inner elements are scanned; the first NaN encountered wins
        and stops the scan. Among non-NaN values, ``is_better`` selects the
        extremum. Ties keep the earlier element. Used by ``min`` and ``max``.
        """
        axes_set = set(axes)
        reduced_shape = tuple(self.shape[axis] for axis in axes)

        result_shape = tuple(
            size for axis, size in enumerate(self.shape) if axis not in axes_set
        )
        result_buffer: list[Any] = []

        for outer_index in iter_indices(result_shape):
            first_inner = tuple(0 for _ in axes)
            source_index = merge_index(
                self.ndim,
                axes_set,
                outer_index,
                first_inner,
            )
            best = self._get_value(source_index)

            if _is_nan_value(best):
                result_buffer.append(best)
                continue

            for inner_index in iter_indices(reduced_shape):
                if inner_index == first_inner:
                    continue

                source_index = merge_index(
                    self.ndim,
                    axes_set,
                    outer_index,
                    inner_index,
                )
                candidate = self._get_value(source_index)

                if _is_nan_value(candidate):
                    best = candidate
                    break

                if is_better(candidate, best):
                    best = candidate

            result_buffer.append(best)

        if keepdims:
            result_shape = keepdims_shape(self.shape, axes_set)

        return self._make_reduction_result(result_buffer, result_shape)

    def _reduce_all(
        self,
        reducer: Callable[[Any, Any], Any],
        initial: Any,
    ) -> Any:
        """Fold ``reducer`` over every element in C-order.

        Parameters
        ----------
        reducer : callable
            Binary accumulator function.
        initial : Any
            Starting value before the first element.

        Returns
        -------
        Any
            Single scalar result.

        Notes
        -----
        Used when ``axis is None`` for associative reductions (sum, prod,
        any, all, count_nonzero). Does not handle NaN specially.
        """
        result = initial

        for value in self._iter_values():
            result = reducer(result, value)

        return result

    def _axis_element_count(
        self,
        axes: tuple[int, ...],
    ) -> int:
        """Product of ``self.shape[axis]`` for each axis in ``axes`` (mean divisor)."""
        return size_of_shape(tuple(self.shape[axis] for axis in axes))

    def _reduce_extremum_all(
        self,
        *,
        operation: str,
        is_better: Callable[[Any, Any], bool],
    ) -> Any:
        """Global min/max over all elements with NaN propagation.

        Parameters
        ----------
        operation : str
            Name for error messages (``"min"`` or ``"max"``).
        is_better : callable
            Comparison ``(candidate, best) -> bool``.

        Returns
        -------
        Any
            Extremum scalar.

        Raises
        ------
        ValueError
            If ``self.size == 0`` (``empty_reduction``).

        Notes
        -----
        The first element seeds ``best``. NaN in the first element or any
        later element short-circuits with that NaN. Otherwise ``is_better``
        updates ``best``. Ties favor the first occurrence.
        """
        if self.size == 0:
            raise empty_reduction(operation=operation)

        iterator = self._iter_values()
        best = next(iterator)

        if _is_nan_value(best):
            return best

        for candidate in iterator:
            if _is_nan_value(candidate):
                return candidate

            if is_better(candidate, best):
                best = candidate

        return best

    def _welford_variance_all(self) -> Any:
        """Population variance over all elements using Welford's online algorithm.

        Returns
        -------
        float or int-promoted float
            ``m2 / count`` where ``m2`` is the sum of squared deviations.

        Notes
        -----
        Single pass; numerically stable for floating-point data. Caller
        (``var`` with ``axis is None``) must ensure ``self.size > 0``.
        """
        count = 0
        mean = 0
        m2 = 0

        for value in self._iter_values():
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)

        return m2 / count

    def _welford_variance_axes(
        self,
        axes: tuple[int, ...],
        keepdims: bool,
    ) -> Any | ndarray:
        """Population variance over selected axes (Welford per outer slice).

        Parameters
        ----------
        axes : tuple of int
            Axes to reduce.
        keepdims : bool
            If ``True``, reduced axes appear as length-1 dimensions.

        Returns
        -------
        scalar or ndarray
            Variance per slice orthogonal to ``axes``, dtype from
            ``result_dtype_for_mean``.

        Notes
        -----
        For each outer index, Welford's algorithm accumulates ``mean`` and
        ``m2`` over the inner product of reduced axes. The result divides
        ``m2`` by the fixed inner count ``size_of_shape(reduced_shape)``
        (population variance, ``ddof=0``). Empty axes are rejected by the
        caller before invocation.
        """
        axes_set = set(axes)
        result_shape = tuple(
            size for axis, size in enumerate(self.shape) if axis not in axes_set
        )
        reduced_shape = tuple(self.shape[axis] for axis in axes)
        count = size_of_shape(reduced_shape)
        result_buffer: list[Any] = []

        for outer_index in iter_indices(result_shape):
            sample_count = 0
            mean = 0
            m2 = 0

            for inner_index in iter_indices(reduced_shape):
                source_index = merge_index(
                    self.ndim,
                    axes_set,
                    outer_index,
                    inner_index,
                )
                value = self._get_value(source_index)
                sample_count += 1
                delta = value - mean
                mean += delta / sample_count
                m2 += delta * (value - mean)

            result_buffer.append(m2 / count)

        if keepdims:
            result_shape = keepdims_shape(self.shape, axes_set)

        return self._make_reduction_result(
            result_buffer,
            result_shape,
            dtype=result_dtype_for_mean(self.dtype),
        )

    def _arg_reduce_axis(
        self,
        axis: int,
        is_better: Callable[[Any, Any], bool],
    ) -> Any | ndarray:
        """Argmin/argmax along a single axis with NaN propagation.

        Parameters
        ----------
        axis : int
            Normalized axis index (length must be non-zero).
        is_better : callable
            Comparison ``(candidate, best_value) -> bool``.

        Returns
        -------
        ndarray of int
            Shape ``self.shape`` with ``axis`` removed; each entry is the
            winning index along ``axis`` (0-based).

        Notes
        -----
        For each fixed index on the other axes, index 0 seeds both
        ``best_value`` and ``best_axis_index``. If that value is NaN, 0 is
        recorded and the scan stops. Otherwise later indices are visited;
        the first NaN sets ``best_axis_index`` to that index and breaks.
        Among non-NaN values, ``is_better`` updates the winner; ties keep
        the earlier index. Used by ``argmin`` and ``argmax``.
        """
        result_shape = self.shape[:axis] + self.shape[axis + 1 :]
        result_buffer: list[int] = []

        for outer_index in iter_indices(result_shape):
            first_source = outer_index[:axis] + (0,) + outer_index[axis:]
            best_value = self._get_value(first_source)
            best_axis_index = 0

            if _is_nan_value(best_value):
                result_buffer.append(0)
                continue

            for axis_index in range(1, self.shape[axis]):
                source_index = (
                    outer_index[:axis] + (axis_index,) + outer_index[axis:]
                )
                candidate = self._get_value(source_index)

                if _is_nan_value(candidate):
                    best_axis_index = axis_index
                    break

                if is_better(candidate, best_value):
                    best_value = candidate
                    best_axis_index = axis_index

            result_buffer.append(best_axis_index)

        return self._make_reduction_result(result_buffer, result_shape)

    def _arg_extremum_all(
        self,
        *,
        operation: str,
        is_better: Callable[[Any, Any], bool],
    ) -> int:
        """Flat argmin/argmax returning a single linear C-order index.

        Parameters
        ----------
        operation : str
            Name for error messages (``"argmin"`` or ``"argmax"``).
        is_better : callable
            Comparison ``(candidate, best_value) -> bool``.

        Returns
        -------
        int
            Linear index of the extremum in row-major flatten order.

        Raises
        ------
        ValueError
            If ``self.size == 0``.

        Notes
        -----
        Iterates ``iter_indices(self.shape)`` with an enumerated linear
        counter. The first NaN encountered is returned immediately. Among
        non-NaN values, ``is_better`` selects the winner; ties favor the
        lowest linear index.
        """
        if self.size == 0:
            raise empty_reduction(operation=operation)

        best_linear_index = 0
        best_value = None

        for linear_index, source_index in enumerate(iter_indices(self.shape)):
            candidate = self._get_value(source_index)

            if _is_nan_value(candidate):
                return linear_index

            if best_value is None or is_better(candidate, best_value):
                best_value = candidate
                best_linear_index = linear_index

        return best_linear_index

    def _cumulate_axis(
        self,
        axis: int,
        reducer: Callable[[Any, Any], Any],
    ) -> ndarray:
        """Cumulative scan along one axis, preserving input shape.

        Parameters
        ----------
        axis : int
            Normalized axis to scan (length must be non-zero).
        reducer : callable
            Binary function ``(running_total, value) -> new_total``;
            addition for ``cumsum``, multiplication for ``cumprod``.

        Returns
        -------
        ndarray
            Same shape and dtype as ``self``; position ``i`` along ``axis``
            stores the reduction of ``self[..., :i+1, ...]``.

        Notes
        -----
        Allocates a flat ``result_buffer`` of length ``self.size``. For each
        slice orthogonal to ``axis``, walks axis indices in order: the first
        element copies through without calling ``reducer``; subsequent
        positions apply ``reducer(total, value)``. Writes use
        ``flat_index`` for C-order placement. Used by ``cumsum`` and
        ``cumprod`` when ``axis`` is specified.
        """
        result_buffer = [default_fill_value(self.dtype)] * self.size
        outer_shape = self.shape[:axis] + self.shape[axis + 1 :]

        for outer_index in iter_indices(outer_shape):
            total = None

            for axis_index in range(self.shape[axis]):
                source_index = (
                    outer_index[:axis] + (axis_index,) + outer_index[axis:]
                )
                value = self._get_value(source_index)

                if total is None:
                    total = value
                else:
                    total = reducer(total, value)

                result_buffer[flat_index(source_index, self.shape)] = total

        return contiguous_ndarray(
            result_buffer,
            self.shape,
            dtype=self.dtype,
        )

    def _cumulate_flat(
        self,
        reducer: Callable[[Any, Any], Any],
        initial: Any,
    ) -> ndarray:
        """Cumulative reduction over flattened C-order elements.

        Parameters
        ----------
        reducer : callable
            Binary accumulator (add or multiply).
        initial : Any
            Value combined with the first element via ``reducer``; ``0`` for
            ``cumsum``, ``1`` for ``cumprod``.

        Returns
        -------
        ndarray
            1-D array of shape ``(self.size,)`` with running totals.

        Notes
        -----
        Used when ``cumsum``/``cumprod`` are called with ``axis is None``
        (array is logically flattened first). Each step applies ``reducer``
        to the running total and the next element from ``_iter_values``.
        """
        total = initial
        result_buffer: list[Any] = []

        for value in self._iter_values():
            total = reducer(total, value)
            result_buffer.append(total)

        return contiguous_ndarray(
            result_buffer,
            (self.size,),
            dtype=self.dtype,
        )


def _is_nan_value(value: Any) -> bool:
    """True for float/complex NaN; used by min/max/arg reductions to propagate NaN."""
    if isinstance(value, float):
        return math.isnan(value)

    if isinstance(value, complex):
        return cmath.isnan(value)

    return False
