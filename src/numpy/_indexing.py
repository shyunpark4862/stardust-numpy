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

"""Multi-dimensional indexing and assignment for ``ndarray``.

:class:`IndexingMixin` implements ``__getitem__`` and ``__setitem__`` with
basic, slice, newaxis, integer fancy, and boolean mask indexing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._dtype import is_bool_dtype, is_int_dtype
from ._errors import (
    boolean_index_shape_mismatch,
    boolean_mask_must_be_bool,
    could_not_broadcast_assignment,
    fancy_index_must_be_int_or_bool,
    index_must_be_int,
    index_must_be_int_slice_or_ellipsis,
    index_out_of_bounds,
    integer_fancy_index_required,
    invalid_assignment_target,
    invalid_fancy_index,
    single_ellipsis_allowed,
    too_many_indices,
)
from ._factory import contiguous_ndarray, view_ndarray
from ._iter import iter_indices
from ._shape import is_c_contiguous, slice_length, unravel_index
from ._types import is_ndarray
from ._validation import coerce_to_ndarray, is_basic_array_like

if TYPE_CHECKING:
    from .ndarray import ndarray


class IndexingMixin:
    """Mixin providing ``__getitem__`` and ``__setitem__`` for ``ndarray``."""

    def __getitem__(
        self,
        index: Any,
    ) -> Any | ndarray:
        """
        Return an array element, sub-array, or view selected by ``index``.

        Parameters
        ----------
        index : int, slice, Ellipsis, None, tuple, or ndarray
            Index expression. ``None`` inserts a length-1 axis (newaxis).
            Integer and slice indexing return a scalar or a view when possible.
            Integer and boolean array indices trigger fancy indexing.

        Returns
        -------
        scalar or ndarray
            A scalar when every indexed dimension is removed by an integer
            index; otherwise an ``ndarray`` view or copy.

        Notes
        -----
        Unlike NumPy, only integer and boolean ``ndarray`` fancy indices are
        supported (no object or float index arrays). Boolean masks must match
        the shape of the indexed dimensions exactly. Multiple integer index
        arrays are broadcast together. When fancy index slots are not
        contiguous, the broadcast fancy shape is prepended to the result
        shape rather than replacing the indexed block inline.
        """
        normalized_index, fancy_shape, broadcasted_fancy = (
            self._prepare_fancy_index(index)
        )

        if broadcasted_fancy is not None:
            return self._getitem_integer_fancy_mixed(
                normalized_index,
                fancy_shape,
                broadcasted_fancy,
            )

        return self._getitem_basic_normalized(normalized_index)

    def __setitem__(
        self,
        index: Any,
        value: Any,
    ) -> None:
        """
        Assign ``value`` to the elements selected by ``index``.

        Parameters
        ----------
        index : int, slice, Ellipsis, None, tuple, or ndarray
            Index expression with the same rules as ``__getitem__``.
        value : scalar, sequence, or ndarray
            Value to store. Scalars broadcast to the indexed shape. Array
            values must broadcast to the assignment target shape.

        Notes
        -----
        Fancy assignment copies array values before writing so that overlapping
        source and destination buffers do not corrupt the read side.
        """
        normalized_index, fancy_shape, broadcasted_fancy = (
            self._prepare_fancy_index(index)
        )

        if broadcasted_fancy is not None:
            if is_ndarray(value) or is_basic_array_like(value):
                if not is_ndarray(value):
                    value = coerce_to_ndarray(
                        value,
                        name="assignment value",
                    )

                self._setitem_integer_fancy_array(
                    normalized_index,
                    fancy_shape,
                    broadcasted_fancy,
                    value,
                )
                return

            self._setitem_integer_fancy_scalar(
                normalized_index,
                fancy_shape,
                broadcasted_fancy,
                value,
            )
            return

        self._setitem_basic_normalized(normalized_index, value)

    def _normalize_int_index(
        self,
        index: int,
        axis: int,
    ) -> int:
        """Apply negative indexing and bounds checks for one axis.

        Called by basic getitem/setitem and fancy index iteration.
        """
        if not isinstance(index, int):
            raise index_must_be_int()

        axis_size = self.shape[axis]

        if index < 0:
            index += axis_size

        if index < 0 or index >= axis_size:
            raise index_out_of_bounds(index=index, axis=axis, size=axis_size)

        return index

    def _buffer_position(
        self,
        source_index: tuple[int, ...],
    ) -> int:
        """Map a multi-dimensional index to a flat buffer offset.

        Called by ``_get_value``, ``_set_value``, and assignment helpers.
        """
        buffer_offset = self.offset

        for axis, index in enumerate(source_index):
            buffer_offset += index * self.strides[axis]

        return buffer_offset

    def _get_value(
        self,
        source_index: tuple[int, ...],
    ) -> Any:
        """Read one element at ``source_index``.

        Called by indexing, reduction, and fancy getitem paths.
        """
        return self.buffer[self._buffer_position(source_index)]

    def _set_value(
        self,
        source_index: tuple[int, ...],
        value: Any,
    ) -> None:
        """Write one element at ``source_index``.

        Called by ``_write_flat_values`` and basic scalar setitem.
        """
        self.buffer[self._buffer_position(source_index)] = value

    def _is_c_contiguous(self) -> bool:
        """Return whether the array layout is C-contiguous.

        Called by indexing, linalg, and ``astype`` fast paths.
        """
        from ._shape import is_c_contiguous

        return is_c_contiguous(self.shape, self.strides)

    def _flat_buffer_slice(self) -> list[Any]:
        """Return logical elements in C-order.

        Called by ``astype``, ``copy``, and boolean mask conversion.
        """
        if self._is_c_contiguous():
            return self.buffer[self.offset : self.offset + self.size]

        return [
            self._get_value(source_index)
            for source_index in iter_indices(self.shape)
        ]

    def _iter_values(self):
        """Yield elements in C-order.

        Called by reduction, linalg, and ``astype`` helpers.
        """
        if self._is_c_contiguous():
            buffer = self.buffer
            start = self.offset
            end = start + self.size

            for position in range(start, end):
                yield buffer[position]
            return

        for source_index in iter_indices(self.shape):
            yield self._get_value(source_index)

    def _write_flat_values(
        self,
        values: list[Any],
    ) -> None:
        """Write C-order flat values back into the array buffer.

        Called by reduction helpers that materialize results in place.
        """
        if self._is_c_contiguous():
            buffer = self.buffer
            start = self.offset

            for offset, value in enumerate(values):
                buffer[start + offset] = value

            return

        for flat_index, value in enumerate(values):
            self._set_value(unravel_index(flat_index, self.shape), value)

    def _coerce_index_array(
        self,
        index: Any,
    ) -> Any:
        """Coerce a fancy index operand to ``ndarray``.

        Called by fancy index preparation; raises if the operand is invalid.
        """
        if is_ndarray(index) or is_basic_array_like(index):
            return coerce_to_ndarray(index, name="fancy index")

        raise invalid_fancy_index(got=index)

    def _is_integer_index_array(
        self,
        index: ndarray,
    ) -> bool:
        """Return whether ``index`` is an integer-dtype fancy index array.

        Called by ``_collect_integer_fancy_indices``.
        """
        return is_int_dtype(index.dtype)

    def _is_boolean_index_array(
        self,
        index: ndarray,
    ) -> bool:
        """Return whether ``index`` is a boolean-dtype mask array.

        Called by ``_expand_boolean_masks`` and index collection.
        """
        return is_bool_dtype(index.dtype)

    def _expand_ellipsis(
        self,
        index: tuple[Any, ...],
        used_index_slots: int,
    ) -> tuple[Any, ...]:
        """Expand ``...`` and trailing missing axes to a full-rank index tuple.

        Called by ``_normalize_mixed_index``.
        """
        ellipsis_count = sum(item is Ellipsis for item in index)

        if ellipsis_count > 1:
            raise single_ellipsis_allowed()

        if used_index_slots > self.ndim:
            raise too_many_indices(ndim=self.ndim, index_count=used_index_slots)

        if ellipsis_count == 1:
            missing_slots = self.ndim - used_index_slots
            expanded: list[Any] = []

            for item in index:
                if item is Ellipsis:
                    expanded.extend([slice(None)] * missing_slots)
                else:
                    expanded.append(item)

            return tuple(expanded)

        return index + (slice(None),) * (self.ndim - used_index_slots)

    def _normalize_mixed_index(
        self,
        index: Any,
    ) -> tuple[Any, ...]:
        """Normalize a user index to a tuple with ellipsis and newaxis expanded.

        Called by ``_prepare_fancy_index``.
        """
        if not isinstance(index, tuple):
            index = (index,)

        def count_index_slots(item: Any) -> int:
            if item is None or item is Ellipsis:
                return 0

            if isinstance(item, (int, slice)):
                return 1

            fancy = self._coerce_index_array(item)

            if is_bool_dtype(fancy.dtype):
                return fancy.ndim

            return 1

        used_index_slots = sum(
            count_index_slots(item) for item in index if item is not Ellipsis
        )

        return self._expand_ellipsis(index, used_index_slots)

    def _collect_integer_fancy_indices(
        self,
        normalized_index: tuple[Any, ...],
    ) -> list[tuple[int, ndarray]]:
        """Collect integer fancy index arrays and their axis slots.

        Called by ``_prepare_fancy_index``; boolean masks are skipped.
        """
        fancy_indices: list[tuple[int, ndarray]] = []

        for index_slot, item in enumerate(normalized_index):
            if isinstance(item, (int, slice)) or item is None:
                continue

            fancy = self._coerce_index_array(item)

            if self._is_boolean_index_array(fancy):
                continue

            if not self._is_integer_index_array(fancy):
                raise fancy_index_must_be_int_or_bool()

            fancy_indices.append((index_slot, fancy))

        return fancy_indices

    def _broadcast_integer_fancy_indices(
        self,
        fancy_indices: list[tuple[int, ndarray]],
    ) -> tuple[tuple[int, ...], dict[int, ndarray]]:
        """Broadcast integer fancy arrays to a common shape.

        Called by ``_prepare_fancy_index``.
        """
        if not fancy_indices:
            return (), {}

        from .broadcasting import broadcast_shape

        fancy_shape = broadcast_shape(
            *(index.shape for _, index in fancy_indices)
        )

        broadcasted = {
            index_slot: index.broadcast_to(fancy_shape)
            for index_slot, index in fancy_indices
        }

        return fancy_shape, broadcasted

    def _fancy_indices_are_contiguous(
        self,
        broadcasted_fancy: dict[int, ndarray],
    ) -> bool:
        """Return whether fancy index slots occupy a contiguous axis block.

        Called by result-shape computation and fancy index iteration.
        """
        fancy_index_slots = tuple(broadcasted_fancy)
        first = fancy_index_slots[0]
        last = fancy_index_slots[-1]

        return all(
            index_slot in broadcasted_fancy
            for index_slot in range(first, last + 1)
        )

    def _integer_fancy_result_shape(
        self,
        normalized_index: tuple[Any, ...],
        fancy_shape: tuple[int, ...],
        broadcasted_fancy: dict[int, ndarray],
    ) -> tuple[int, ...]:
        """Compute the output shape for integer fancy indexing.

        Parameters
        ----------
        normalized_index : tuple
            Index tuple after ellipsis, newaxis, and boolean-mask expansion.
            May contain ``int``, ``slice``, ``None``, and integer ``ndarray``
            entries.
        fancy_shape : tuple of int
            Broadcast shape shared by all integer fancy index arrays.
        broadcasted_fancy : dict[int, ndarray]
            Mapping from index slot to the broadcast fancy index array for
            that slot.

        Returns
        -------
        tuple of int
            Shape of the fancy-index result. Integer indices remove their
            source axis from the shape; ``None`` inserts a length-1 axis;
            slices contribute their output length.

        Notes
        -----
        When fancy index slots form a contiguous block in the normalized
        index tuple, ``fancy_shape`` replaces that block inline::

            shapes_before + fancy_shape + shapes_after

        When fancy slots are non-contiguous (for example, an integer array
        at axis 0 and another at axis 2), the broadcast fancy shape is
        prepended instead::

            fancy_shape + shapes_before + shapes_after

        This differs from NumPy, which always places the broadcast fancy
        shape at the position of the indexed dimensions. Empty fancy index
        dictionaries are rejected via :func:`integer_fancy_index_required`.
        """
        if not broadcasted_fancy:
            raise integer_fancy_index_required()

        fancy_index_slots = tuple(broadcasted_fancy)
        first_fancy_slot = fancy_index_slots[0]

        fancy_are_contiguous = self._fancy_indices_are_contiguous(
            broadcasted_fancy
        )

        source_axis = 0
        shapes_before: list[int] = []
        shapes_after: list[int] = []

        for index_slot, item in enumerate(normalized_index):
            shape_segment = (
                shapes_before if index_slot < first_fancy_slot else shapes_after
            )

            if item is None:
                shape_segment.append(1)
                continue

            if isinstance(item, int):
                source_axis += 1
                continue

            if isinstance(item, slice):
                start, stop, step = item.indices(self.shape[source_axis])
                shape_segment.append(slice_length(start, stop, step))
                source_axis += 1
                continue

            source_axis += 1

        if fancy_are_contiguous:
            return tuple(shapes_before + list(fancy_shape) + shapes_after)

        return tuple(list(fancy_shape) + shapes_before + shapes_after)

    def _getitem_integer_fancy_mixed(
        self,
        normalized_index: tuple[Any, ...],
        fancy_shape: tuple[int, ...],
        broadcasted_fancy: dict[int, ndarray],
    ) -> Any | ndarray:
        """Gather elements via integer (and expanded boolean) fancy indexing.

        Parameters
        ----------
        normalized_index : tuple
            Fully normalized index tuple produced by
            :meth:`_prepare_fancy_index`. Boolean masks have already been
            converted to per-axis integer coordinate arrays.
        fancy_shape : tuple of int
            Broadcast shape of the integer fancy index arrays.
        broadcasted_fancy : dict[int, ndarray]
            Mapping from index slot to broadcast integer index array.

        Returns
        -------
        scalar or ndarray
            A scalar when the computed result shape is ``()``; otherwise a
            new C-contiguous ``ndarray`` containing the gathered values.

        Notes
        -----
        Called by ``__getitem__`` when fancy indices are present. Unlike
        basic indexing, fancy getitem always copies elements into a new
        buffer rather than returning a view. Overlapping source indices are
        read independently for each result element (no deduplication).
        """
        result_shape = self._integer_fancy_result_shape(
            normalized_index,
            fancy_shape,
            broadcasted_fancy,
        )

        result_buffer = [
            self._get_value(source_index)
            for source_index in self._iter_fancy_source_indices(
                normalized_index,
                fancy_shape,
                broadcasted_fancy,
            )
        ]

        if result_shape == ():
            return result_buffer[0]

        return contiguous_ndarray(
            result_buffer,
            result_shape,
            dtype=self.dtype,
        )

    def _boolean_mask_to_integer_indices(
        self,
        mask: ndarray,
    ) -> tuple[ndarray, ...]:
        """Convert a boolean mask to per-axis integer coordinate arrays.

        Parameters
        ----------
        mask : ndarray
            Boolean array whose ``True`` entries select source coordinates.
            Must have boolean dtype.

        Returns
        -------
        tuple of ndarray
            One integer coordinate array per mask dimension. Each array lists
            the source-axis index for every ``True`` entry, in C-order
            traversal of the mask. All returned arrays have the same length
            (the number of ``True`` values).

        Notes
        -----
        Called by :meth:`_expand_boolean_masks` before integer fancy
        broadcasting. A mask with no ``True`` entries yields empty integer
        arrays (one per dimension). Unlike NumPy, boolean indexing that
        would produce a 1-D result is not supported directly; masks are
        always expanded into separate per-axis integer arrays that participate
        in the integer fancy path. Uses a contiguous fast path when the mask
        layout is C-contiguous.
        """
        from .creation import array

        if not is_bool_dtype(mask.dtype):
            raise boolean_mask_must_be_bool(got=mask.dtype)

        coordinates = [[] for _ in range(mask.ndim)]

        if mask._is_c_contiguous():
            for flat_index, value in enumerate(mask._flat_buffer_slice()):
                if not value:
                    continue

                mask_index = unravel_index(flat_index, mask.shape)

                for axis, coordinate in enumerate(mask_index):
                    coordinates[axis].append(coordinate)
        else:
            for mask_index in iter_indices(mask.shape):
                if not mask._get_value(mask_index):
                    continue

                for axis, coordinate in enumerate(mask_index):
                    coordinates[axis].append(coordinate)

        return tuple(
            array(axis_coordinates, dtype=int)
            for axis_coordinates in coordinates
        )

    def _expected_boolean_mask_shape(
        self,
        source_axis: int,
        mask_ndim: int,
    ) -> tuple[int, ...]:
        """Return the shape a boolean mask must have for ``source_axis``.

        Called by ``_expand_boolean_masks`` for shape validation.
        """
        return self.shape[source_axis : source_axis + mask_ndim]

    def _expand_boolean_masks(
        self,
        normalized_index: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Replace boolean mask entries with integer coordinate arrays.

        Parameters
        ----------
        normalized_index : tuple
            Index tuple after ellipsis and newaxis expansion from
            :meth:`_normalize_mixed_index`. Entries may be ``int``, ``slice``,
            ``None``, integer ``ndarray``, or boolean ``ndarray``.

        Returns
        -------
        tuple
            Index tuple of the same logical rank, where each boolean mask
            occupying ``k`` source axes is replaced by ``k`` integer
            coordinate arrays (one per mask dimension).

        Raises
        ------
        ValueError
            If a boolean mask spans more axes than remain in the source
            array, or if its shape does not exactly match the indexed
            source dimensions.

        Notes
        -----
        Called by :meth:`_prepare_fancy_index` before integer fancy index
        collection. A single boolean mask with ``ndim > 1`` consumes multiple
        consecutive source axes and expands into multiple index slots. Unlike
        NumPy, boolean masks must match the shape of the indexed dimensions
        exactly (no broadcasting of mismatched boolean arrays). After
        expansion, boolean entries are indistinguishable from explicit
        integer fancy indices and follow the same broadcast and result-shape
        rules.
        """
        expanded: list[Any] = []
        source_axis = 0

        for item in normalized_index:
            if item is None:
                expanded.append(item)
                continue

            if isinstance(item, (int, slice)):
                expanded.append(item)
                source_axis += 1
                continue

            fancy = self._coerce_index_array(item)

            if not self._is_boolean_index_array(fancy):
                expanded.append(fancy)
                source_axis += 1
                continue

            end_axis = source_axis + fancy.ndim

            if end_axis > self.ndim:
                raise too_many_indices(ndim=self.ndim, index_count=end_axis)

            expected_shape = self._expected_boolean_mask_shape(
                source_axis,
                fancy.ndim,
            )

            if fancy.shape != expected_shape:
                raise boolean_index_shape_mismatch(
                    got=fancy.shape,
                    expected=expected_shape,
                )

            expanded.extend(self._boolean_mask_to_integer_indices(fancy))

            source_axis += fancy.ndim

        return tuple(expanded)

    def _getitem_basic_normalized(
        self,
        normalized_index: tuple[Any, ...],
    ) -> Any | ndarray:
        """Basic (int/slice/newaxis) getitem; returns a scalar or view.

        Called by ``__getitem__`` and ``_setitem_basic_normalized``.
        """
        offset = self.offset
        new_shape: list[int] = []
        new_strides: list[int] = []
        source_axis = 0

        for item in normalized_index:
            if item is None:
                new_shape.append(1)
                new_strides.append(0)
                continue

            if isinstance(item, int):
                normalized = self._normalize_int_index(item, source_axis)
                offset += normalized * self.strides[source_axis]
                source_axis += 1
                continue

            if isinstance(item, slice):
                start, stop, step = item.indices(self.shape[source_axis])
                length = slice_length(start, stop, step)

                offset += start * self.strides[source_axis]
                new_shape.append(length)
                new_strides.append(self.strides[source_axis] * step)
                source_axis += 1
                continue

            raise index_must_be_int_slice_or_ellipsis()

        if len(new_shape) == 0:
            return self.buffer[offset]

        return view_ndarray(
            self.buffer,
            tuple(new_shape),
            tuple(new_strides),
            offset=offset,
            dtype=self.dtype,
        )

    def _prepare_fancy_index(
        self,
        index: Any,
    ) -> tuple[
        tuple[Any, ...], tuple[int, ...] | None, dict[int, ndarray] | None
    ]:
        """Normalize an index and detect broadcast integer fancy indices.

        Parameters
        ----------
        index : int, slice, Ellipsis, None, tuple, or ndarray
            Raw user index passed to ``__getitem__`` or ``__setitem__``.

        Returns
        -------
        normalized_index : tuple
            Fully expanded index tuple with ellipsis filled in, boolean masks
            converted to integer coordinate arrays, and trailing missing axes
            padded with ``slice(None)``.
        fancy_shape : tuple of int or None
            Broadcast shape of integer fancy index arrays, or ``None`` when
            no integer fancy indices are present.
        broadcasted_fancy : dict[int, ndarray] or None
            Mapping from index slot to broadcast integer index array, or
            ``None`` when the index contains only basic (int/slice/newaxis)
            entries.

        Notes
        -----
        Called by ``__getitem__`` and ``__setitem__`` as the first step of
        every index operation. Processing order is:

        1. :meth:`_normalize_mixed_index` — tuple coercion, ellipsis, newaxis.
        2. :meth:`_expand_boolean_masks` — boolean arrays to integer coords.
        3. :meth:`_collect_integer_fancy_indices` — gather integer arrays.
        4. :meth:`_broadcast_integer_fancy_indices` — broadcast when needed.

        When ``broadcasted_fancy`` is ``None``, callers should route to the
        basic indexing path. Only integer and boolean ``ndarray`` fancy
        indices are supported; float or object index arrays are rejected
        earlier during coercion or dtype checks.
        """
        normalized_index = self._normalize_mixed_index(index)
        normalized_index = self._expand_boolean_masks(normalized_index)
        fancy_indices = self._collect_integer_fancy_indices(normalized_index)

        if not fancy_indices:
            return normalized_index, None, None

        fancy_shape, broadcasted_fancy = self._broadcast_integer_fancy_indices(
            fancy_indices
        )

        return normalized_index, fancy_shape, broadcasted_fancy

    def _iter_fancy_source_indices(
        self,
        normalized_index: tuple[Any, ...],
        fancy_shape: tuple[int, ...],
        broadcasted_fancy: dict[int, ndarray],
    ):
        """Yield source indices for each element of a fancy-index result.

        Parameters
        ----------
        normalized_index : tuple
            Fully normalized index tuple from :meth:`_prepare_fancy_index`.
        fancy_shape : tuple of int
            Broadcast shape of the integer fancy index arrays.
        broadcasted_fancy : dict[int, ndarray]
            Mapping from index slot to broadcast integer index array.

        Yields
        ------
        tuple of int
            Source-axis coordinates into ``self`` for one result element,
            in C-order over the fancy result shape.

        Notes
        -----
        Called by fancy getitem and setitem helpers. Result coordinates are
        mapped back to source coordinates by combining:

        - Integer index entries (fixed per yield).
        - Slice entries (offset by basic result indices times step).
        - Fancy index entries (looked up from ``broadcasted_fancy`` at the
          broadcast coordinates for the current result element).

        When fancy slots are contiguous, the fancy shape segment sits inline
        within the result index tuple (between basic axes before and after
        the fancy block). When fancy slots are non-contiguous, the fancy
        shape leads the result index and basic slice/newaxis coordinates
        follow. Negative fancy indices are normalized via
        :meth:`_normalize_int_index` on each lookup.
        """
        result_shape = self._integer_fancy_result_shape(
            normalized_index,
            fancy_shape,
            broadcasted_fancy,
        )

        fancy_index_slots = tuple(broadcasted_fancy)
        first_fancy_slot = fancy_index_slots[0]

        fancy_are_contiguous = self._fancy_indices_are_contiguous(
            broadcasted_fancy
        )

        before_basic_slots = [
            index_slot
            for index_slot, item in enumerate(normalized_index)
            if index_slot < first_fancy_slot
            and (item is None or isinstance(item, slice))
        ]

        after_basic_slots = [
            index_slot
            for index_slot, item in enumerate(normalized_index)
            if index_slot >= first_fancy_slot
            and (item is None or isinstance(item, slice))
        ]

        basic_index_slots = before_basic_slots + after_basic_slots

        for result_index in iter_indices(result_shape):
            if fancy_are_contiguous:
                # Fancy shape replaces the contiguous index block inline.
                before_count = len(before_basic_slots)

                fancy_result_coords = result_index[
                    before_count : before_count + len(fancy_shape)
                ]

                basic_result_indices = (
                    result_index[:before_count]
                    + result_index[before_count + len(fancy_shape) :]
                )
            else:
                # Non-contiguous fancy slots: fancy shape leads the result.
                fancy_result_coords = result_index[: len(fancy_shape)]
                basic_result_indices = result_index[len(fancy_shape) :]

            basic_index_map = dict(zip(basic_index_slots, basic_result_indices))

            source_index: list[int] = []
            source_axis = 0

            for index_slot, item in enumerate(normalized_index):
                if item is None:
                    continue

                if isinstance(item, int):
                    source_index.append(
                        self._normalize_int_index(item, source_axis)
                    )
                    source_axis += 1
                    continue

                if isinstance(item, slice):
                    start, _, step = item.indices(self.shape[source_axis])
                    source_index.append(
                        start + basic_index_map[index_slot] * step
                    )
                    source_axis += 1
                    continue

                fancy = broadcasted_fancy[index_slot]
                source_index.append(
                    self._normalize_int_index(
                        fancy[fancy_result_coords],
                        source_axis,
                    )
                )
                source_axis += 1

            yield tuple(source_index)

    def _assign_scalar(
        self,
        value: Any,
    ) -> None:
        """Broadcast a scalar to every element of ``self``.

        Called by ``_setitem_basic_normalized`` for scalar assignment targets.
        """
        for source_index in iter_indices(self.shape):
            self.buffer[self._buffer_position(source_index)] = value

    def _assign_array(
        self,
        value: ndarray,
    ) -> None:
        """Assign an array after broadcasting to ``self.shape``.

        Called by ``_setitem_basic_normalized`` for array assignment targets.
        """
        try:
            value = value.broadcast_to(self.shape)
        except ValueError as error:
            raise could_not_broadcast_assignment(
                from_shape=value.shape,
                to_shape=self.shape,
            ) from error

        for source_index in iter_indices(self.shape):
            self.buffer[self._buffer_position(source_index)] = value[
                source_index
            ]

    def _setitem_integer_fancy_scalar(
        self,
        normalized_index: tuple[Any, ...],
        fancy_shape: tuple[int, ...],
        broadcasted_fancy: dict[int, ndarray],
        value: Any,
    ) -> None:
        """Fancy assignment of a scalar to each selected element.

        Called by ``__setitem__`` when the value is not an array.
        """
        for source_index in self._iter_fancy_source_indices(
            normalized_index,
            fancy_shape,
            broadcasted_fancy,
        ):
            self.buffer[self._buffer_position(source_index)] = value

    def _setitem_integer_fancy_array(
        self,
        normalized_index: tuple[Any, ...],
        fancy_shape: tuple[int, ...],
        broadcasted_fancy: dict[int, ndarray],
        value: ndarray,
    ) -> None:
        """Fancy assignment from a broadcast array.

        Called by ``__setitem__``; copies ``value`` before writing to avoid
        read/write overlap when source and destination buffers alias.
        """
        result_shape = self._integer_fancy_result_shape(
            normalized_index,
            fancy_shape,
            broadcasted_fancy,
        )

        try:
            value = value.broadcast_to(result_shape).copy()
        except ValueError as error:
            raise could_not_broadcast_assignment(
                from_shape=value.shape,
                to_shape=result_shape,
            ) from error

        for source_index, result_index in zip(
            self._iter_fancy_source_indices(
                normalized_index,
                fancy_shape,
                broadcasted_fancy,
            ),
            iter_indices(result_shape),
        ):
            self.buffer[self._buffer_position(source_index)] = value[
                result_index
            ]

    def _setitem_basic_normalized(
        self,
        normalized_index: tuple[Any, ...],
        value: Any,
    ) -> None:
        """Basic (int/slice/newaxis) assignment via view or scalar write.

        Called by ``__setitem__`` when no fancy indices are present.
        """
        is_scalar_index = all(
            isinstance(item, int) for item in normalized_index
        )

        if is_scalar_index:
            source_index = tuple(
                self._normalize_int_index(item, axis)
                for axis, item in enumerate(normalized_index)
            )
            self.buffer[self._buffer_position(source_index)] = value
            return

        assignment_view = self._getitem_basic_normalized(normalized_index)

        if not is_ndarray(assignment_view):
            raise invalid_assignment_target()

        if is_ndarray(value):
            assignment_view._assign_array(value)
        elif is_basic_array_like(value):
            assignment_view._assign_array(
                coerce_to_ndarray(value, name="assignment value")
            )
        else:
            assignment_view._assign_scalar(value)
