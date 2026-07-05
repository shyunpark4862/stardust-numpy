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

"""Human-readable formatting for :meth:`ndarray.__repr__` and ``__str__``.

Implements R-style layout for multi-dimensional display, edge truncation,
and 80-character line wrapping. Unlike NumPy, ``repr`` and ``str`` follow
different conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from ._iter import iter_indices

if TYPE_CHECKING:
    from .ndarray import ndarray

REPR_FULL_DATA_SIZE = 12

EDGE_ITEMS = 3

MAX_LINE_WIDTH = 80

MIN_TRUNCATED_CELL_WIDTH = 4

ELLIPSIS = "..."

VECTOR_ROW_LABEL_WIDTH = len("[1]")

NUMERIC_DTYPES = frozenset({int, float, complex, bool})


def array_repr(array: ndarray) -> str:
    """Return the string representation of an array.

    Parameters
    ----------
    array : ndarray
        Array to represent.

    Returns
    -------
    str
        Debug-oriented string showing ``shape``, ``dtype``, and either
        full ``data`` (for small arrays) or a flat ``preview``.

    Notes
    -----
    Unlike NumPy's ``repr``, the format is ``array(shape=..., dtype=...,
    data=...)`` rather than ``array([...])``. Arrays with more than
    ``REPR_FULL_DATA_SIZE`` elements show a summarized flat preview
    instead of nested data.
    """
    shape = array.shape
    dtype_name = array.dtype.__name__

    if array.size == 0:
        return f"array(shape={shape!r}, dtype={dtype_name})"

    if array.ndim == 0:
        value = array._get_value(())
        return (
            f"array(shape=(), dtype={dtype_name}, "
            f"value={_format_element_repr_data(value)})"
        )

    parts = [f"shape={shape!r}", f"dtype={dtype_name}"]

    if array.size <= REPR_FULL_DATA_SIZE:
        parts.append(f"data={_format_repr_nested(array, ())}")
    else:
        parts.append(f"preview={_format_repr_flat(array)}")

    return f"array({', '.join(parts)})"


def array_str(array: ndarray) -> str:
    """Return a human-readable string for an array.

    Parameters
    ----------
    array : ndarray
        Array to format.

    Returns
    -------
    str
        R-style tabular layout for 1-D and 2-D arrays, block headers
        for higher dimensions, or a scalar string for 0-D arrays.

    Notes
    -----
    Unlike NumPy's ``str``, output uses R-style row/column labels
    (``[1]``, ``[1,]``, ``[,1]``) and wraps or truncates to fit
    ``MAX_LINE_WIDTH``. Empty arrays render as ``[1]`` for 1-D or a
    blank line for higher dimensions.
    """
    if array.ndim == 0:
        return _format_element_str(array._get_value(()), array.dtype)

    if array.size == 0:
        return _format_empty_str(array.shape)

    return _format_str_body(array)


def _format_repr_nested(array: ndarray, prefix: tuple[int, ...]) -> str:
    """Recursively build nested ``[...]`` repr data for ``array_repr``.

    Called by ``array_repr`` (via ``_format_repr_flat``'s sibling path) when
    the array is small enough to show full nested data.
    """
    if len(prefix) == array.ndim:
        return _format_element_repr_data(array._get_value(prefix))

    axis = len(prefix)
    items = [
        _format_repr_nested(array, prefix + (index,))
        for index in range(array.shape[axis])
    ]
    return "[" + ", ".join(items) + "]"


def _format_repr_flat(array: ndarray) -> str:
    """Build a flat ``[...]`` preview with edge truncation for large arrays.

    Used by ``array_repr`` when ``array.size > REPR_FULL_DATA_SIZE``.
    """
    values = [
        array._get_value(source_index)
        for source_index in iter_indices(array.shape)
    ]
    parts = [
        ELLIPSIS if index is None else _format_element_repr_data(values[index])
        for index in _summarized_indices(len(values))
    ]
    return "[" + ", ".join(parts) + "]"


def _format_element_repr_data(value: Any) -> str:
    """Format one scalar for ``array_repr`` (bools as ``True``/``False``)."""
    if isinstance(value, bool):
        return "True" if value else "False"

    return repr(value)


def _format_element_str(value: Any, dtype: type[Any]) -> str:
    """Format one scalar for ``array_str``, dispatching by ``dtype``."""
    if dtype is bool:
        return "True" if value else "False"

    if dtype is str:
        return repr(value)

    if dtype is float:
        return _format_float_str(value)

    if dtype is complex:
        return _format_complex_str(value)

    return str(value)


def _format_float_str(value: float) -> str:
    """Format a float for display (NaN, inf, ``.8g``, trailing dot rule)."""
    if value != value:
        return "nan"

    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"

    text = format(value, ".8g")

    if "e" in text or "E" in text:
        return text

    if "." not in text and "inf" not in text and "nan" not in text:
        return text + "."

    return text


def _format_complex_str(value: complex) -> str:
    """Format a complex number as ``(real+/-imagj)`` via ``_format_float_str``."""
    real = _format_float_str(value.real)
    imag = _format_float_str(abs(value.imag))

    if value.imag >= 0:
        return f"({real}+{imag}j)"

    return f"({real}-{imag}j)"


def _format_empty_str(shape: tuple[int, ...]) -> str:
    """Return ``[1]`` for empty 1-D or a blank line for higher dimensions."""
    if len(shape) <= 1:
        return "[1]"

    return "\n"


def _format_str_body(array: ndarray) -> str:
    """Route ``array_str`` to vector, matrix, or tensor formatters by ``ndim``."""
    if array.ndim == 1:
        return _format_r_vector(array, prefix=())

    if array.ndim == 2:
        return _format_r_matrix(array, prefix=())

    return _format_r_tensor(array)


def _format_r_tensor(array: ndarray) -> str:
    """Format arrays with ``ndim > 2`` as labeled 2-D blocks separated by blank lines.

    Parameters
    ----------
    array : ndarray
        Array with at least three dimensions.

    Returns
    -------
    str
        One or more blocks, each starting with an R-style outer index header
        (e.g. ``1, 2, ,``) followed by a formatted 2-D matrix slice.

    Notes
    -----
    Outer indices beyond the last two axes are summarized via
    ``_summarized_outer_indices``; ``None`` entries become standalone
    ``...`` lines between blocks. Each non-ellipsis block delegates to
    ``_format_r_matrix`` with the corresponding ``prefix`` tuple.
    """
    outer_shape = array.shape[:-2]
    blocks: list[str] = []

    for outer_index in _summarized_outer_indices(outer_shape):
        if outer_index is None:
            blocks.append(ELLIPSIS)
            continue

        header = _format_r_block_header(outer_index)
        matrix = _format_r_matrix(array, prefix=outer_index)
        blocks.append(f"{header}\n\n{matrix}")

    return "\n\n".join(blocks)


def _format_r_block_header(outer_index: tuple[int, ...]) -> str:
    """Build the R-style slice label (1-based indices plus trailing ``, ,``)."""
    labels = ", ".join(str(index + 1) for index in outer_index)
    return f"{labels}, ,"


def _format_r_vector(array: ndarray, *, prefix: tuple[int, ...]) -> str:
    """Format a 1-D slice as a single R-style row with ``[1]`` label.

    Parameters
    ----------
    array : ndarray
        Source array (may be a sub-slice of a higher-dimensional array).
    prefix : tuple of int
        Fixed index prefix selecting the 1-D axis to format.

    Returns
    -------
    str
        One line: ``[1]`` followed by space-separated, width-fitted cells.

    Notes
    -----
    All elements along the axis at ``len(prefix)`` are formatted and passed
    through ``_format_visible_cells`` so the row respects ``MAX_LINE_WIDTH``.
    """
    axis_size = array.shape[len(prefix)]
    formatted = [
        _format_element_str(
            array._get_value(prefix + (index,)),
            array.dtype,
        )
        for index in range(axis_size)
    ]
    cells = _format_visible_cells(
        formatted,
        dtype=array.dtype,
        row_label_width=VECTOR_ROW_LABEL_WIDTH,
    )
    return "[1] " + " ".join(cells)


def _format_r_matrix(array: ndarray, *, prefix: tuple[int, ...]) -> str:
    """Format a 2-D slice as an R-style matrix with row/column labels.

    Parameters
    ----------
    array : ndarray
        Source array (may be a sub-slice of a tensor when ``prefix`` is set).
    prefix : tuple of int
        Fixed index prefix selecting the two trailing axes as rows/columns.

    Returns
    -------
    str
        Multi-line string: column header row plus one line per visible row.
        Rows and columns may be truncated with ``...`` when they exceed
        ``MAX_LINE_WIDTH``.

    Notes
    -----
    Empty row or column count yields a blank row-label prefix only (via
    ``_row_prefix``). Non-empty matrices:

    1. Summarize row indices with ``_summarized_indices``.
    2. Build raw cell strings per row via ``_matrix_rows_by_index``.
    3. Compute column widths and visible column indices via
       ``_matrix_column_layout``.
    4. Render the ``[,j]`` header and each row (or an ellipsis row for
       ``None`` row indices) through ``_format_visible_cells``.
    """
    row_count = array.shape[len(prefix)]
    column_count = array.shape[len(prefix) + 1]

    if row_count == 0 or column_count == 0:
        return _row_prefix(_row_label_width(max(row_count, 1)))

    row_indices = _summarized_indices(row_count)
    row_label_width = _row_label_width(row_count)
    grid_by_row = _matrix_rows_by_index(
        array,
        prefix=prefix,
        row_indices=row_indices,
        column_count=column_count,
    )
    column_widths, column_indices = _matrix_column_layout(
        grid_by_row.values(),
        dtype=array.dtype,
        column_count=column_count,
        row_label_width=row_label_width,
    )
    header = _format_r_matrix_column_header(
        column_indices,
        column_widths,
        row_label_width,
    )
    rows = [
        _format_r_matrix_ellipsis_row(row_label_width)
        if row_index is None
        else _format_r_matrix_row(
            row_index,
            _format_visible_cells(
                grid_by_row[row_index],
                dtype=array.dtype,
                row_label_width=row_label_width,
                column_widths=column_widths,
                column_indices=column_indices,
            ),
            row_label_width,
        )
        for row_index in row_indices
    ]
    return "\n".join([header, *rows])


def _matrix_rows_by_index(
    array: ndarray,
    *,
    prefix: tuple[int, ...],
    row_indices: list[int | None],
    column_count: int,
) -> dict[int, list[str]]:
    """Map each visible row index to its unaligned string cells.

    Parameters
    ----------
    array : ndarray
        Source array.
    prefix : tuple of int
        Fixed index prefix for the 2-D slice.
    row_indices : list of int or None
        Row indices to include; ``None`` entries are skipped (ellipsis rows
        are handled separately by the caller).
    column_count : int
        Number of columns in the slice.

    Returns
    -------
    dict[int, list[str]]
        Keys are row indices; values are lists of ``_format_element_str``
        results, one per column, in column order.
    """
    display_rows = [
        row_index for row_index in row_indices if row_index is not None
    ]
    return {
        row_index: [
            _format_element_str(
                array._get_value(prefix + (row_index, column_index)),
                array.dtype,
            )
            for column_index in range(column_count)
        ]
        for row_index in display_rows
    }


def _matrix_column_layout(
    rows: Iterable[list[str]],
    *,
    dtype: type[Any],
    column_count: int,
    row_label_width: int,
) -> tuple[list[int], list[int | None]]:
    """Derive column widths and which columns fit on one line.

    Parameters
    ----------
    rows : iterable of list[str]
        Formatted cell strings per row (used to measure natural widths).
    dtype : type
        Element dtype; controls alignment in width estimation.
    column_count : int
        Total number of columns in the matrix.
    row_label_width : int
        Width reserved for the ``[i,]`` row label column.

    Returns
    -------
    column_widths : list[int]
        Per-column display width (capped to ``_max_content_width``).
    column_indices : list[int or None]
        Visible column indices; ``None`` marks an ellipsis gap.

    Notes
    -----
    Combines ``_column_widths``, ``_cap_column_widths``, and
    ``_select_column_indices`` so matrix header and body share the same
    layout.
    """
    column_widths = _cap_column_widths(
        _column_widths(list(rows), column_count),
        row_label_width,
    )
    column_indices = _select_column_indices(
        column_count,
        column_widths,
        dtype,
        row_label_width=row_label_width,
    )
    return column_widths, column_indices


def _format_visible_cells(
    formatted_row: list[str],
    *,
    dtype: type[Any],
    row_label_width: int,
    column_widths: list[int] | None = None,
    column_indices: list[int | None] | None = None,
) -> list[str]:
    """Pick, align, and truncate cells so a data row fits ``MAX_LINE_WIDTH``.

    Parameters
    ----------
    formatted_row : list[str]
        Unaligned string values for every column in the row.
    dtype : type
        Element dtype; numeric types are right-aligned, others left-aligned.
    row_label_width : int
        Width reserved for the row label prefix.
    column_widths : list[int], optional
        Precomputed column widths. When omitted, widths are derived from
        ``formatted_row`` and capped.
    column_indices : list[int or None], optional
        Visible column indices. When omitted, ``_select_column_indices``
        chooses them from widths and ``dtype``.

    Returns
    -------
    list[str]
        Final display cells (aligned and possibly truncated) ready to join
        with spaces after the row label.

    Notes
    -----
    Pipeline: optional width/index selection → ``_pick_cells`` →
    ``_align_data_cells`` → ``_fit_cells_to_line``. Used by
    ``_format_r_vector`` and ``_format_r_matrix``.
    """
    if column_widths is None:
        column_widths = _cap_column_widths(
            [len(text) for text in formatted_row],
            row_label_width,
        )

    if column_indices is None:
        column_indices = _select_column_indices(
            len(formatted_row),
            column_widths,
            dtype,
            row_label_width=row_label_width,
        )

    cells = _align_data_cells(
        _pick_cells(formatted_row, column_indices),
        column_widths,
        column_indices,
        dtype,
    )
    return _fit_cells_to_line(
        cells,
        column_widths=column_widths,
        column_indices=column_indices,
        max_content_width=_max_content_width(row_label_width),
    )


def _format_r_matrix_column_header(
    column_indices: list[int | None],
    column_widths: list[int],
    row_label_width: int,
) -> str:
    """Format the ``[,j]`` column header row for a matrix.

    Parameters
    ----------
    column_indices : list[int or None]
        Visible columns; ``None`` becomes ``...``.
    column_widths : list[int]
        Per-column widths (indexed by actual column number).
    row_label_width : int
        Width of the blank row-label prefix.

    Returns
    -------
    str
        Header line beginning with ``_row_prefix`` followed by aligned
        column labels.

    Notes
    -----
    If the initial header exceeds ``MAX_LINE_WIDTH``, labels are re-fitted
    via ``_fit_cells_to_line`` (same strategy as data rows).
    """
    headers: list[str] = []

    for column_index in column_indices:
        if column_index is None:
            headers.append(ELLIPSIS)
            continue

        header = f"[,{column_index + 1}]"
        width = min(
            max(column_widths[column_index], len(header)),
            _max_content_width(row_label_width),
        )
        headers.append(_truncate_display(header.rjust(width), width))

    header_line = _row_prefix(row_label_width) + " ".join(headers)
    if len(header_line) <= MAX_LINE_WIDTH:
        return header_line

    fitted_headers = _fit_cells_to_line(
        headers,
        column_widths=column_widths,
        column_indices=column_indices,
        max_content_width=_max_content_width(row_label_width),
    )
    return _row_prefix(row_label_width) + " ".join(fitted_headers)


def _format_r_matrix_ellipsis_row(row_label_width: int) -> str:
    """Return a matrix row consisting of only ``...`` after the label prefix."""
    return _row_prefix(row_label_width) + ELLIPSIS


def _format_r_matrix_row(
    row_index: int,
    cells: list[str],
    row_label_width: int,
) -> str:
    """Join ``[i,]`` row label with pre-formatted ``cells`` on one line."""
    row_label = f"[{row_index + 1},]"
    return row_label.ljust(row_label_width) + " " + " ".join(cells)


def _row_prefix(row_label_width: int) -> str:
    """Blank padding matching the row-label column; used for header/ellipsis rows."""
    return " " * row_label_width + " "


def _row_label_width(row_count: int) -> int:
    """Character width of the widest row label ``[{row_count},]``."""
    return len(f"[{row_count},]")


def _summarized_outer_indices(
    outer_shape: tuple[int, ...],
) -> list[tuple[int, ...] | None]:
    """Summarize multi-index tuples for tensor outer dimensions.

    Maps ``_summarized_indices`` over the flattened outer index space;
    ``None`` marks an ellipsis between blocks in ``_format_r_tensor``.
    """
    all_indices = list(iter_indices(outer_shape))
    return [
        None if index is None else all_indices[index]
        for index in _summarized_indices(len(all_indices))
    ]


def _pick_cells(
    row: list[str],
    column_indices: list[int | None],
) -> list[str]:
    """Subset ``row`` to visible columns, substituting ``...`` for ``None`` gaps."""
    return [
        ELLIPSIS if column_index is None else row[column_index]
        for column_index in column_indices
    ]


def _truncate_display(text: str, max_width: int) -> str:
    """Truncate ``text`` to ``max_width``, using ``...`` when ``max_width >= 4``."""
    if max_width <= 0:
        return ""

    if len(text) <= max_width:
        return text

    if max_width <= 3:
        return "." * max_width

    return text[: max_width - 3] + ELLIPSIS


def _max_content_width(row_label_width: int) -> int:
    """Usable cell area: ``MAX_LINE_WIDTH`` minus row label and one space."""
    return MAX_LINE_WIDTH - row_label_width - 1


def _cap_column_widths(
    column_widths: list[int],
    row_label_width: int,
) -> list[int]:
    """Clamp each column width to ``_max_content_width(row_label_width)``."""
    max_cell_width = _max_content_width(row_label_width)
    return [min(width, max_cell_width) for width in column_widths]


def _fit_cells_to_line(
    cells: list[str],
    *,
    column_widths: list[int],
    column_indices: list[int | None],
    max_content_width: int,
) -> list[str]:
    """Shrink cells iteratively until their joined width fits the line budget.

    Parameters
    ----------
    cells : list[str]
        Display strings (may already be aligned); one entry per visible column
        slot including ellipsis placeholders.
    column_widths : list[int]
        Full per-column width table (indexed by actual column number).
    column_indices : list[int or None]
        Maps each cell slot to a source column index; ``None`` for ellipsis
        slots (those cells are not truncated).
    max_content_width : int
        Maximum allowed width of ``" ".join(cells)`` (excludes row label).

    Returns
    -------
    list[str]
        Possibly shortened copy of ``cells`` that fits within
        ``max_content_width``, or the shortest achievable set if every cell
        is already at ``MIN_TRUNCATED_CELL_WIDTH``.

    Notes
    -----
    Each cell is first truncated to its nominal column width via
    ``_truncate_display``. While the row is still too wide, the widest
    non-ellipsis cell is shortened by one character at a time. The loop
    stops when the row fits, when no cell can shrink further, or when a
    cell would drop below ``MIN_TRUNCATED_CELL_WIDTH`` (prevents
    unreadable single-character columns).
    """
    fitted = [
        _truncate_display(cell, column_widths[column_indices[cell_index]])
        if column_indices[cell_index] is not None
        else cell
        for cell_index, cell in enumerate(cells)
    ]

    while fitted and len(" ".join(fitted)) > max_content_width:
        widest = max(range(len(fitted)), key=lambda index: len(fitted[index]))

        if len(fitted[widest]) <= MIN_TRUNCATED_CELL_WIDTH:
            break

        fitted[widest] = _truncate_display(
            fitted[widest], len(fitted[widest]) - 1
        )

    return fitted


def _line_width(row_label_width: int, cells: list[str]) -> int:
    """Total display width of a row: label + space + space-joined cells."""
    if not cells:
        return row_label_width

    return row_label_width + 1 + len(" ".join(cells))


def _select_column_indices(
    column_count: int,
    column_widths: list[int],
    dtype: type[Any],
    *,
    row_label_width: int,
) -> list[int | None]:
    """Choose which columns to show before horizontal truncation is applied.

    Parameters
    ----------
    column_count : int
        Total number of columns in the matrix or vector.
    column_widths : list[int]
        Display width assigned to each column (including header width).
    dtype : type
        Element dtype; passed to ``_prototype_cells`` for alignment-aware
        width estimation.
    row_label_width : int
        Width reserved for the ``[i,]`` row label.

    Returns
    -------
    list[int or None]
        Column indices to display in order. ``None`` marks a central
        ellipsis gap when edge columns alone do not fit.

    Notes
    -----
    Estimates line width using placeholder cells from ``_prototype_cells``
    (numeric columns right-padded with ``9`` digits, others left-padded).
    If all columns fit within ``MAX_LINE_WIDTH``, every index is returned.
    Otherwise the algorithm tries ``edge_items`` from ``EDGE_ITEMS`` down
    to 1 via ``_summarized_indices`` and picks the largest edge window
    that still fits. As a last resort, ``edge_items=1`` is used even when
    the line remains too wide (``_fit_cells_to_line`` handles overflow).
    """
    if column_count == 0:
        return []

    all_indices = list(range(column_count))

    if (
        _line_width(
            row_label_width, _prototype_cells(column_widths, all_indices, dtype)
        )
        <= MAX_LINE_WIDTH
    ):
        return all_indices

    for edge_items in range(EDGE_ITEMS, 0, -1):
        indices = _summarized_indices(column_count, edge_items=edge_items)
        if (
            _line_width(
                row_label_width,
                _prototype_cells(column_widths, indices, dtype),
            )
            <= MAX_LINE_WIDTH
        ):
            return indices

    return _summarized_indices(column_count, edge_items=1)


def _prototype_cells(
    column_widths: list[int],
    column_indices: list[int | None],
    dtype: type[Any],
) -> list[str]:
    """Build worst-case placeholder cells for line-width estimation.

    Parameters
    ----------
    column_widths : list[int]
        Per-column display widths.
    column_indices : list[int or None]
        Visible column slots; ``None`` becomes ``...``.
    dtype : type
        Numeric dtypes yield right-aligned ``9``-filled strings; others
        are left-aligned.

    Returns
    -------
    list[str]
        Placeholder cells matching the alignment rules of ``_align_data_cells``.

    Notes
    -----
    Used by ``_select_column_indices`` and ``_line_width`` so column
    selection accounts for padding without formatting real data.
    """
    cells: list[str] = []

    for column_index in column_indices:
        if column_index is None:
            cells.append(ELLIPSIS)
            continue

        width = column_widths[column_index]
        placeholder = "9" * width

        if dtype in NUMERIC_DTYPES:
            cells.append(placeholder.rjust(width))
        else:
            cells.append(placeholder.ljust(width))

    return cells


def _column_widths(
    grid: list[list[str]],
    column_count: int,
) -> list[int]:
    """Compute natural per-column widths from formatted grid rows.

    Parameters
    ----------
    grid : list[list[str]]
        Formatted cell strings for each visible row.
    column_count : int
        Number of columns in the full matrix.

    Returns
    -------
    list[int]
        For each column, the maximum of the cell length, the ``[,j]`` header
        width, and zero (empty grid yields an empty list when
        ``column_count == 0``).

    Notes
    -----
    Header width ``len(f"[,{j+1}]")`` ensures column labels fit even when
    all data cells are shorter.
    """
    if not grid:
        return []

    widths = [0] * column_count

    for row in grid:
        for column_index, cell in enumerate(row):
            header_width = len(f"[,{column_index + 1}]")
            widths[column_index] = max(
                widths[column_index],
                len(cell),
                header_width,
            )

    return widths


def _align_data_cells(
    row: list[str],
    column_widths: list[int],
    column_indices: list[int | None],
    dtype: type[Any],
) -> list[str]:
    """Pad or truncate cells to column width with dtype-appropriate alignment.

    Parameters
    ----------
    row : list[str]
        Cell strings in visible-column order (from ``_pick_cells``).
    column_widths : list[int]
        Full per-column width table.
    column_indices : list[int or None]
        Maps each cell to its source column; ``None`` for ellipsis slots.
    dtype : type
        Numeric dtypes (``NUMERIC_DTYPES``) are right-justified; others
        left-justified.

    Returns
    -------
    list[str]
        Fixed-width display strings. A single-element ``[ELLIPSIS]`` row
        passes through unchanged (whole-row ellipsis from row truncation).

    Notes
    -----
    Each non-ellipsis cell is truncated with ``_truncate_display`` before
    padding so over-wide values do not break column alignment.
    """
    if row == [ELLIPSIS]:
        return row

    aligned: list[str] = []

    for cell_index, cell in enumerate(row):
        if cell == ELLIPSIS:
            aligned.append(ELLIPSIS)
            continue

        column_index = column_indices[cell_index]

        if column_index is None:
            aligned.append(cell)
            continue

        width = column_widths[column_index]
        display = _truncate_display(cell, width)

        if dtype in NUMERIC_DTYPES:
            aligned.append(display.rjust(width))
        else:
            aligned.append(display.ljust(width))

    return aligned


def _summarized_indices(
    axis_size: int,
    *,
    edge_items: int = EDGE_ITEMS,
) -> list[int | None]:
    """Return leading and trailing indices with a central ellipsis gap.

    Parameters
    ----------
    axis_size : int
        Length of the axis being summarized (rows, columns, or flat preview).
    edge_items : int, optional
        Number of indices to keep at each end (default ``EDGE_ITEMS``).

    Returns
    -------
    list[int or None]
        Either ``range(axis_size)`` when ``axis_size <= 2 * edge_items``,
        or ``[0..edge-1, None, axis_size-edge..axis_size-1]``.

    Notes
    -----
    Shared by row/column truncation in matrices, flat ``array_repr``
    previews, and outer tensor block selection. ``None`` is rendered as
    ``...`` by callers.
    """
    if axis_size <= edge_items * 2:
        return list(range(axis_size))

    return (
        list(range(edge_items))
        + [None]
        + list(range(axis_size - edge_items, axis_size))
    )
