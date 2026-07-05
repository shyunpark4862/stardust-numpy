"""Basic indexing: integers, slices, views, and ellipsis."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_1d_integer_index_supports_negative_indices():
    a = array([10, 20, 30])

    assert a[0] == 10
    assert a[1] == 20
    assert a[-1] == 30


def test_1d_integer_index_raises_for_out_of_range():
    a = array([10, 20, 30])

    with pytest.raises(IndexError):
        a[3]

    with pytest.raises(IndexError):
        a[-4]


def test_basic_slice_returns_view():
    a = array([10, 20, 30, 40, 50])

    b = a[1:4]

    assert b.tolist() == [20, 30, 40]
    assert b.shape == (3,)
    assert b.strides == (1,)
    assert b.offset == 1

    b[0] = 999

    assert a.tolist() == [10, 999, 30, 40, 50]


def test_slice_with_step():
    a = array([10, 20, 30, 40, 50, 60])

    b = a[1:6:2]

    assert b.tolist() == [20, 40, 60]
    assert b.strides == (2,)


def test_reverse_slice():
    a = array([10, 20, 30, 40])

    b = a[::-1]

    assert b.tolist() == [40, 30, 20, 10]
    assert b.strides == (-1,)

    b[0] = 999

    assert a.tolist() == [10, 20, 30, 999]


def test_2d_integer_indexing():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a[0, 0] == 1
    assert a[0, 2] == 3
    assert a[1, 0] == 4
    assert a[1, 2] == 6


def test_2d_negative_indexing():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a[-1, -1] == 6
    assert a[-2, -3] == 1


def test_partial_integer_index_returns_view():
    a = array([[1, 2, 3], [4, 5, 6]])

    row = a[1]

    assert row.tolist() == [4, 5, 6]
    assert row.shape == (3,)
    assert row.strides == (1,)
    assert row.offset == 3
    assert row.buffer is a.buffer


def test_partial_integer_index_view_modifies_original():
    a = array([[1, 2, 3], [4, 5, 6]])

    row = a[1]
    row[0] = 99

    assert a.tolist() == [
        [1, 2, 3],
        [99, 5, 6],
    ]


def test_3d_partial_integer_index():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    view = a[1, 0]

    assert view.tolist() == [5, 6]
    assert view.shape == (2,)
    assert view.offset == 4


def test_column_slice_returns_view():
    a = array([[1, 2, 3], [4, 5, 6]])

    column = a[:, 1]

    assert column.tolist() == [2, 5]
    assert column.shape == (2,)
    assert column.strides == (3,)
    assert column.offset == 1
    assert column.buffer is a.buffer


def test_mixed_integer_and_slice_indexing():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a[1, 1:]

    assert result.tolist() == [5, 6]
    assert result.shape == (2,)
    assert result.strides == (1,)
    assert result.offset == 4


def test_reverse_columns():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a[:, ::-1]

    assert result.tolist() == [
        [3, 2, 1],
        [6, 5, 4],
    ]
    assert result.strides == (3, -1)


def test_slice_view_modifies_original():
    a = array([[1, 2, 3], [4, 5, 6]])

    view = a[:, 1]
    view[0] = 99

    assert a.tolist() == [
        [1, 99, 3],
        [4, 5, 6],
    ]


def test_operation_on_non_contiguous_view_creates_contiguous_result():
    a = array([[1, 2, 3], [4, 5, 6]])

    column = a[:, 1]
    result = column + 10

    assert result.tolist() == [12, 15]
    assert result.strides == (1,)
    assert result.buffer == [12, 15]


def test_ellipsis_indexing():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a[..., 1].tolist() == [2, 5]
    assert a[1, ...].tolist() == [4, 5, 6]
    assert a[...,].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_newaxis_indexing():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a[:, None]

    assert result.shape == (2, 1, 3)
    assert result.strides == (3, 0, 1)
    assert result.tolist() == [
        [[1, 2, 3]],
        [[4, 5, 6]],
    ]


def test_newaxis_view_shares_buffer():
    a = array([[1, 2, 3], [4, 5, 6]])

    view = a[:, None]
    view[0, 0, 1] = 99

    assert a.tolist() == [
        [1, 99, 3],
        [4, 5, 6],
    ]


def test_add_where_with_out():
    a = array([1, 2, 3])
    out = array([100, 100, 100])

    result = add(
        a,
        10,
        out=out,
        where=array([True, False, True]),
    )

    assert result is out
    assert out.tolist() == [11, 100, 13]


def test_add_where_without_out():
    result = add(
        array([1, 2, 3]),
        10,
        where=array([True, False, True]),
    )

    assert result.tolist() == [11, 0, 13]


def test_index_rejects_non_integer():
    a = array([10, 20, 30])

    with pytest.raises(TypeError):
        a[1.5]


def test_index_rejects_multiple_ellipsis():
    a = array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(IndexError):
        a[..., ..., 0]


def test_index_rejects_too_many_indices():
    a = array([1, 2, 3])

    with pytest.raises(IndexError):
        a[0, 0]


def test_index_rejects_invalid_index_type():
    a = array([1, 2, 3])

    with pytest.raises(TypeError):
        a["bad"]


def test_normalize_int_index_rejects_non_int():
    with pytest.raises(TypeError, match="must be an integer"):
        array([1, 2, 3])._normalize_int_index(1.5, 0)
