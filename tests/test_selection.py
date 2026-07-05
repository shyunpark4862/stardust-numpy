"""Selection utilities: where, nonzero, take, clip."""

from __future__ import annotations

import pytest

from numpy import *  # noqa: F403


def test_where_selects_from_x_and_y():
    condition = array([True, False, True, False])
    x = array([10, 20, 30, 40])
    y = array([1, 2, 3, 4])

    assert where(condition, x, y).tolist() == [10, 2, 30, 4]


def test_where_broadcasts_scalars():
    condition = array([[True, False], [False, True]])

    assert where(condition, 1, 0).tolist() == [
        [1, 0],
        [0, 1],
    ]


def test_where_rejects_non_bool_condition():
    with pytest.raises(TypeError):
        where(array([1, 0, 1]), 0, 1)


def test_nonzero_returns_index_arrays():
    a = array([[0, 1, 0], [2, 0, 3]])

    row_indices, col_indices = nonzero(a)

    assert row_indices.tolist() == [0, 1, 1]
    assert col_indices.tolist() == [1, 0, 2]


def test_take_flattens_by_default():
    a = array([[10, 20, 30], [40, 50, 60]])

    assert take(a, array([0, 2, 5])).tolist() == [10, 30, 60]


def test_take_along_axis():
    a = array([[10, 20, 30], [40, 50, 60]])

    assert take(a, array([0, 1, 0]), axis=0).tolist() == [
        [10, 20, 30],
        [40, 50, 60],
        [10, 20, 30],
    ]


def test_take_rejects_out_of_range_index():
    a = array([10, 20, 30])

    with pytest.raises(IndexError):
        take(a, array([3]))


def test_clip_limits_values():
    a = array([-2, 0, 3, 8])

    assert clip(a, 0, 5).tolist() == [0, 0, 3, 5]


def test_clip_with_only_min_or_max():
    a = array([-2, 0, 3, 8])

    assert clip(a, min_value=0).tolist() == [0, 0, 3, 8]
    assert clip(a, max_value=5).tolist() == [-2, 0, 3, 5]


def test_clip_without_bounds_returns_copy():
    a = array([1, 2, 3])
    clipped = clip(a)

    clipped[0] = 99

    assert a.tolist() == [1, 2, 3]


def test_where_coerces_array_like_operands():
    assert where(array([True, False]), [1, 2], (3, 4)).tolist() == [1, 4]


def test_take_rejects_non_integer_indices():
    with pytest.raises(TypeError):
        take(array([1, 2, 3]), array([0.0, 1.0]))


def test_take_coerces_array_like_source():
    assert take([10, 20, 30], array([2, 0])).tolist() == [30, 10]


def test_take_supports_negative_indices():
    assert take(array([10, 20, 30]), array([-1])).tolist() == [30]
