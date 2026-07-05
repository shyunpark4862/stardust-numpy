"""Sorting: in-place sort, argsort, and numpy.sort."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_sort_1d_in_place():
    a = array([3, 1, 2])

    result = a.sort()

    assert result is None
    assert a.tolist() == [1, 2, 3]


def test_sort_1d_with_duplicates():
    a = array([3, 1, 3, 2, 1])

    a.sort()

    assert a.tolist() == [1, 1, 2, 3, 3]


def test_sort_1d_view():
    a = array([9, 3, 1, 2, 8])
    view = a[1:4]

    view.sort()

    assert view.tolist() == [1, 2, 3]
    assert a.tolist() == [9, 1, 2, 3, 8]


def test_sort_2d_axis_1():
    a = array(
        [
            [3, 1, 2],
            [6, 4, 5],
        ]
    )

    a.sort(axis=1)

    assert a.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_sort_2d_axis_0():
    a = array(
        [
            [3, 1, 2],
            [6, 4, 5],
            [0, 7, 3],
        ]
    )

    a.sort(axis=0)

    assert a.tolist() == [
        [0, 1, 2],
        [3, 4, 3],
        [6, 7, 5],
    ]


def test_sort_negative_axis():
    a = array(
        [
            [3, 1, 2],
            [6, 4, 5],
        ]
    )

    a.sort(axis=-1)

    assert a.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_sort_axis_none_flattens_and_sorts_in_place():
    a = array([[3, 1, 2], [6, 4, 5]])

    a.sort(axis=None)

    assert a.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_sort_axis_none_on_non_contiguous_array():
    matrix = array([[3, 1, 2], [6, 4, 5]]).reshape(3, 2).T

    matrix.sort(axis=None)

    assert matrix.ravel().tolist() == [1, 2, 3, 4, 5, 6]


def test_functional_sort_axis_none():
    a = array([[3, 1, 2], [6, 4, 5]])

    result = sort(a, axis=None)

    assert result.shape == (6,)
    assert result.tolist() == [1, 2, 3, 4, 5, 6]
    assert a.tolist() == [
        [3, 1, 2],
        [6, 4, 5],
    ]


def test_argsort_1d():
    a = array([30, 10, 20])

    result = a.argsort()

    assert result.tolist() == [1, 2, 0]
    assert a[result].tolist() == [10, 20, 30]


def test_argsort_2d_axis_1():
    a = array(
        [
            [30, 10, 20],
            [60, 40, 50],
        ]
    )

    result = a.argsort(axis=1)

    assert result.tolist() == [
        [1, 2, 0],
        [1, 2, 0],
    ]


def test_argsort_2d_axis_0():
    a = array(
        [
            [3, 1],
            [1, 3],
            [2, 2],
        ]
    )

    result = a.argsort(axis=0)

    assert result.tolist() == [
        [1, 0],
        [2, 2],
        [0, 1],
    ]


def test_argsort_axis_none_returns_flat_indices():
    a = array([[3, 1, 2], [6, 4, 5]])

    assert a.argsort(axis=None).tolist() == [1, 2, 0, 4, 5, 3]
    assert a.ravel()[a.argsort(axis=None)].tolist() == [1, 2, 3, 4, 5, 6]


def test_functional_sort_does_not_change_original():
    a = array(
        [
            [3, 1, 2],
            [6, 4, 5],
        ]
    )

    result = sort(a, axis=1)

    assert result.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]

    assert a.tolist() == [
        [3, 1, 2],
        [6, 4, 5],
    ]


def test_functional_sort_axis_0():
    a = array(
        [
            [3, 1],
            [1, 3],
            [2, 2],
        ]
    )

    result = sort(a, axis=0)

    assert result.tolist() == [
        [1, 1],
        [2, 2],
        [3, 3],
    ]


def test_unique_returns_sorted_values():
    assert unique(array([3, 1, 3, 2])).tolist() == [1, 2, 3]


def test_unique_with_return_index():
    values, indices = unique(array([3, 1, 3, 2]), return_index=True)

    assert values.tolist() == [1, 2, 3]
    assert indices.tolist() == [1, 3, 0]


def test_unique_with_return_inverse():
    values, inverse = unique(array([3, 1, 3, 2]), return_inverse=True)

    assert values.tolist() == [1, 2, 3]
    assert inverse.tolist() == [2, 0, 2, 1]


def test_unique_with_return_counts():
    values, counts = unique(array([3, 1, 3, 2]), return_counts=True)

    assert values.tolist() == [1, 2, 3]
    assert counts.tolist() == [1, 1, 2]


def test_unique_treats_nan_as_equal():
    values = unique(array([3.0, float("nan"), 1.0, float("nan")]))

    assert values.tolist()[0] == 1.0
    assert values.tolist()[1] == 3.0
    assert math.isnan(values.tolist()[2])


def test_unique_on_complex_array():
    values = unique(array([1 + 2j, 3 + 4j, 1 + 2j]))

    assert values.tolist() == [1 + 2j, 3 + 4j]


def test_sort_orders_infinities():
    assert sort(array([float("inf"), 1.0, float("-inf"), 3.0])).tolist() == [
        float("-inf"),
        1.0,
        3.0,
        float("inf"),
    ]


def test_unique_treats_complex_nans_as_equal():
    left = complex(float("nan"), 1.0)
    right = complex(float("nan"), 2.0)

    assert unique(array([left, right])).shape == (1,)
