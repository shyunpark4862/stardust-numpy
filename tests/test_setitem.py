"""Array assignment: slices, fancy indices, and broadcasting."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_1d_integer_index_assignment_mutates_buffer():
    a = array([1, 2, 3])

    a[1] = 99
    a[-1] = 50

    assert a.tolist() == [1, 99, 50]


def test_2d_assignment():
    a = array([[1, 2], [3, 4]])

    a[1, 0] = 99

    assert a[1, 0] == 99
    assert a.buffer == [1, 2, 99, 4]


def test_slice_scalar_assignment():
    a = array([[1, 2, 3], [4, 5, 6]])

    a[:, 1] = 99

    assert a.tolist() == [
        [1, 99, 3],
        [4, 99, 6],
    ]


def test_slice_array_assignment():
    a = array([[1, 2, 3], [4, 5, 6]])

    a[0, 1:] = array([7, 8])

    assert a.tolist() == [
        [1, 7, 8],
        [4, 5, 6],
    ]


def test_slice_array_assignment_shape_mismatch():
    a = array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError):
        a[:, 1] = array([10, 20, 30])


def test_slice_array_assignment_broadcast():
    a = array([[1, 2, 3], [4, 5, 6]])

    a[:, 1] = array([99])

    assert a.tolist() == [
        [1, 99, 3],
        [4, 99, 6],
    ]


def test_slice_list_assignment_broadcast():
    a = array([[1, 2, 3], [4, 5, 6]])

    a[0, 1:] = [7, 8]

    assert a.tolist() == [
        [1, 7, 8],
        [4, 5, 6],
    ]


def test_ellipsis_scalar_assignment():
    a = array([[1, 2, 3], [4, 5, 6]])

    a[..., 1] = 99

    assert a.tolist() == [
        [1, 99, 3],
        [4, 99, 6],
    ]


def test_ellipsis_row_assignment():
    a = array([[1, 2, 3], [4, 5, 6]])

    a[1, ...] = 0

    assert a.tolist() == [
        [1, 2, 3],
        [0, 0, 0],
    ]


def test_elementwise_boolean_mask_assignment():
    a = array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    a[a > 2] = 0

    assert a.tolist() == [
        [1, 2],
        [0, 0],
    ]


def test_boolean_mask_axis_assignment():
    a = array([[1, 2], [3, 4]])

    # mask.ndim == 1 covers the first axis only
    a[array([True, False])] = 0

    assert a.tolist() == [
        [0, 0],
        [3, 4],
    ]


def test_fancy_setitem_scalar_with_integer_indices():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[[0, 1], [2, 0]] = 99

    assert a.tolist() == [
        [10, 11, 99],
        [99, 21, 22],
    ]


def test_fancy_setitem_scalar_with_slice_and_integer_list():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[:, [2, 0]] = 0

    assert a.tolist() == [
        [0, 11, 0],
        [0, 21, 0],
    ]


def test_fancy_setitem_scalar_with_boolean_mask():
    a = array([10, 20, 30, 40])

    a[a > 20] = -1

    assert a.tolist() == [10, 20, -1, -1]


def test_fancy_setitem_scalar_with_2d_boolean_mask():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    mask = array(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    a[mask] = 0

    assert a.tolist() == [
        [0, 11, 0],
        [20, 0, 22],
    ]


def test_fancy_setitem_scalar_with_separated_fancy_indices():
    a = array(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            [
                [6, 7, 8],
                [9, 10, 11],
            ],
        ]
    )

    a[[0, 1], :, [2, 0]] = -1

    assert a.tolist() == [
        [
            [0, 1, -1],
            [3, 4, -1],
        ],
        [
            [-1, 7, 8],
            [-1, 10, 11],
        ],
    ]


def test_fancy_setitem_array_rhs():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[[0, 1], [2, 0]] = array([100, 200])

    assert a.tolist() == [
        [10, 11, 100],
        [200, 21, 22],
    ]


def test_fancy_setitem_array_rhs_with_slice():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[:, [2, 0]] = array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    assert a.tolist() == [
        [2, 11, 1],
        [4, 21, 3],
    ]


def test_fancy_setitem_array_rhs_broadcasts():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[:, [2, 0]] = array([100, 200])

    assert a.tolist() == [
        [200, 11, 100],
        [200, 21, 100],
    ]


def test_fancy_setitem_array_rhs_shape_error():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    with pytest.raises(ValueError):
        a[:, [2, 0]] = array([1, 2, 3])


def test_fancy_setitem_list_rhs():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[[0, 1], [2, 0]] = [100, 200]

    assert a.tolist() == [
        [10, 11, 100],
        [200, 21, 22],
    ]


def test_fancy_setitem_nested_list_rhs():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[:, [2, 0]] = [
        [1, 2],
        [3, 4],
    ]

    assert a.tolist() == [
        [2, 11, 1],
        [4, 21, 3],
    ]


def test_fancy_setitem_list_rhs_broadcasts():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    a[:, [2, 0]] = [100, 200]

    assert a.tolist() == [
        [200, 11, 100],
        [200, 21, 100],
    ]


def test_fancy_setitem_reads_rhs_before_writing():
    a = array([10, 20, 30])

    a[[0, 1]] = a[[1, 0]]

    assert a.tolist() == [20, 10, 30]


def test_fancy_setitem_reversed_rows_from_same_array():
    a = array(
        [
            [10, 11],
            [20, 21],
        ]
    )

    a[[0, 1]] = a[[1, 0]]

    assert a.tolist() == [
        [20, 21],
        [10, 11],
    ]
