"""Fancy and boolean indexing (read paths)."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_integer_fancy_indexing_1d():
    a = array([10, 20, 30, 40])

    assert a[[2, 0, -1]].tolist() == [30, 10, 40]


def test_integer_fancy_indexing_allows_duplicates():
    a = array([10, 20, 30])

    assert a[[1, 1, 0]].tolist() == [20, 20, 10]


def test_integer_fancy_indexing_returns_copy():
    a = array([10, 20, 30])
    selected = a[[1, 2]]

    selected[0] = 999

    assert a.tolist() == [10, 20, 30]
    assert selected.tolist() == [999, 30]


def test_boolean_mask_indexing_1d():
    a = array([10, 20, 30, 40])

    result = a[[True, False, True, False]]

    assert result.tolist() == [10, 30]


def test_boolean_mask_length_mismatch():
    a = array([10, 20, 30])

    with pytest.raises(ValueError):
        a[[True, False]]


def test_boolean_mask_returns_copy():
    a = array([10, 20, 30])

    selected = a[[True, False, True]]
    selected[0] = 999

    assert a.tolist() == [10, 20, 30]
    assert selected.tolist() == [999, 30]


def test_boolean_mask_ndarray():
    a = array([10, 20, 30, 40])

    result = a[a > 20]

    assert result.tolist() == [30, 40]


def test_boolean_mask_ndarray_length_mismatch():
    a = array([10, 20, 30])
    mask = array([True, False])

    with pytest.raises(ValueError):
        a[mask]


def test_integer_ndarray_fancy_indexing():
    a = array([10, 20, 30, 40])

    result = a[array([2, 0, -1])]

    assert result.tolist() == [30, 10, 40]


def test_integer_ndarray_fancy_indexing_returns_copy():
    a = array([10, 20, 30])

    selected = a[array([2, 0])]
    selected[0] = 999

    assert a.tolist() == [10, 20, 30]
    assert selected.tolist() == [999, 10]


def test_invalid_fancy_index_dtype_raises():
    a = array([10, 20, 30])

    with pytest.raises(TypeError):
        a[array([1.0, 0.0])]

    with pytest.raises(TypeError):
        a[array(["0", "1"])]


def test_integer_fancy_indexing_first_axis_2d():
    a = array(
        [
            [10, 20],
            [30, 40],
            [50, 60],
        ]
    )

    assert a[[2, 0]].tolist() == [
        [50, 60],
        [10, 20],
    ]


def test_integer_fancy_indexing_first_axis_3d():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    assert a[[1, 0]].tolist() == [
        [[5, 6], [7, 8]],
        [[1, 2], [3, 4]],
    ]


def test_boolean_mask_first_axis_2d():
    a = array(
        [
            [10, 20],
            [30, 40],
            [50, 60],
        ]
    )

    result = a[[True, False, True]]

    assert result.shape == (2, 2)
    assert result.tolist() == [
        [10, 20],
        [50, 60],
    ]


def test_boolean_mask_first_axis_3d():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    result = a[[False, True]]

    assert result.shape == (1, 2, 2)
    assert result.tolist() == [
        [[5, 6], [7, 8]],
    ]


def test_elementwise_boolean_mask():
    a = array(
        [
            [10, 20],
            [30, 40],
        ]
    )

    result = a[a > 20]

    assert result.shape == (2,)
    assert result.tolist() == [30, 40]


def test_elementwise_boolean_mask_returns_copy():
    a = array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    selected = a[a > 2]
    selected[0] = 999

    assert a.tolist() == [
        [1, 2],
        [3, 4],
    ]
    assert selected.tolist() == [999, 4]


def test_fancy_index_on_second_axis():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    assert a[:, [2, 0]].tolist() == [
        [12, 10],
        [22, 20],
    ]


def test_two_fancy_indices_pairwise():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    assert a[[0, 1], [2, 0]].tolist() == [
        12,
        20,
    ]


def test_fancy_indices_broadcast():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    assert a[[[0], [1]], [2, 0]].tolist() == [
        [12, 10],
        [22, 20],
    ]


def test_separated_fancy_indices():
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

    assert a[[0, 1], :, [2, 0]].tolist() == [
        [2, 5],
        [6, 9],
    ]


def test_fancy_with_basic_integer():
    a = array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ]
    )

    assert a[1, :, [2, 0]].tolist() == [
        [8, 6],
        [11, 9],
    ]


def test_fancy_with_none():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    result = a[None, [1, 0], [2, 0]]

    assert result.shape == (1, 2)
    assert result.tolist() == [[22, 10]]


def test_fancy_negative_indices():
    a = array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    assert a[[-1, 0], [-1, 1]].tolist() == [
        22,
        11,
    ]


def test_fancy_broadcast_with_slice():
    a = array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ]
    )

    result = a[[[0], [1]], :, [2, 0]]

    assert result.shape == (2, 2, 2)
    assert result.tolist() == [
        [[2, 5], [0, 3]],
        [[8, 11], [6, 9]],
    ]


def test_boolean_mask_becomes_integer_fancy_indices():
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

    assert a[mask].tolist() == [10, 12, 21]


def test_boolean_mask_inside_index_tuple():
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

    mask = array(
        [
            [True, False],
            [False, True],
        ]
    )

    assert a[mask].tolist() == [
        [0, 1, 2],
        [9, 10, 11],
    ]


def test_2d_boolean_mask():
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

    assert a[mask].tolist() == [10, 12, 21]


def test_boolean_mask_after_slice_selects_subcolumns():
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

    mask = array(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    result = a[:, mask]

    assert result.shape == (2, 3)
    assert result.tolist() == [[0, 2, 4], [6, 8, 10]]


def test_boolean_mask_after_slice_rejects_shape_mismatch():
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

    mask = array(
        [
            [True, False],
            [False, True],
        ]
    )

    with pytest.raises(ValueError):
        a[:, mask]


def test_boolean_mask_before_remaining_axis():
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

    mask = array(
        [
            [True, False],
            [False, True],
        ]
    )

    result = a[mask]

    assert result.shape == (2, 3)
    assert result.tolist() == [
        [0, 1, 2],
        [9, 10, 11],
    ]


def test_boolean_mask_with_integer_fancy_index():
    a = array(
        [
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            [
                [12, 13, 14, 15],
                [16, 17, 18, 19],
                [20, 21, 22, 23],
            ],
        ]
    )

    mask = array(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    result = a[mask, [3, 1, 0]]

    assert result.shape == (3,)
    assert result.tolist() == [3, 9, 16]


def test_zero_dimensional_fancy_index_returns_scalar():
    a = array([[1, 2], [3, 4]])
    row = array([0]).reshape(())
    column = array([1]).reshape(())

    assert a[row, column] == 2


def test_fancy_index_rejects_non_integer_values():
    a = array([10, 20, 30])

    with pytest.raises(TypeError):
        a[[1.0, 0.0]]


def test_boolean_mask_rejects_excess_dimensions():
    a = array([1, 2, 3])
    mask = array([[True, False, False], [False, True, False]])

    with pytest.raises(IndexError):
        a[mask]


def test_boolean_mask_rejects_wrong_shape_after_partial_index():
    matrix = array([[1, 2], [3, 4]])

    with pytest.raises(IndexError, match="too many indices"):
        matrix[0, array([[True]])]

    with pytest.raises(ValueError, match="boolean index shape"):
        matrix[array([True, False, True])]


def test_fancy_index_rejects_non_integer_arrays():
    values = array([1, 2, 3, 4])

    with pytest.raises(
        TypeError, match="array index must contain only integers or booleans"
    ):
        values[array([1.0, 2.0])]

    with pytest.raises(
        TypeError, match="index must be int, slice, ndarray, None, or Ellipsis"
    ):
        values["bad"]


def nc_bool_matrix_2x3(rows: list[list[bool]]) -> ndarray:
    from numpy._factory import view_ndarray

    buffer = [False] * 6
    layout = [
        (0, 0, rows[0][0]),
        (1, 0, rows[1][0]),
        (0, 1, rows[0][1]),
        (1, 1, rows[1][1]),
        (0, 2, rows[0][2]),
        (1, 2, rows[1][2]),
    ]

    for row, column, value in layout:
        buffer[row + column * 2] = value

    return view_ndarray(buffer, (2, 3), (1, 2), dtype=bool)


def test_boolean_index_with_non_contiguous_mask():
    base = array([[1, 2, 3], [4, 5, 6]])
    mask = nc_bool_matrix_2x3(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    assert base[mask].tolist() == [1, 3, 5]


def test_boolean_mask_to_indices_requires_bool_dtype():
    with pytest.raises(TypeError, match="boolean mask must have dtype bool"):
        array([1, 2, 3])._boolean_mask_to_integer_indices(array([1, 0, 1]))
