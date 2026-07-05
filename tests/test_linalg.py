"""Linear algebra: dot and matmul."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_dot_1d_vectors():
    a = array([1, 2, 3])
    b = array([4, 5, 6])

    assert a.dot(b) == 32


def test_dot_1d_vectors_with_negative_values():
    a = array([1, -2, 3])
    b = array([4, 5, -6])

    assert a.dot(b) == -24


def test_dot_1d_shape_error():
    a = array([1, 2])
    b = array([1, 2, 3])

    with pytest.raises(ValueError):
        a.dot(b)


def test_dot_matrix_vector():
    matrix = array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    vector = array([10, 20, 30])

    result = matrix.dot(vector)

    assert result.shape == (2,)
    assert result.tolist() == [140, 320]


def test_dot_matrix_vector_shape_error():
    matrix = array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    vector = array([10, 20])

    with pytest.raises(ValueError):
        matrix.dot(vector)


def test_dot_vector_matrix():
    vector = array([10, 20])

    matrix = array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = vector.dot(matrix)

    assert result.shape == (3,)
    assert result.tolist() == [90, 120, 150]


def test_dot_vector_matrix_shape_error():
    vector = array([10, 20, 30])

    matrix = array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    with pytest.raises(ValueError):
        vector.dot(matrix)


def test_dot_matrix_matrix():
    left = array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    right = array(
        [
            [10, 20],
            [30, 40],
            [50, 60],
        ]
    )

    result = left.dot(right)

    assert result.shape == (2, 2)
    assert result.tolist() == [
        [220, 280],
        [490, 640],
    ]


def test_dot_matrix_matrix_shape_error():
    left = array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    right = array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    with pytest.raises(ValueError):
        left.dot(right)


def test_matmul_vector_vector():
    left = array([1, 2, 3])
    right = array([4, 5, 6])

    assert left @ right == 32


def test_matmul_matrix_vector():
    left = array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    right = array([10, 20, 30])

    assert (left @ right).tolist() == [140, 320]


def test_matmul_vector_matrix():
    left = array([10, 20])
    right = array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    assert (left @ right).tolist() == [90, 120, 150]


def test_matmul_matrix_matrix():
    left = array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    right = array(
        [
            [5, 6],
            [7, 8],
        ]
    )

    assert (left @ right).tolist() == [
        [19, 22],
        [43, 50],
    ]


def test_matmul_batched_matrix_with_shared_matrix():
    left = array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
    )

    right = array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    result = left @ right

    assert result.shape == (2, 2, 2)
    assert result.tolist() == [
        [
            [22, 28],
            [49, 64],
        ],
        [
            [76, 100],
            [103, 136],
        ],
    ]


def test_matmul_batched_matrix_matrix():
    left = array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ]
    )

    right = array(
        [
            [
                [10, 20],
                [30, 40],
            ],
            [
                [50, 60],
                [70, 80],
            ],
        ]
    )

    result = left @ right

    assert result.shape == (2, 2, 2)
    assert result.tolist() == [
        [
            [70, 100],
            [150, 220],
        ],
        [
            [670, 780],
            [910, 1060],
        ],
    ]


def test_matmul_broadcasts_both_batch_shapes():
    left = array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ],
            ],
            [
                [
                    [5, 6],
                    [7, 8],
                ],
            ],
        ]
    )
    # shape: (2, 1, 2, 2)

    right = array(
        [
            [
                [
                    [10, 20],
                    [30, 40],
                ],
                [
                    [50, 60],
                    [70, 80],
                ],
                [
                    [90, 100],
                    [110, 120],
                ],
            ],
        ]
    )
    # shape: (1, 3, 2, 2)

    result = left @ right

    assert result.shape == (2, 3, 2, 2)

    assert result.tolist() == [
        [
            [
                [70, 100],
                [150, 220],
            ],
            [
                [190, 220],
                [430, 500],
            ],
            [
                [310, 340],
                [710, 780],
            ],
        ],
        [
            [
                [230, 340],
                [310, 460],
            ],
            [
                [670, 780],
                [910, 1060],
            ],
            [
                [1110, 1220],
                [1510, 1660],
            ],
        ],
    ]


def test_batched_matrix_vector_matmul():
    left = array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
    )
    # shape (2, 2, 3)

    right = array([10, 20, 30])
    # shape (3,)

    result = left @ right

    assert result.shape == (2, 2)
    assert result.tolist() == [
        [140, 320],
        [500, 680],
    ]


def test_vector_batched_matrix_matmul():
    left = array([10, 20])
    # shape (2,)

    right = array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
    )
    # shape (2, 2, 3)

    result = left @ right

    assert result.shape == (2, 3)
    assert result.tolist() == [
        [90, 120, 150],
        [270, 300, 330],
    ]


def test_batched_matmul_shape_error():
    left = array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        ]
    )

    right = array([10, 20])

    with pytest.raises(ValueError):
        left @ right


def test_matmul_rejects_scalar():
    matrix = array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    with pytest.raises(ValueError):
        matrix @ 3


def test_matmul_batch_broadcast_error():
    left = array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ]
    )
    # shape (2, 2, 2)

    right = array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
            [
                [9, 10],
                [11, 12],
            ],
        ]
    )
    # shape (3, 2, 2)

    with pytest.raises(ValueError):
        left @ right


def test_dot_rejects_high_dimensional_inputs():
    left = array([[[1, 2], [3, 4]]])
    right = array([[[5, 6], [7, 8]]])

    with pytest.raises(NotImplementedError):
        left.dot(right)


def test_matmul_rejects_scalar_operands():
    scalar = full((), 7)

    with pytest.raises(ValueError):
        scalar @ array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        array([[1, 2], [3, 4]]) @ scalar


def test_diagonal_extracts_2d_diagonals():
    a = arange(4).reshape(2, 2)

    assert diagonal(a).tolist() == [0, 3]
    assert diagonal(a, offset=1).tolist() == [1]


def test_diagonal_on_3d_array():
    a = arange(8).reshape(2, 2, 2)

    assert diagonal(a, axis1=0, axis2=1).tolist() == [
        [0, 6],
        [1, 7],
    ]


def test_trace_on_2d_array():
    assert trace(eye(3)) == 3


def test_trace_on_3d_array():
    a = arange(8).reshape(2, 2, 2)

    assert trace(a).tolist() == [6, 8]


def test_trace_on_4d_array():
    a = arange(24).reshape(2, 2, 2, 3)

    assert trace(a).shape == (2, 3)


def test_outer_product():
    a = array([1, 2, 3])
    b = array([10, 20])

    assert outer(a, b).tolist() == [
        [10, 20],
        [20, 40],
        [30, 60],
    ]


def test_outer_flattens_inputs():
    left = array([[1, 2], [3, 4]])
    right = array([[10], [20]])

    assert outer(left, right).shape == (4, 2)
    assert outer(left, right).tolist() == [
        [10, 20],
        [20, 40],
        [30, 60],
        [40, 80],
    ]


def test_diagonal_rejects_insufficient_dimensions():
    with pytest.raises(ValueError):
        diagonal(array([1, 2, 3]))


def test_diagonal_validates_arguments():
    with pytest.raises(ValueError):
        diagonal(array([[1, 2], [3, 4]]), axis1=0, axis2=0)

    with pytest.raises(TypeError):
        diagonal(array([[1, 2], [3, 4]]), offset=1.5)


def test_dot_matrix_vector_non_contiguous():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T

    assert matrix.dot(array([10, 20, 30])).tolist() == [220, 280]


def test_dot_vector_matrix_non_contiguous():
    vector = array([10, 20])
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T

    assert vector.dot(matrix).tolist() == [50, 110, 170]


def test_dot_matrices_non_contiguous():
    left = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    right = array(
        [
            [10, 20],
            [30, 40],
            [50, 60],
        ]
    )

    assert left.dot(right).tolist() == [
        [350, 440],
        [440, 560],
    ]
