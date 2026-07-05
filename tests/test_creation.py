"""Array construction: array, zeros, ones, full, arange."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_array_exposes_shape_ndim_size_and_values():
    a = array([1, 2, 3])

    assert a.shape == (3,)
    assert a.ndim == 1
    assert a.size == 3
    assert a.tolist() == [1, 2, 3]


def test_2d_array_has_c_order_buffer_and_strides():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.buffer == [1, 2, 3, 4, 5, 6]
    assert a.shape == (2, 3)
    assert a.strides == (3, 1)
    assert a.ndim == 2
    assert a.size == 6


def test_copy_allocates_independent_buffer():
    a = array([1, 2, 3])
    b = a.copy()

    b[0] = 99

    assert a.tolist() == [1, 2, 3]
    assert b.tolist() == [99, 2, 3]


def test_zeros_ones_and_full_create_expected_values():
    assert zeros(3).tolist() == [0, 0, 0]
    assert ones((3,)).tolist() == [1, 1, 1]
    assert full(3, "x").tolist() == ["x", "x", "x"]


def test_arange():
    assert arange(5).tolist() == [0, 1, 2, 3, 4]
    assert arange(2, 8, 2).tolist() == [2, 4, 6]
    assert arange(5, 0, -2).tolist() == [5, 3, 1]


def test_array_rejects_ragged_nested_list():
    with pytest.raises(ValueError):
        array([[1, 2], [3]])


def test_3d_array_creation():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    assert a.buffer == [1, 2, 3, 4, 5, 6, 7, 8]
    assert a.shape == (2, 2, 2)
    assert a.strides == (4, 2, 1)


def test_2d_tolist():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_3d_tolist():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    assert a.tolist() == [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]


def test_multidimensional_creation_functions():
    assert zeros((2, 3)).tolist() == [
        [0, 0, 0],
        [0, 0, 0],
    ]

    assert ones((2, 2)).tolist() == [
        [1, 1],
        [1, 1],
    ]

    assert full((2, 2), "x").tolist() == [
        ["x", "x"],
        ["x", "x"],
    ]


def test_array_from_tuple():
    assert array((1, 2, 3)).tolist() == [1, 2, 3]
    assert array(((1, 2), (3, 4))).tolist() == [[1, 2], [3, 4]]


def test_array_from_range():
    assert array(range(4)).tolist() == [0, 1, 2, 3]


def test_array_rejects_non_array_like():
    with pytest.raises(TypeError):
        array(5)

    with pytest.raises(TypeError):
        array("abc")


def test_array_infers_str_dtype():
    assert array(["a", "b"]).dtype is str


def test_creation_validates_triangle_arguments():
    with pytest.raises(TypeError):
        tri(-1)

    with pytest.raises(TypeError):
        eye(3, m=-1)

    with pytest.raises(TypeError):
        tril(array([[1, 2], [3, 4]]), k=1.5)


def test_tril_and_triu_handle_scalar_shape():
    value = full((), 7)

    assert tril(value).tolist() == 7
    assert triu(value).tolist() == 7


def test_tril_promotes_1d_input_to_square_matrix():
    assert tril(array([1, 2, 3, 4])).tolist() == [
        [1, 0, 0, 0],
        [1, 2, 0, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 4],
    ]


def test_array_rejects_non_sequence_row():
    with pytest.raises(TypeError):
        array([[1, 2], 3])


def test_eye_creates_identity_matrix():
    assert eye(3).tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_eye_with_custom_shape_and_dtype():
    assert eye(2, 3, dtype=int).tolist() == [
        [1, 0, 0],
        [0, 1, 0],
    ]


def test_eye_with_offset_diagonal():
    assert eye(3, k=1).tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]


def test_diag_extracts_diagonal_from_2d_array():
    x = arange(9).reshape(3, 3)

    assert diag(x).tolist() == [0, 4, 8]
    assert diag(x, k=1).tolist() == [1, 5]
    assert diag(x, k=-1).tolist() == [3, 7]


def test_diag_builds_diagonal_matrix_from_1d_array():
    assert diag(array([1, 2, 3])).tolist() == [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
    ]


def test_diag_builds_offset_diagonal_matrix():
    assert diag(array([1, 2, 3]), k=1).tolist() == [
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
    ]


def test_diag_builds_negative_offset_diagonal_matrix():
    assert diag(array([1, 2, 3]), k=-1).tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
    ]


def test_diag_round_trip():
    x = arange(9).reshape(3, 3)

    assert diag(diag(x)).tolist() == [
        [0, 0, 0],
        [0, 4, 0],
        [0, 0, 8],
    ]


def test_diag_rejects_higher_dimensional_input():
    with pytest.raises(ValueError):
        diag(array([[[1, 2], [3, 4]]]))


def test_tri_creates_lower_triangle_of_ones():
    assert tri(3, 5, 2, dtype=int).tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]


def test_tri_with_negative_k():
    assert tri(3, 5, -1).tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ]


def test_tril_zeros_above_diagonal():
    a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    assert tril(a, -1).tolist() == [
        [0, 0, 0],
        [4, 0, 0],
        [7, 8, 0],
        [10, 11, 12],
    ]


def test_triu_zeros_below_diagonal():
    a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    assert triu(a, -1).tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [0, 8, 9],
        [0, 0, 12],
    ]


def test_tril_applies_to_last_two_axes():
    a = arange(3 * 4 * 5).reshape(3, 4, 5)

    assert tril(a)[0, 1, 2] == 0
    assert tril(a)[0, 1, 1] == 6
    assert tril(a)[1, 0, 0] == 20


def test_tril_and_triu_return_copies():
    a = array([[1, 2], [3, 4]])
    lower = tril(a)

    lower[0, 0] = 99

    assert a.tolist() == [[1, 2], [3, 4]]


def test_arange_rejects_zero_step():
    with pytest.raises(ValueError):
        arange(0, 5, 0)


def test_scalar_array_and_repr():
    scalar = full((), 42)

    assert scalar.shape == ()
    assert scalar.ndim == 0
    assert scalar.tolist() == 42
    assert repr(scalar) == "array(shape=(), dtype=int, value=42)"
    assert str(scalar) == "42"


def test_ndarray_constructor_defaults():
    from numpy.ndarray import ndarray

    a = ndarray([1, 2, 3])

    assert a.shape == (3,)
    assert a.strides == (1,)


def test_ndarray_rejects_mismatched_shape_and_strides():
    from numpy.ndarray import ndarray

    with pytest.raises(ValueError):
        ndarray([1, 2, 3], shape=(3,), strides=(1, 2))


def test_triangle_row_column_for_1d_shape():
    from numpy.creation import _triangle_row_column

    assert _triangle_row_column((2,), (3,)) == (0, 2)
