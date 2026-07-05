"""Shape operations: transpose, reshape, ravel, squeeze, and expand_dims."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def nc_matrix_2x3() -> ndarray:
    """Non-contiguous (2, 3) view: [[1, 3, 5], [2, 4, 6]]."""
    return array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T


def test_transpose_returns_view():
    a = array([[1, 2, 3], [4, 5, 6]])

    b = a.T

    assert b.tolist() == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]

    assert b.shape == (3, 2)
    assert b.strides == (1, 3)
    assert b.buffer is a.buffer


def test_transpose_view_modifies_original():
    a = array([[1, 2, 3], [4, 5, 6]])

    a.T[1, 0] = 99

    assert a.tolist() == [
        [1, 99, 3],
        [4, 5, 6],
    ]


def test_transpose_custom_axes():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    b = a.transpose(1, 0, 2)

    assert b.tolist() == [
        [[1, 2], [5, 6]],
        [[3, 4], [7, 8]],
    ]


def test_ravel_contiguous_returns_view():
    a = array([[1, 2, 3], [4, 5, 6]])

    b = a.ravel()

    assert b.tolist() == [1, 2, 3, 4, 5, 6]
    assert b.buffer is a.buffer

    b[1] = 99

    assert a.tolist() == [
        [1, 99, 3],
        [4, 5, 6],
    ]


def test_flatten_returns_copy():
    a = array([[1, 2], [3, 4]])

    b = a.flatten()
    b[0] = 99

    assert a.tolist() == [
        [1, 2],
        [3, 4],
    ]

    assert b.tolist() == [99, 2, 3, 4]


def test_ravel_non_contiguous_returns_copy():
    a = array([[1, 2, 3], [4, 5, 6]])

    b = a.T.ravel()

    assert b.tolist() == [1, 4, 2, 5, 3, 6]
    assert b.buffer is not a.buffer


def test_reshape_contiguous_returns_view():
    a = array([[1, 2, 3], [4, 5, 6]])

    b = a.reshape(3, 2)

    assert b.tolist() == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    assert b.buffer is a.buffer
    assert b.shape == (3, 2)
    assert b.strides == (2, 1)

    b[1, 0] = 99

    assert a.tolist() == [
        [1, 2, 99],
        [4, 5, 6],
    ]


def test_reshape_accepts_tuple():
    a = array([1, 2, 3, 4])

    assert a.reshape((2, 2)).tolist() == [
        [1, 2],
        [3, 4],
    ]


def test_reshape_with_minus_one():
    a = array([1, 2, 3, 4, 5, 6])

    assert a.reshape(2, -1).tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_reshape_non_contiguous_returns_copy():
    a = array([[1, 2, 3], [4, 5, 6]])

    b = a.T.reshape(6)

    assert b.tolist() == [1, 4, 2, 5, 3, 6]
    assert b.buffer is not a.buffer


def test_squeeze():
    a = array([[[1, 2, 3]]])

    assert a.squeeze().shape == (3,)
    assert a.squeeze().tolist() == [1, 2, 3]
    assert a.squeeze(axis=0).shape == (1, 3)


def test_expand_dims():
    a = array([1, 2, 3])

    assert expand_dims(a, 0).tolist() == [[1, 2, 3]]
    assert expand_dims(a, 1).tolist() == [[1], [2], [3]]


def test_swapaxes():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.swapaxes(0, 1).tolist() == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]


def test_transpose_negative_axes():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.transpose(-1, 0).tolist() == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]


def test_transpose_rejects_wrong_number_of_axes():
    a = array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError):
        a.transpose(0)


def test_reshape_rejects_invalid_shapes():
    a = array([1, 2, 3, 4, 5, 6])

    with pytest.raises(ValueError):
        a.reshape(2, -1, -1)

    with pytest.raises(ValueError):
        a.reshape(2, -2)

    with pytest.raises(TypeError):
        a.reshape(2, 2.5)


def test_squeeze_rejects_non_singleton_axis():
    a = array([1, 2, 3])

    with pytest.raises(ValueError):
        a.squeeze(axis=0)

    with pytest.raises(TypeError):
        a.squeeze(axis=1.5)


def test_reshape_rejects_size_mismatch():
    a = array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        a.reshape(3, -1)

    with pytest.raises(ValueError):
        array([]).reshape(0, -1)

    with pytest.raises(ValueError):
        array([1, 2, 3, 4]).reshape(2, 3)


def test_ravel_treats_length_one_axes_as_contiguous():
    a = array([[1, 2, 3]])

    b = a.ravel()

    assert b.tolist() == [1, 2, 3]
    assert b.buffer is a.buffer


def test_expand_dims_rejects_invalid_axis():
    a = array([1, 2, 3])

    with pytest.raises(TypeError):
        expand_dims(a, 1.5)

    with pytest.raises(ValueError):
        expand_dims(a, 99)


def test_non_contiguous_view_is_not_c_contiguous():
    matrix = nc_matrix_2x3()

    assert matrix.shape == (2, 3)
    assert matrix.tolist() == [[1, 3, 5], [2, 4, 6]]
    assert not matrix._is_c_contiguous()


def test_astype_non_contiguous():
    matrix = nc_matrix_2x3().astype(float)

    assert matrix.dtype is float
    assert matrix.tolist() == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]


def test_copy_non_contiguous():
    matrix = nc_matrix_2x3()
    copied = matrix.copy()

    copied[0, 0] = 99

    assert matrix[0, 0] == 1
    assert copied[0, 0] == 99


def test_flat_buffer_slice_non_contiguous():
    matrix = nc_matrix_2x3()

    assert matrix._flat_buffer_slice() == [1, 3, 5, 2, 4, 6]


def test_write_flat_values_non_contiguous():
    matrix = nc_matrix_2x3()
    matrix._write_flat_values([9, 8, 7, 6, 5, 4])

    assert matrix._flat_buffer_slice() == [9, 8, 7, 6, 5, 4]


def test_unravel_index_empty_shape():
    from numpy._shape import unravel_index

    assert unravel_index(0, ()) == ()


def test_slice_length_empty_result():
    from numpy._shape import slice_length

    assert slice_length(3, 1, 1) == 0


def test_slice_length_zero_step():
    from numpy._shape import slice_length

    with pytest.raises(ValueError, match="slice step cannot be zero"):
        slice_length(0, 5, 0)
