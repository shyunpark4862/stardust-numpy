"""Iteration: iter, len, flat, ndindex, ndenumerate, nditer."""

from __future__ import annotations

import pytest

from numpy import *  # noqa: F403


def test_len_returns_first_axis_size():
    assert len(array([1, 2, 3])) == 3
    assert len(array([[1, 2], [3, 4], [5, 6]])) == 3
    assert len(array([])) == 0


def test_len_raises_for_zero_dimensional_array():
    with pytest.raises(TypeError, match="len\\(\\) of unsized object"):
        len(full((), 7))


def test_iter_yields_scalars_for_1d():
    assert list(iter(array([10, 20, 30]))) == [10, 20, 30]


def test_iter_yields_subarray_views_for_2d():
    matrix = array([[1, 2, 3], [4, 5, 6]])

    rows = list(iter(matrix))

    assert len(rows) == 2
    assert rows[0].tolist() == [1, 2, 3]
    assert rows[1].tolist() == [4, 5, 6]
    assert rows[0].shape == (3,)
    assert rows[0].buffer is matrix.buffer


def test_iter_subarray_view_modifies_original():
    matrix = array([[1, 2], [3, 4]])

    row = next(iter(matrix))
    row[0] = 99

    assert matrix.tolist() == [[99, 2], [3, 4]]


def test_iter_yields_subarray_views_for_3d():
    cube = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    layers = list(iter(cube))

    assert len(layers) == 2
    assert layers[0].shape == (2, 2)
    assert layers[0].tolist() == [[1, 2], [3, 4]]


def test_iter_raises_for_zero_dimensional_array():
    with pytest.raises(TypeError, match="iteration over a 0-d array"):
        list(iter(full((), 7)))


def test_flat_yields_scalars_in_c_order():
    matrix = array([[1, 2, 3], [4, 5, 6]])

    assert list(matrix.flat) == [1, 2, 3, 4, 5, 6]


def test_flat_does_not_create_views_or_copies():
    matrix = array([[1, 2], [3, 4]])
    flat = matrix.flat

    assert list(flat) == [1, 2, 3, 4]
    assert list(flat) == [1, 2, 3, 4]


def test_flat_follows_c_order_on_non_contiguous_view():
    matrix = array([[1, 2, 3], [4, 5, 6]])
    transposed = matrix.T

    assert list(transposed.flat) == [1, 4, 2, 5, 3, 6]


def test_flat_yields_single_scalar_for_zero_dimensional_array():
    assert list(full((), 7).flat) == [7]


def test_flat_is_empty_for_empty_array():
    assert list(array([]).flat) == []


def test_ndindex_accepts_multiple_dimensions():
    assert list(ndindex(2, 3)) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def test_ndindex_accepts_shape_tuple():
    assert list(ndindex((2, 2))) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_ndindex_accepts_single_dimension():
    assert list(ndindex(3)) == [(0,), (1,), (2,)]


def test_ndindex_yields_nothing_for_empty_shape():
    assert list(ndindex(0)) == []
    assert list(ndindex(2, 0)) == []


def test_ndindex_yields_empty_tuple_for_scalar_shape():
    assert list(ndindex()) == [()]


def test_ndenumerate_yields_index_value_pairs_in_c_order():
    matrix = array([[10, 20], [30, 40]])

    assert list(ndenumerate(matrix)) == [
        ((0, 0), 10),
        ((0, 1), 20),
        ((1, 0), 30),
        ((1, 1), 40),
    ]


def test_ndenumerate_works_for_1d_and_zero_dimensional_arrays():
    assert list(ndenumerate(array([5, 6]))) == [((0,), 5), ((1,), 6)]
    assert list(ndenumerate(full((), 9))) == [((), 9)]


def test_nditer_single_operand_yields_scalars():
    assert list(nditer(array([1, 2, 3]))) == [1, 2, 3]


def test_nditer_multiple_operands_yields_scalar_tuples():
    left = array([1, 2, 3])
    right = array([10, 20, 30])

    assert list(nditer([left, right])) == [(1, 10), (2, 20), (3, 30)]


def test_nditer_broadcasts_operands_to_common_shape():
    left = array([[1], [2], [3]])
    right = array([10, 20, 30])

    assert list(nditer((left, right))) == [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 20),
        (2, 30),
        (3, 10),
        (3, 20),
        (3, 30),
    ]


def test_nditer_rejects_unsupported_op_flags():
    with pytest.raises(ValueError, match="only 'readonly' is supported"):
        nditer(array([1, 2]), op_flags="readwrite")


def test_nditer_rejects_empty_operand_sequence():
    with pytest.raises(ValueError, match="at least one operand"):
        nditer([])


def test_nditer_rejects_invalid_operand_type():
    with pytest.raises(TypeError, match="nditer operands must be"):
        nditer(123)


def test_iteration_handles_empty_arrays():
    empty = array([])
    empty_matrix = zeros((0, 3))
    ragged = zeros((3, 0))

    assert len(empty) == 0
    assert list(iter(empty)) == []
    assert list(empty.flat) == []
    assert list(ndenumerate(empty)) == []
    assert list(nditer(empty)) == []

    assert len(empty_matrix) == 0
    assert list(iter(empty_matrix)) == []
    assert list(empty_matrix.flat) == []

    assert len(ragged) == 3
    assert all(row.shape == (0,) for row in iter(ragged))
    assert list(ragged.flat) == []


def test_iteration_preserves_nan_and_inf():
    import math

    values = array([1.0, float("nan"), float("inf"), float("-inf")])

    flat = list(values.flat)
    assert flat[0] == 1.0
    assert math.isnan(flat[1])
    assert flat[2] == float("inf")
    assert flat[3] == float("-inf")

    iterated = list(nditer(values))
    assert iterated[0] == 1.0
    assert math.isnan(iterated[1])
    assert iterated[2] == float("inf")
    assert iterated[3] == float("-inf")

    enumerated = list(ndenumerate(values))
    assert enumerated[0] == ((0,), 1.0)
    assert math.isnan(enumerated[1][1])
    assert enumerated[2][1] == float("inf")
    assert enumerated[3][1] == float("-inf")


def test_ndenumerate_accepts_array_like():
    assert list(ndenumerate([1, 2, 3])) == [((0,), 1), ((1,), 2), ((2,), 3)]


def test_nditer_accepts_array_like_operands_in_sequence():
    assert list(nditer([[1, 2, 3], [10, 20, 30]])) == [
        (1, 10),
        (2, 20),
        (3, 30),
    ]


def test_nditer_broadcast_mismatch_raises_value_error():
    with pytest.raises(
        ValueError, match="operands could not be broadcast together"
    ):
        list(nditer([array([1, 2]), array([1, 2, 3])]))


def test_nditer_with_non_contiguous_operands():
    left = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    right = array([[10, 20, 30], [40, 50, 60]])

    assert list(nditer((left, right))) == [
        (1, 10),
        (3, 20),
        (5, 30),
        (2, 40),
        (4, 50),
        (6, 60),
    ]


def test_nditer_single_non_contiguous_operand():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T

    assert list(nditer(matrix)) == [1, 3, 5, 2, 4, 6]


def test_nditer_matches_flat_iteration_for_non_contiguous_views():
    left = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    right = array([[10, 20, 30], [40, 50, 60]])

    assert list(nditer((left, right))) == list(
        zip(list(left.flat), list(right.flat))
    )
