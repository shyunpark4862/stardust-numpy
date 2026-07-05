"""Public API validation and array-like coercion."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_public_api_accepts_list_tuple_and_range():
    assert expand_dims([1, 2, 3], 0).tolist() == [[1, 2, 3]]
    assert expand_dims((1, 2, 3), 1).tolist() == [[1], [2], [3]]

    a = array([1, 2, 3])
    b = array([4, 5, 6])

    assert concatenate([a, (4, 5, 6)]).tolist() == [1, 2, 3, 4, 5, 6]
    assert concatenate([a, b]).tolist() == [1, 2, 3, 4, 5, 6]

    left, right = broadcast_arrays([1], (4, 5, 6))
    assert left.tolist() == [1, 1, 1]
    assert right.tolist() == [4, 5, 6]

    assert sort((3, 1, 2)).tolist() == [1, 2, 3]
    assert add([1, 2, 3], 1).tolist() == [2, 3, 4]
    assert (array([1, 2, 3]) + [10, 20, 30]).tolist() == [11, 22, 33]
    assert (array([1, 2, 3]) * (2, 3, 4)).tolist() == [2, 6, 12]


def test_public_api_rejects_non_array_like_operands():
    with pytest.raises(TypeError):
        expand_dims(5, 0)

    with pytest.raises(TypeError):
        sort("abc")

    with pytest.raises(TypeError):
        add("abc", 1)


def test_arange_rejects_non_numeric_arguments():
    with pytest.raises(TypeError):
        arange(True, 5)

    with pytest.raises(TypeError):
        arange(0, "5")


def test_full_rejects_invalid_shape():
    with pytest.raises(TypeError):
        full([2, 3], 0)

    with pytest.raises(ValueError):
        full((2, -1), 0)

    with pytest.raises(TypeError):
        full((2, 2.5), 0)
