"""Array manipulation: concatenate and stack."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_concatenate_axis_zero():
    a = array([[1, 2], [3, 4]])
    b = array([[5, 6]])

    result = concatenate([a, b], axis=0)

    assert result.tolist() == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_concatenate_axis_one():
    a = array([[1, 2], [3, 4]])
    b = array([[5], [6]])

    result = concatenate([a, b], axis=1)

    assert result.tolist() == [
        [1, 2, 5],
        [3, 4, 6],
    ]


def test_concatenate_negative_axis():
    a = array([[1], [2]])
    b = array([[3], [4]])

    assert concatenate([a, b], axis=-1).tolist() == [
        [1, 3],
        [2, 4],
    ]


def test_stack_axis_zero():
    a = array([1, 2])
    b = array([3, 4])

    result = stack([a, b], axis=0)

    assert result.shape == (2, 2)
    assert result.tolist() == [
        [1, 2],
        [3, 4],
    ]


def test_stack_axis_one():
    a = array([1, 2])
    b = array([3, 4])

    result = stack([a, b], axis=1)

    assert result.shape == (2, 2)
    assert result.tolist() == [
        [1, 3],
        [2, 4],
    ]


def test_stack_negative_axis():
    a = array([1, 2])
    b = array([3, 4])

    assert stack([a, b], axis=-1).tolist() == [
        [1, 3],
        [2, 4],
    ]


def test_stack_shape_mismatch_raises():
    a = array([1, 2])
    b = array([3])

    with pytest.raises(ValueError):
        stack([a, b])


def test_concatenate_rejects_empty_input():
    with pytest.raises(ValueError):
        concatenate([])


def test_concatenate_rejects_zero_dimensional_arrays():
    with pytest.raises(ValueError):
        concatenate([full((), 1)])


def test_concatenate_rejects_ndim_mismatch():
    a = array([1, 2])
    b = array([[3, 4]])

    with pytest.raises(ValueError):
        concatenate([a, b])


def test_concatenate_rejects_shape_mismatch():
    a = array([[1, 2], [3, 4]])
    b = array([[5, 6, 7]])

    with pytest.raises(ValueError):
        concatenate([a, b], axis=0)


def test_stack_rejects_empty_input():
    with pytest.raises(ValueError):
        stack([])


def test_concatenate_with_non_contiguous_array():
    left = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    right = array([[7, 8, 9]])

    assert concatenate([left, right], axis=0).tolist() == [
        [1, 3, 5],
        [2, 4, 6],
        [7, 8, 9],
    ]
