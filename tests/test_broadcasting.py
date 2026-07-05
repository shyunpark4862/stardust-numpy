"""Broadcasting rules and broadcast_to / broadcast_arrays."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_broadcast_shape():
    assert broadcast_shape((3, 1), (1, 4)) == (3, 4)
    assert broadcast_shape((2, 3), (3,)) == (2, 3)
    assert broadcast_shape((3,), ()) == (3,)
    assert broadcast_shape() == ()


def test_broadcast_arrays_empty():
    assert broadcast_arrays() == ()


def test_broadcast_to():
    a = array([[1], [2], [3]])

    b = broadcast_to(a, (3, 4))

    assert b.shape == (3, 4)
    assert b.strides == (1, 0)
    assert b.tolist() == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]

    assert b.buffer is a.buffer


def test_broadcasting_2d_and_1d():
    a = array([[1], [2], [3]])
    b = array([10, 20, 30])

    result = a + b

    assert result.shape == (3, 3)
    assert result.tolist() == [
        [11, 21, 31],
        [12, 22, 32],
        [13, 23, 33],
    ]


def test_broadcasting_scalar_shape():
    a = array([[1, 2, 3], [4, 5, 6]])
    b = array([10, 20, 30])

    assert (a + b).tolist() == [
        [11, 22, 33],
        [14, 25, 36],
    ]


def test_broadcasting_incompatible_shapes():
    a = array([[1, 2, 3], [4, 5, 6]])
    b = array([10, 20])

    with pytest.raises(ValueError):
        a + b


def test_public_broadcast_to():
    a = array([1, 2, 3])

    result = broadcast_to(a, (2, 3))

    assert result.shape == (2, 3)
    assert result.tolist() == [
        [1, 2, 3],
        [1, 2, 3],
    ]


def test_broadcast_arrays():
    a = array([[1], [2], [3]])
    b = array([10, 20, 30])

    left, right = broadcast_arrays(a, b)

    assert left.shape == (3, 3)
    assert right.shape == (3, 3)

    assert left.tolist() == [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ]

    assert right.tolist() == [
        [10, 20, 30],
        [10, 20, 30],
        [10, 20, 30],
    ]


def test_ufunc_out_with_broadcasting():
    a = array([[1], [2]])
    b = array([10, 20, 30])
    out = zeros((2, 3))

    add(a, b, out=out)

    assert out.tolist() == [
        [11, 21, 31],
        [12, 22, 32],
    ]


def test_add_where_broadcasting():
    result = add(
        array([[1], [2]]),
        array([10, 20, 30]),
        where=array([True, False, True]),
    )

    assert result.tolist() == [
        [11, 0, 31],
        [12, 0, 32],
    ]


def test_broadcast_to_rejects_fewer_target_dimensions():
    a = array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError):
        a.broadcast_to((3,))
