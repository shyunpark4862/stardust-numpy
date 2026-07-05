"""Empty array behavior across operations."""

from __future__ import annotations

import pytest

from numpy import *  # noqa: F403


def test_empty_array_creation_and_shape_ops():
    assert array([]).shape == (0,)
    assert zeros((0, 3)).shape == (0, 3)
    assert eye(0).shape == (0, 0)
    assert arange(0, 0).shape == (0,)

    empty = array([])
    assert empty.ravel().shape == (0,)
    assert empty.reshape(0).shape == (0,)
    assert expand_dims(empty, 0).shape == (1, 0)


def test_empty_array_elementwise_and_ufuncs():
    empty = array([])

    assert (empty + 1).shape == (0,)
    assert sqrt(empty).shape == (0,)
    assert divide(empty, empty).shape == (0,)


def test_empty_array_reductions():
    empty = array([])

    assert empty.sum() == 0
    assert empty.prod() == 1
    assert empty.any() is False
    assert empty.all() is True
    assert empty.cumsum().shape == (0,)
    assert empty.cumprod().shape == (0,)

    with pytest.raises(ValueError, match="min of empty array"):
        empty.min()

    with pytest.raises(ValueError, match="max of empty array"):
        empty.max()

    with pytest.raises(ValueError, match="mean of empty array"):
        empty.mean()

    with pytest.raises(ValueError, match="argmin of empty array"):
        empty.argmin()


def test_empty_axis_reductions():
    matrix = zeros((0, 3))

    assert matrix.sum(axis=0).tolist() == [0, 0, 0]
    assert matrix.sum(axis=1).shape == (0,)
    assert matrix.prod(axis=1).tolist() == []

    with pytest.raises(
        ValueError, match="mean: cannot reduce axis 0 with size 0"
    ):
        matrix.mean(axis=0)


def test_empty_array_sorting_and_unique():
    empty = array([])

    assert sort(empty.copy()).shape == (0,)
    assert empty.argsort().shape == (0,)
    assert unique(empty).shape == (0,)

    matrix = zeros((0, 3))
    assert sort(matrix.copy(), axis=None).shape == (0,)
    assert matrix.argsort(axis=1).shape == (0, 3)


def test_empty_array_selection():
    empty = array([])
    mask = array([], dtype=bool)

    assert where(mask, empty, 0).shape == (0,)
    assert where(mask, 1, empty).shape == (0,)
    assert where(mask, empty, empty).shape == (0,)
    assert take(empty, array([], dtype=int)).shape == (0,)
    assert clip(empty, 0, 1).shape == (0,)
    assert empty[mask].shape == (0,)
    assert empty[array([], dtype=int)].shape == (0,)


def test_empty_array_linalg_and_diagonal():
    empty = array([])

    assert empty.dot(empty) == 0
    assert vdot(empty, empty) == 0
    assert outer(empty, empty).shape == (0, 0)
    assert outer(empty, array([1, 2])).shape == (0, 2)
    assert (zeros((0, 3)) @ ones((3, 2))).shape == (0, 2)

    assert diag(empty).shape == (0, 0)
    assert diagonal(zeros((0, 3))).shape == (0,)
    assert trace(zeros((0, 3))) == 0


def test_empty_array_manipulation():
    empty = array([])

    assert concatenate([empty, empty]).shape == (0,)
    assert concatenate([zeros((0, 3)), zeros((0, 3))]).shape == (0, 3)
    assert stack([empty, empty]).shape == (2, 0)

    with pytest.raises(ValueError):
        concatenate([])

    with pytest.raises(ValueError):
        stack([])
