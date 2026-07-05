"""Reductions: sum, prod, min, max, mean, var, and related methods."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_sum_without_axis():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.sum() == 21
    assert a.sum(axis=None) == 21


def test_sum_axis_none_keepdims():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a.sum(axis=None, keepdims=True)

    assert result.shape == (1, 1)
    assert result.tolist() == [[21]]


def test_prod_without_axis():
    a = array([[1, 2], [3, 4]])

    assert a.prod() == 24


def test_min_max_without_axis():
    a = array([[3, 8], [-2, 5]])

    assert a.min() == -2
    assert a.max() == 8


def test_empty_sum_and_prod():
    a = array([])

    assert a.sum() == 0
    assert a.prod() == 1


def test_empty_min_max_raise():
    a = array([])

    with pytest.raises(ValueError):
        a.min()

    with pytest.raises(ValueError):
        a.max()


def test_sum_axis_zero():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.sum(axis=0).tolist() == [5, 7, 9]


def test_sum_axis_one():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.sum(axis=1).tolist() == [6, 15]


def test_sum_negative_axis():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.sum(axis=-1).tolist() == [6, 15]


def test_prod_axis_zero():
    a = array([[1, 2], [3, 4]])

    assert a.prod(axis=0).tolist() == [3, 8]


def test_min_max_axis_zero():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.min(axis=0).tolist() == [2, 5, 1]
    assert a.max(axis=0).tolist() == [3, 8, 9]


def test_min_max_axis_one():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.min(axis=1).tolist() == [1, 2]
    assert a.max(axis=1).tolist() == [8, 9]


def test_min_max_negative_axis():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.min(axis=-1).tolist() == [1, 2]


def test_sum_keepdims_axis_zero():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a.sum(axis=0, keepdims=True)

    assert result.shape == (1, 3)
    assert result.tolist() == [[5, 7, 9]]


def test_sum_keepdims_axis_one():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a.sum(axis=1, keepdims=True)

    assert result.shape == (2, 1)
    assert result.tolist() == [[6], [15]]


def test_min_keepdims():
    a = array([[3, 8, 1], [2, 5, 9]])

    result = a.min(axis=1, keepdims=True)

    assert result.shape == (2, 1)
    assert result.tolist() == [[1], [2]]


def test_sum_multiple_axes():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    assert a.sum(axis=(0, 2)).tolist() == [14, 22]


def test_sum_multiple_axes_keepdims():
    a = array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    result = a.sum(axis=(0, 2), keepdims=True)

    assert result.shape == (1, 2, 1)
    assert result.tolist() == [
        [[14], [22]],
    ]


def test_duplicate_axis_raises():
    a = array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        a.sum(axis=(0, 0))


def test_mean_all():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.mean() == 3.5


def test_mean_axis_zero():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.mean(axis=0).tolist() == [2.5, 3.5, 4.5]


def test_mean_axis_one_keepdims():
    a = array([[1, 2, 3], [4, 5, 6]])

    result = a.mean(axis=1, keepdims=True)

    assert result.shape == (2, 1)
    assert result.tolist() == [[2.0], [5.0]]


def test_any_all_without_axis():
    a = array([[0, 0], [0, 3]])

    assert a.any() is True
    assert a.all() is False


def test_any_all_axis():
    a = array([[0, 2], [3, 0]])

    assert a.any(axis=0).tolist() == [True, True]
    assert a.all(axis=0).tolist() == [False, False]


def test_empty_any_all():
    a = array([])

    assert a.any() is False
    assert a.all() is True


def test_count_nonzero_all():
    a = array([[0, 1, 0], [2, 3, 0]])

    assert a.count_nonzero() == 3


def test_count_nonzero_axis():
    a = array([[0, 1, 0], [2, 3, 0]])

    assert a.count_nonzero(axis=0).tolist() == [1, 2, 0]
    assert a.count_nonzero(axis=1).tolist() == [1, 2]


def test_count_nonzero_keepdims():
    a = array([[0, 1], [2, 0]])

    result = a.count_nonzero(axis=1, keepdims=True)

    assert result.shape == (2, 1)
    assert result.tolist() == [[1], [1]]


def test_var_all():
    a = array([1, 2, 3, 4])

    assert a.var() == 1.25


def test_std_all():
    a = array([1, 2, 3, 4])

    assert a.std() == 1.25**0.5


def test_var_axis_zero():
    a = array([[1, 2], [3, 4]])

    assert a.var(axis=0).tolist() == [1.0, 1.0]


def test_var_axis_one_keepdims():
    a = array([[1, 3], [5, 9]])

    result = a.var(axis=1, keepdims=True)

    assert result.shape == (2, 1)
    assert result.tolist() == [[1.0], [4.0]]


def test_argmin_argmax():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.argmin() == 2
    assert a.argmax() == 5


def test_argmin_argmax_first_occurrence():
    a = array([4, 1, 3, 1])

    assert a.argmin() == 1


def test_argmin_axis_zero():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.argmin(axis=0).tolist() == [1, 1, 0]


def test_argmax_axis_one():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.argmax(axis=1).tolist() == [1, 2]


def test_argmin_negative_axis():
    a = array([[3, 8, 1], [2, 5, 9]])

    assert a.argmin(axis=-1).tolist() == [2, 0]


def test_cumsum_axis_one():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.cumsum(axis=1).tolist() == [
        [1, 3, 6],
        [4, 9, 15],
    ]


def test_cumsum_axis_zero():
    a = array([[1, 2, 3], [4, 5, 6]])

    assert a.cumsum(axis=0).tolist() == [
        [1, 2, 3],
        [5, 7, 9],
    ]


def test_cumsum_without_axis():
    a = array([[1, 2], [3, 4]])

    assert a.cumsum().tolist() == [1, 3, 6, 10]


def test_cumprod_axis_one():
    a = array([[2, 3], [4, 5]])

    assert a.cumprod(axis=1).tolist() == [
        [2, 6],
        [4, 20],
    ]


def test_cumsum_empty_axis_raises():
    a = array([[]])

    with pytest.raises(ValueError):
        a.cumsum(axis=1)


def test_empty_mean_and_var_raise():
    a = array([])

    with pytest.raises(ValueError):
        a.mean()

    with pytest.raises(ValueError):
        a.var()


def test_empty_argmin_and_argmax_raise():
    a = array([])

    with pytest.raises(ValueError):
        a.argmin()

    with pytest.raises(ValueError):
        a.argmax()


def test_argmin_and_argmax_reject_empty_axis():
    a = zeros((2, 0))

    with pytest.raises(ValueError):
        a.argmin(axis=1)

    with pytest.raises(ValueError):
        a.argmax(axis=1)


def test_axis_reduction_can_return_scalar():
    assert array([5]).min(axis=0) == 5


def test_reduction_rejects_empty_axis():
    a = zeros((2, 0))

    with pytest.raises(ValueError):
        a.min(axis=1)

    with pytest.raises(ValueError):
        a.mean(axis=1)


def test_reduction_rejects_non_tuple_axis_sequence():
    a = array([[1, 2], [3, 4]])

    with pytest.raises(TypeError):
        a.sum(axis=[0, 1])

    with pytest.raises(TypeError):
        a.sum(axis=1.5)


def test_reduction_rejects_out_of_range_axis():
    a = array([1, 2, 3])

    with pytest.raises(ValueError):
        a.sum(axis=3)


def test_cumprod_without_axis():
    a = array([[2, 3], [4, 5]])

    assert a.cumprod().tolist() == [2, 6, 24, 120]


def test_cumprod_empty_axis_raises():
    a = zeros((2, 0))

    with pytest.raises(ValueError):
        a.cumprod(axis=1)


def test_var_on_non_contiguous_array():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T.astype(float)
    reference = array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])

    assert matrix.var() == reference.var()


def test_cumsum_complex_uses_complex_fill():
    values = array([[1 + 1j, 2 + 0j], [3 + 0j, 4 + 0j]])

    assert values.cumsum(axis=1).tolist() == [
        [1 + 1j, 3 + 1j],
        [3 + 0j, 7 + 0j],
    ]


def test_cumsum_float_uses_float_fill():
    values = array([[1.0, 2.0], [3.0, 4.0]])

    assert values.cumsum(axis=1).tolist() == [
        [1.0, 3.0],
        [3.0, 7.0],
    ]
