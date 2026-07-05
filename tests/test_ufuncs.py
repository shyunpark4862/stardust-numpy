"""Universal functions and the out / where keyword arguments."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_functional_ufuncs():
    a = array([1, 2, 3])

    assert add(a, 10).tolist() == [11, 12, 13]
    assert multiply(a, 2).tolist() == [2, 4, 6]
    assert absolute(array([-1, 2, -3])).tolist() == [1, 2, 3]


def test_ufunc_out():
    a = array([1, 2, 3])
    out = zeros(3)

    result = add(a, 10, out=out)

    assert result is out
    assert out.tolist() == [11, 12, 13]


def test_ufunc_out_shape_mismatch():
    a = array([1, 2, 3])
    out = zeros((2, 2))

    with pytest.raises(ValueError):
        add(a, 1, out=out)


def test_unary_ufuncs():
    a = array([1, 4, 9])

    assert square(a).tolist() == [1, 16, 81]
    assert sqrt(a).tolist() == [1.0, 2.0, 3.0]


def test_logical_not():
    a = array([0, 1, 2, 3], dtype=bool)

    assert logical_not(a).tolist() == [True, False, False, False]
    assert logical_not(a).dtype is bool


def test_logical_and_and_or():
    a = array([True, False, True])
    b = array([True, True, False])

    assert logical_and(a, b).tolist() == [True, False, False]
    assert logical_or(a, b).tolist() == [True, True, True]
    assert logical_and(a, b).dtype is bool


def test_round():
    a = array([1.2, 2.5, 3.8])

    assert round(a).tolist() == [1, 2, 4]


def test_math_ufuncs():
    a = array([0.0, 1.0])

    assert exp(a).tolist() == [1.0, math.e]
    assert isnan(array([1.0, float("nan")])).tolist() == [False, True]


def test_ufunc_where_false_skips_computation():
    a = array([1, 2, 3])
    out = array([100, 100, 100])

    result = add(a, 10, out=out, where=False)

    assert result is out
    assert out.tolist() == [100, 100, 100]


def test_ufunc_rejects_invalid_where_type():
    a = array([1, 2, 3])

    with pytest.raises(TypeError):
        add(a, 1, where=1.5)


def test_ufunc_where_mask_and_out_validation():
    values = array([1, 2, 3])
    mask = array([True, False, True])

    assert add(values, 10, where=mask).tolist() == [11, 0, 13]

    with pytest.raises(
        TypeError, match="where must be bool or an array with dtype bool"
    ):
        add(values, 1, where=mask.astype(int))

    with pytest.raises(TypeError, match="out has dtype"):
        add(array([1.0, 2.0, 3.0]), 1, out=zeros(3, dtype=int))


def test_ufunc_out_with_contiguous_array_operands():
    left = array([1.0, 2.0, 3.0])
    right = array([4.0, 5.0, 6.0])
    out = zeros(3, dtype=float)

    result = add(left, right, out=out)

    assert result is out
    assert out.tolist() == [5.0, 7.0, 9.0]


def test_ufunc_out_with_non_contiguous_operands():
    left = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    right = array([[1, 1, 1], [1, 1, 1]])
    out = zeros((2, 3))

    result = add(left, right, out=out)

    assert result is out
    assert out.tolist() == [[2, 4, 6], [3, 5, 7]]


def test_ufunc_out_with_non_contiguous_scalar_operand():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    out = zeros((2, 3))

    result = add(matrix, 1, out=out)

    assert result is out
    assert out.tolist() == [[2, 4, 6], [3, 5, 7]]
