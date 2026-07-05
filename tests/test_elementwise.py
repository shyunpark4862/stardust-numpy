"""Elementwise operators, comparisons, and logical ops on ndarrays."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_1d_array_broadcasts_with_scalar_operands():
    a = array([1, 2, 3])

    assert (a + 1).tolist() == [2, 3, 4]
    assert (a * 10).tolist() == [10, 20, 30]
    assert (-a).tolist() == [-1, -2, -3]
    assert abs(array([-1, 2, -3])).tolist() == [1, 2, 3]


def test_1d_arrays_support_elementwise_binary_ops():
    a = array([1, 2, 3])
    b = array([10, 20, 30])

    assert (a + b).tolist() == [11, 22, 33]
    assert (b - a).tolist() == [9, 18, 27]


def test_1d_array_binary_ops_reject_incompatible_shapes():
    a = array([1, 2, 3])
    b = array([10, 20])

    with pytest.raises(ValueError):
        a + b


def test_1d_array_supports_reverse_scalar_operands():
    a = array([1, 2, 4])

    assert (10 + a).tolist() == [11, 12, 14]
    assert (10 - a).tolist() == [9, 8, 6]
    assert (12 / a).tolist() == [12.0, 6.0, 3.0]
    assert (2**a).tolist() == [2, 4, 16]


def test_comparison_with_scalar():
    a = array([1, 2, 3])

    assert (a == 2).tolist() == [False, True, False]
    assert (a != 2).tolist() == [True, False, True]
    assert (a > 1).tolist() == [False, True, True]
    assert (a >= 2).tolist() == [False, True, True]
    assert (a < 3).tolist() == [True, True, False]
    assert (a <= 2).tolist() == [True, True, False]


def test_comparison_between_arrays():
    a = array([1, 5, 3])
    b = array([2, 5, 1])

    assert (a == b).tolist() == [False, True, False]
    assert (a > b).tolist() == [False, False, True]


def test_logical_operations():
    a = array([True, False, True])
    b = array([True, True, False])

    assert (a & b).tolist() == [True, False, False]
    assert (a | b).tolist() == [True, True, True]
    assert (~a).tolist() == [False, True, False]


def test_comparison_masks_can_be_combined():
    a = array([-2, 0, 3, 8, 15])

    mask = (a > 0) & (a < 10)

    assert mask.tolist() == [False, False, True, True, False]


def test_logical_operations_convert_values_to_bool():
    a = array([0, 2, -3])
    b = array([5, 0, 7])

    assert (a & b).tolist() == [False, False, True]
    assert (a | b).tolist() == [True, True, True]
    assert (~a).tolist() == [True, False, False]


def test_2d_scalar_operations():
    a = array([[1, 2], [3, 4]])

    assert (a + 10).tolist() == [
        [11, 12],
        [13, 14],
    ]


def test_2d_array_operations():
    a = array([[1, 2], [3, 4]])
    b = array([[10, 20], [30, 40]])

    assert (a + b).tolist() == [
        [11, 22],
        [33, 44],
    ]


def test_2d_comparison():
    a = array([[1, 2], [3, 4]])

    assert (a > 2).tolist() == [
        [False, False],
        [True, True],
    ]


def test_floor_division_and_modulo():
    a = array([7, 8, 9])

    assert (a // 2).tolist() == [3, 4, 4]
    assert (a % 2).tolist() == [1, 0, 1]
    assert (17 // a).tolist() == [2, 2, 1]
    assert (17 % a).tolist() == [3, 1, 8]


def test_reverse_logical_operators():
    a = array([True, False, True])

    assert (True & a).tolist() == [True, False, True]
    assert (False | a).tolist() == [True, False, True]


def test_reverse_multiplication():
    a = array([1, 2, 3])

    assert (2 * a).tolist() == [2, 4, 6]


def test_binary_op_rejects_invalid_where_type():
    a = array([1, 2, 3])

    with pytest.raises(TypeError):
        a._binary_op(1, lambda x, y: x + y, where=1.5)


def test_unary_non_contiguous():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T

    assert (-matrix).tolist() == [[-1, -3, -5], [-2, -4, -6]]


def test_binary_scalar_non_contiguous():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T

    assert (matrix + 10).tolist() == [[11, 13, 15], [12, 14, 16]]


def test_binary_array_non_contiguous():
    left = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T
    right = array([[1, 1, 1], [1, 1, 1]])

    assert (left + right).tolist() == [[2, 4, 6], [3, 5, 7]]


def test_power_non_contiguous():
    matrix = array([1, 2, 3, 4, 5, 6]).reshape(3, 2).T

    assert (matrix**2).tolist() == [[1, 9, 25], [4, 16, 36]]
