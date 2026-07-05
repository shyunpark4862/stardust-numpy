"""Complex number operations and vdot."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403


def test_conjugate_on_complex_array():
    a = array([1 + 2j, 3 - 4j])

    assert conjugate(a).tolist() == [1 - 2j, 3 + 4j]
    assert a.conjugate().tolist() == [1 - 2j, 3 + 4j]
    assert a.conj().tolist() == [1 - 2j, 3 + 4j]
    assert conj(a).tolist() == [1 - 2j, 3 + 4j]


def test_conjugate_preserves_real_dtype():
    a = array([1, 2, 3])

    assert conjugate(a).tolist() == [1, 2, 3]
    assert conjugate(a).dtype is int


def test_real_and_imag_properties():
    a = array([1 + 2j, 3 - 4j])

    assert a.real.tolist() == [1.0, 3.0]
    assert a.imag.tolist() == [2.0, -4.0]
    assert real(a).tolist() == [1.0, 3.0]
    assert imag(a).tolist() == [2.0, -4.0]


def test_real_and_imag_for_real_array():
    a = array([1, 2, 3])

    assert a.real.tolist() == [1.0, 2.0, 3.0]
    assert a.imag.tolist() == [0.0, 0.0, 0.0]


def test_absolute_on_complex_returns_float():
    a = array([3 + 4j])

    assert absolute(a).tolist() == [5.0]
    assert absolute(a).dtype is float
    assert abs(a).tolist() == [5.0]
    assert abs(a).dtype is float


def test_angle_on_complex_array():
    a = array([1 + 0j, 0 + 1j, -1 + 0j])
    result = angle(a).tolist()

    assert result[0] == 0.0
    assert abs(result[1] - math.pi / 2) < 1e-12
    assert abs(result[2] - math.pi) < 1e-12


def test_vdot_conjugates_first_operand():
    a = array([1 + 2j, 3 + 4j])
    b = array([1 + 1j, 1 + 1j])

    assert vdot(a, b) == (1 - 2j) * (1 + 1j) + (3 - 4j) * (1 + 1j)
    assert a.vdot(b) == vdot(a, b)


def test_vdot_with_real_vectors():
    a = array([1, 2, 3])
    b = array([4, 5, 6])

    assert vdot(a, b) == 32


def test_vdot_shape_error():
    a = array([1, 2])
    b = array([1, 2, 3])

    with pytest.raises(ValueError):
        vdot(a, b)
