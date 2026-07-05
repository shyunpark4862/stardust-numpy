"""Non-finite values: inf and nan handling across operations."""

from __future__ import annotations

import math

import pytest

from numpy import *  # noqa: F403

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")


def test_array_accepts_nan_and_inf():
    assert array([1.0, NAN, INF, NINF]).dtype is float


def test_division_by_zero_returns_non_finite():
    result = (
        array([1.0, 0.0, -1.0, NAN]) / array([0.0, 0.0, 0.0, 0.0])
    ).tolist()

    assert result[0] == INF
    assert math.isnan(result[1])
    assert result[2] == NINF
    assert math.isnan(result[3])


def test_floor_divide_and_mod_by_zero():
    floordiv = (array([1.0, 0.0]) // array([0.0, 0.0])).tolist()
    mod = (array([1.0, 0.0]) % array([0.0, 0.0])).tolist()

    assert floordiv[0] == INF
    assert math.isnan(floordiv[1])
    assert math.isnan(mod[0])
    assert math.isnan(mod[1])


def test_log_and_sqrt_handle_non_finite_inputs():
    values = array([1.0, 0.0, -1.0, NAN, INF])

    assert log(values).tolist()[0] == 0.0
    assert log(values).tolist()[1] == NINF
    assert math.isnan(log(values).tolist()[2])
    assert math.isnan(log(values).tolist()[3])
    assert log(values).tolist()[4] == INF

    roots = sqrt(array([4.0, -1.0, NAN, INF, NINF]))
    assert roots.tolist()[0] == 2.0
    assert math.isnan(roots.tolist()[1])
    assert math.isnan(roots.tolist()[2])
    assert roots.tolist()[3] == INF
    assert math.isnan(roots.tolist()[4])


def test_floor_ceil_and_round_preserve_nan():
    values = array([NAN, INF, NINF, 2.7])

    assert math.isnan(floor(values).tolist()[0])
    assert floor(values).tolist()[1] == INF
    assert floor(values).tolist()[2] == NINF
    assert math.isnan(round(array([NAN])).tolist()[0])


def test_power_returns_nan_for_invalid_real_exponent():
    result = power(array([-1.0, NAN]), array([0.5, 2.0]))

    assert math.isnan(result.tolist()[0])
    assert math.isnan(result.tolist()[1])


def test_reductions_propagate_nan_and_handle_inf():
    values = array([1.0, NAN, 3.0, INF, NINF])

    assert math.isnan(values.sum())
    assert math.isnan(values.mean())
    assert math.isnan(values.min())
    assert math.isnan(values.max())
    assert math.isnan(values.prod())
    assert values.any() is True
    assert values.all() is True


def test_argmin_and_argmax_return_first_nan_index():
    values = array([1.0, NAN, 3.0, INF, NINF])

    assert values.argmin() == 1
    assert values.argmax() == 1


def test_min_max_axis_propagates_nan():
    values = array([[1.0, NAN, 3.0], [4.0, 5.0, 6.0]])

    assert math.isnan(values.min(axis=1).tolist()[0])
    assert math.isnan(values.max(axis=0).tolist()[1])


def test_min_returns_immediately_when_first_value_is_nan():
    assert math.isnan(array([NAN, 1.0, 2.0]).min())


def test_argmin_argmax_axis_with_nan():
    values = array([[1.0, NAN, 3.0], [4.0, 5.0, 6.0]])

    assert values.argmin(axis=0).tolist()[1] == 0

    shifted = array([[1.0, 5.0, 3.0], [4.0, NAN, 6.0]])

    assert shifted.argmax(axis=0).tolist()[1] == 1


def test_is_nan_value_handles_complex_nan():
    from numpy._reductions import _is_nan_value

    assert _is_nan_value(complex(NAN, 0))


def test_sort_and_argsort_place_nan_last():
    values = array([3.0, NAN, 1.0, INF, NINF])

    assert sort(values.copy()).tolist() == [NINF, 1.0, 3.0, INF, NAN]
    assert values.argsort().tolist() == [4, 2, 0, 3, 1]
    assert values.ravel()[values.argsort(axis=None)].tolist() == [
        NINF,
        1.0,
        3.0,
        INF,
        NAN,
    ]


def test_unique_and_where_with_nan():
    assert math.isnan(unique(array([3.0, NAN, 1.0, NAN])).tolist()[-1])
    assert where(
        array([True, False, True]), array([1.0, NAN, 3.0]), 0
    ).tolist() == [
        1.0,
        0.0,
        3.0,
    ]


def test_clip_keeps_nan():
    clipped = clip(array([1.0, NAN, 20.0]), 0, 10)

    assert clipped.tolist()[0] == 1.0
    assert math.isnan(clipped.tolist()[1])
    assert clipped.tolist()[2] == 10.0


def test_astype_int_from_nan_still_raises():
    with pytest.raises(ValueError):
        array([NAN]).astype(int)


def test_complex_division_and_remainder_by_zero():
    assert (array([1 + 2j]) / array([0 + 0j])).tolist() == [
        complex(float("inf"), float("inf"))
    ]

    zero_over_zero = (array([0 + 0j]) / array([0 + 0j])).tolist()[0]
    assert math.isnan(zero_over_zero.real)
    assert math.isnan(zero_over_zero.imag)

    assert (array([0], dtype=int) // array([0], dtype=int)).tolist() == [0]

    remainder = (array([1 + 0j]) % array([0 + 0j])).tolist()[0]
    assert math.isnan(remainder.real)
    assert math.isnan(remainder.imag)

    assert math.isnan(power(array([-1.0]), array([0.5])).tolist()[0])


def test_real_ceil_preserves_inf_and_nan():
    assert ceil(array([float("inf")])).tolist() == [float("inf")]
    assert math.isnan(ceil(array([float("nan")])).tolist()[0])
    assert ceil(array([1.5])).tolist() == [2.0]


def test_float_ops_internal_non_finite_paths():
    from numpy._float_ops import (
        _signed_infinity,
        floor_divide,
        numeric_power,
        order_key,
        remainder,
    )

    assert math.isnan(_signed_infinity(complex(float("nan"), 1.0)).real)
    assert math.isnan(_signed_infinity(0 + 0j).real)
    assert math.isnan(_signed_infinity(0.0))
    assert _signed_infinity(-1 - 1j) == complex(float("-inf"), float("-inf"))

    assert order_key(complex(float("nan"), 0.0))[0] == 1
    assert order_key(complex(float("inf"), 0.0))[0] == 0
    assert order_key(3 + 4j) == (0, 0, 3.0, 4.0)
    assert order_key(5) == (0, 0, 5)

    class ZeroFloorDivFloat(float):
        def __floordiv__(self, other):
            raise ZeroDivisionError

    class ZeroModFloat(float):
        def __mod__(self, other):
            raise ZeroDivisionError

    assert floor_divide(ZeroFloorDivFloat(1.0), 1.0) == INF
    assert math.isnan(remainder(1.0, 0.0))
    assert math.isnan(remainder(ZeroModFloat(1.0), 2.0))

    class RaisingNumber:
        def __pow__(self, other):
            raise ValueError("boom")

    assert math.isnan(numeric_power(RaisingNumber(), 2))

    class RaisingComplex:
        def __pow__(self, other):
            raise OverflowError

    result = numeric_power(RaisingComplex(), 2 + 0j)

    assert math.isnan(result.real)
    assert math.isnan(result.imag)
