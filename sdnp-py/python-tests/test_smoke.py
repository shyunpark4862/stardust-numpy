"""Smoke tests for the sdnp Python binding."""

import pytest

import sdnp


def test_import_and_zeros():
    a = sdnp.zeros((2, 3))
    assert a.shape == [2, 3]
    assert a.dtype is float


def test_add_and_shape():
    a = sdnp.zeros((2, 3))
    b = sdnp.ones((2, 3))
    c = a + b
    assert c.shape == [2, 3]


def test_full_reduce_unwraps_scalar():
    a = sdnp.ones((2, 3))
    result = sdnp.sum(a)
    assert not isinstance(result, sdnp.Array)
    assert result == 6.0


def test_basic_index_unwraps_scalar():
    a = sdnp.arange(0, 6).reshape((2, 3))
    value = a[0, 1]
    assert not isinstance(value, sdnp.Array)
    assert value == 1


def test_rejects_zero_d_creation():
    with pytest.raises(ValueError, match="0-dimensional"):
        sdnp.array(3)
    with pytest.raises(ValueError, match="0-dimensional"):
        sdnp.array([1], shape=())


def test_nan_policy_kwarg():
    a = sdnp.array([[1.0, float("nan")]], dtype=float)
    assert sdnp.sum(a, nan_policy="ignore") == 1.0


def test_excluded_api_not_in_all():
    excluded = {
        "item",
        "gather",
        "scatter",
        "broadcast_to",
        "IndexSpec",
    }
    assert excluded.isdisjoint(set(sdnp.__all__))


def test_read_only_broadcast_write_raises():
    x = sdnp.array([1.0, 2.0])
    y = sdnp.array([10.0, 20.0, 30.0])
    xx, _yy = sdnp.meshgrid(x, y)
    with pytest.raises(ValueError, match="read-only|ReadOnly|writable"):
        xx[0, 0] = 99
