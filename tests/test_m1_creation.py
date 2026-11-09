"""M1: zeros/ones/empty/full, *_like, arange, linspace, eye."""

import math

import numpy as np
import pytest

import stardust_numpy as sdnp
from conftest import NP_DTYPE, assert_same

SHAPES = [3, (2, 3), (0,), (4, 0, 2), (1, 1, 1)]


@pytest.mark.parametrize("shape", SHAPES)
def test_zeros_ones(shape, dtype):
    assert_same(sdnp.zeros(shape, dtype=dtype), np.zeros(shape, dtype=NP_DTYPE[dtype]))
    assert_same(sdnp.ones(shape, dtype=dtype), np.ones(shape, dtype=NP_DTYPE[dtype]))


def test_defaults_are_float64():
    assert sdnp.zeros(2).dtype is sdnp.float64
    assert sdnp.ones(2).dtype is sdnp.float64
    assert sdnp.empty(2).dtype is sdnp.float64
    assert sdnp.eye(2).dtype is sdnp.float64


def test_empty_has_requested_shape(dtype):
    a = sdnp.empty((3, 2), dtype=dtype)
    assert a.shape == (3, 2) and a.dtype is dtype


@pytest.mark.parametrize("bad", [[2, 3], 2.0, True, (2, 3.0), "3"])
def test_shape_must_be_int_or_tuple(bad):
    with pytest.raises(TypeError):
        sdnp.zeros(bad)


@pytest.mark.parametrize("bad", [(), (-1,), (2, -3)])
def test_bad_shapes_are_value_errors(bad):
    with pytest.raises(ValueError):
        sdnp.zeros(bad)


@pytest.mark.parametrize("value", [True, 7, -2.5, 1 + 2j, math.inf])
def test_full_infers_dtype(value):
    assert_same(sdnp.full((2, 2), value), np.full((2, 2), value))


def test_full_with_explicit_dtype_uses_explicit_rules():
    assert sdnp.full(3, 2.7, dtype=sdnp.int64).tolist() == [2, 2, 2]
    assert sdnp.full(2, 0.0, dtype=sdnp.bool).tolist() == [False, False]
    with pytest.raises(TypeError):
        sdnp.full(2, 1j, dtype=sdnp.float64)
    with pytest.raises(TypeError):
        sdnp.full(2, "x")


def test_like_functions_keep_shape_dtype_and_layout():
    f_order = sdnp.asarray(np.asfortranarray(np.arange(6.0).reshape(2, 3)))
    for fn in (sdnp.zeros_like, sdnp.ones_like, sdnp.empty_like):
        out = fn(f_order)
        assert out.shape == (2, 3) and out.dtype is sdnp.float64
        assert out._flags()["f_contiguous"] and not out._flags()["c_contiguous"]
    assert_same(sdnp.ones_like(f_order, dtype=sdnp.int64), np.ones((2, 3), dtype=np.int64))
    assert_same(sdnp.full_like(f_order, 3), np.full((2, 3), 3.0))
    with pytest.raises(TypeError):
        sdnp.zeros_like([1, 2])


# ------------------------------------------------------------- arange ----

@pytest.mark.parametrize(
    "args",
    [(5,), (2, 9), (9, 2), (2, 9, 3), (9, 2, -3), (-5, 5, 2), (0,), (3, 3), (True, 4), (0, 10**12, 10**11)],
)
def test_arange_int(args):
    assert_same(sdnp.arange(*args), np.arange(*args))


@pytest.mark.parametrize("args", [(2.5,), (0, 1, 0.1), (1, 0, -0.25), (0.5, 3), (-1, 1, 0.3), (0, 1e-3, 1e-4)])
def test_arange_float_length_and_values(args):
    got = np.asarray(sdnp.arange(*args))
    want = np.arange(*args)
    assert got.shape == want.shape and got.dtype == np.float64
    start = args[0] if len(args) > 1 else 0
    step = args[2] if len(args) > 2 else 1
    # Ours is exactly start + i*step (numpy derives its step as (start+step)-start),
    # so the two may differ by a few ULPs of the range's magnitude.
    assert np.array_equal(got, start + np.arange(len(want)) * float(step))
    scale = max(abs(start), abs(args[1] if len(args) > 1 else args[0]), abs(step))
    assert np.allclose(got, want, rtol=0, atol=8 * np.finfo(np.float64).eps * scale)


def test_arange_dtype_and_errors():
    assert sdnp.arange(0, 2, 0.5, dtype=sdnp.int64).tolist() == [0, 0, 1, 1]
    assert sdnp.arange(3, dtype=sdnp.complex128).tolist() == [0j, 1 + 0j, 2 + 0j]
    with pytest.raises(ValueError):
        sdnp.arange(0, 5, 0)
    with pytest.raises(TypeError):
        sdnp.arange(1j)
    with pytest.raises(TypeError):
        sdnp.arange(0, 5, dtype=float)


# ----------------------------------------------------------- linspace ----

@pytest.mark.parametrize(
    "start,stop,num,endpoint",
    [(0, 1, 5, True), (0, 1, 5, False), (-3, 7.5, 11, True), (1, 0, 4, True), (2, 2, 3, True),
     (0, 1, 1, True), (0, 1, 1, False), (0, 1, 0, True), (0, 1e-320, 3, True), (5, -5, 100, False)],
)
def test_linspace_matches_numpy_bitwise(start, stop, num, endpoint):
    assert_same(sdnp.linspace(start, stop, num, endpoint=endpoint), np.linspace(start, stop, num, endpoint=endpoint))


def test_linspace_complex_and_dtype():
    got = np.asarray(sdnp.linspace(0, 1 + 2j, 5))
    assert got.dtype == np.complex128
    assert np.allclose(got, np.linspace(0, 1 + 2j, 5), rtol=1e-15, atol=0)
    assert sdnp.linspace(0, 3, 4, dtype=sdnp.int64).tolist() == [0, 1, 2, 3]
    with pytest.raises(TypeError):
        sdnp.linspace(0, 1j, 3, dtype=sdnp.float64)
    with pytest.raises(ValueError):
        sdnp.linspace(0, 1, -1)
    with pytest.raises(TypeError):
        sdnp.linspace(0, 1)  # num is required


# ---------------------------------------------------------------- eye ----

@pytest.mark.parametrize("n,m,k", [(3, None, 0), (3, 5, 1), (4, 2, -1), (3, 3, 5), (3, 3, -5), (0, 2, 0)])
def test_eye(n, m, k, dtype):
    assert_same(sdnp.eye(n, m, k=k, dtype=dtype), np.eye(n, m, k=k, dtype=NP_DTYPE[dtype]))
