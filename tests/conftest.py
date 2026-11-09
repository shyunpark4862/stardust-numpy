"""Shared helpers: numpy is the reference ("oracle") for every test."""

import numpy as np
import pytest

import stardust_numpy as sdnp

NP_DTYPE = {
    sdnp.bool: np.dtype(np.bool_),
    sdnp.int64: np.dtype(np.int64),
    sdnp.float64: np.dtype(np.float64),
    sdnp.complex128: np.dtype(np.complex128),
}
ALL_DTYPES = list(NP_DTYPE)


def _bits_equal(got, want):
    """Element-wise identical bits; any NaN matches any NaN, but -0.0 != +0.0."""
    if got.dtype.kind == "c":
        return _bits_equal(got.real.copy(), want.real.copy()) and _bits_equal(got.imag.copy(), want.imag.copy())
    if got.dtype.kind == "f":
        gn, wn = np.isnan(got), np.isnan(want)
        return np.array_equal(gn, wn) and np.array_equal(got[~gn].view(np.int64), want[~wn].view(np.int64))
    return np.array_equal(got, want)


def assert_same(arr, expected):
    """`arr` (stardust) has exactly the shape, dtype and bits of `expected` (numpy)."""
    expected = np.asarray(expected)
    assert isinstance(arr, sdnp.ndarray), type(arr)
    assert arr.shape == expected.shape
    assert NP_DTYPE[arr.dtype] == expected.dtype, (arr.dtype, expected.dtype)
    got = np.asarray(arr)  # zero-copy via the buffer protocol
    assert _bits_equal(got, expected), f"\n got: {got!r}\nwant: {expected!r}"


@pytest.fixture(params=ALL_DTYPES, ids=str)
def dtype(request):
    return request.param
