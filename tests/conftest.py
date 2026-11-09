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


def _ordered(x):
    """float64 bits mapped to integers that are ordered like the floats, so
    the difference of two of them is their distance in ULPs."""
    return [b if b >= 0 else -(2**63) - b for b in np.asarray(x, np.float64).ravel().view(np.int64).tolist()]


def _max_ulp(got, want):
    gn, wn = np.isnan(got).ravel(), np.isnan(want).ravel()
    assert np.array_equal(gn, wn), "nan positions differ"
    dist = [abs(g - w) for g, w, n in zip(_ordered(got), _ordered(want), gn) if not n]
    return max(dist, default=0)


def assert_ulp(arr, expected, max_ulp):
    """Like assert_same, but floats may differ by up to `max_ulp` units in the
    last place (the accuracy contract of the vForce functions, DESIGN.md §2.6)."""
    expected = np.asarray(expected)
    assert isinstance(arr, sdnp.ndarray), type(arr)
    assert arr.shape == expected.shape
    assert NP_DTYPE[arr.dtype] == expected.dtype, (arr.dtype, expected.dtype)
    got = np.asarray(arr)
    if expected.dtype.kind == "c":
        worst = max(_max_ulp(got.real, expected.real), _max_ulp(got.imag, expected.imag))
    else:
        worst = _max_ulp(got, expected)
    assert worst <= max_ulp, f"{worst} ULP apart\n got: {got!r}\nwant: {expected!r}"
