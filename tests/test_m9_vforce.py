"""M9: the 16 float64 functions now run on Accelerate vForce.

The contract (DESIGN.md §2.6): within 8 ULP of numpy/libm, exact C99 Annex F
special values (signed zeros, infinities, NaN), and a result for an element
that does not depend on where the element is (position, length, layout,
thread count).
"""

import math

import numpy as np
import pytest

import stardust_numpy as sdnp
from conftest import assert_ulp

UNARY = {
    "exp": (-745, 710), "log": (1e-300, 1e300), "log2": (1e-300, 1e300), "log10": (1e-300, 1e300),
    "sin": (-1e5, 1e5), "asin": (-1, 1), "acos": (-1, 1), "atan": (-1e8, 1e8), "sinh": (-710, 710),
    "cosh": (-710, 710), "tanh": (-30, 30), "asinh": (-1e300, 1e300), "acosh": (1, 1e300), "atanh": (-1, 1),
}
SPECIAL = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 5e-324, -5e-324, 2.0, 0.5, -0.5, 1e308]


def wide(lo, hi, n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    if lo > 0 and hi / lo > 1e6:  # log-uniform over many decades
        return np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    if lo < 0 < hi and hi > 1e6:
        return np.sign(rng.uniform(-1, 1, n)) * np.exp(rng.uniform(-30, math.log(hi), n))
    return rng.uniform(lo, hi, n)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_accuracy(name):
    x = wide(*UNARY[name])
    with np.errstate(all="ignore"):
        assert_ulp(getattr(sdnp, name)(sdnp.asarray(x)), getattr(np, name)(x), 8)


@pytest.mark.parametrize("name", ["pow", "atan2"])
def test_binary_accuracy(name):
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 100, 50_000) if name == "pow" else rng.uniform(-1e3, 1e3, 50_000)
    b = rng.uniform(-60, 60, 50_000) if name == "pow" else rng.uniform(-1e3, 1e3, 50_000)
    with np.errstate(all="ignore"):
        assert_ulp(getattr(sdnp, name)(sdnp.asarray(a), sdnp.asarray(b)), getattr(np, name)(a, b), 8)


def same_special(got, want):
    """Bit-exact for zeros (sign!), infinities and NaN; 8 ULP elsewhere."""
    for g, w in zip(np.asarray(got).ravel().tolist(), np.asarray(want).ravel().tolist()):
        if math.isnan(w):
            assert math.isnan(g), (g, w)
        elif w == 0 or math.isinf(w):
            assert g == w and math.copysign(1, g) == math.copysign(1, w), (g, w)
        else:
            assert abs(g - w) <= 8 * math.ulp(w), (g, w)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_special_values(name):
    x = np.array(SPECIAL)
    with np.errstate(all="ignore"):
        same_special(getattr(sdnp, name)(sdnp.asarray(x)), getattr(np, name)(x))


@pytest.mark.parametrize("name", ["pow", "atan2"])
def test_binary_special_values(name):
    v = np.array(SPECIAL + [3.0, -3.0])
    a, b = (np.ascontiguousarray(t) for t in np.broadcast_arrays(v[:, None], v[None, :]))
    with np.errstate(all="ignore"):
        same_special(getattr(sdnp, name)(sdnp.asarray(a), sdnp.asarray(b)), getattr(np, name)(a, b))


@pytest.mark.parametrize("name", ["exp", "sin", "atan", "tanh", "log"])
def test_position_independence(name):
    # The same input gives the same bits whatever the surroundings: a Python
    # scalar (n = 1), any offset, any stride, a reversed view.
    x = np.abs(wide(-50, 50, 5000, seed=3)) + 0.1
    fn = getattr(sdnp, name)
    full = np.asarray(fn(sdnp.asarray(x)))
    bits = full.view(np.int64)
    for i in (0, 1, 7, 1023, 1024, 4999):
        assert np.float64(fn(float(x[i]))).view(np.int64) == bits[i]
    assert np.array_equal(np.asarray(fn(sdnp.asarray(x[5:]))).view(np.int64), bits[5:])
    assert np.array_equal(np.asarray(fn(sdnp.asarray(x[::3]))).view(np.int64), bits[::3])
    assert np.array_equal(np.asarray(fn(sdnp.asarray(x[::-1]))).view(np.int64), bits[::-1])
    two_d = sdnp.asarray(x[:4900].reshape(70, 70).T)  # transposed: the strided gather path
    assert np.array_equal(np.asarray(fn(two_d)).view(np.int64), bits[:4900].reshape(70, 70).T)


def test_scalar_and_broadcast_operands_of_pow():
    x = wide(0.1, 10, 3000, seed=4)
    full = np.asarray(sdnp.pow(sdnp.asarray(x), sdnp.asarray(np.full(3000, 2.5)))).view(np.int64)
    assert np.array_equal(np.asarray(sdnp.pow(sdnp.asarray(x), 2.5)).view(np.int64), full)
    assert np.array_equal(np.asarray(sdnp.asarray(x) ** 2.5).view(np.int64), full)
    assert np.float64(sdnp.pow(float(x[17]), 2.5)).view(np.int64) == full[17]


def test_integer_inputs_go_through_buffered_cast():
    i = np.arange(-500, 500)
    with np.errstate(all="ignore"):
        assert_ulp(sdnp.exp(sdnp.asarray(i % 50)), np.exp((i % 50).astype(float)), 8)
        assert_ulp(sdnp.atan2(sdnp.asarray(i), sdnp.asarray(i[::-1])), np.atan2(i, i[::-1]).astype(float), 8)


def test_thread_count_does_not_change_vforce_results():
    x = sdnp.asarray(wide(-100, 100, 1_000_000, seed=5))
    before = sdnp.get_num_threads()
    try:
        out = []
        for n in (1, 3, 10):
            sdnp.set_num_threads(n)
            out.append((np.asarray(sdnp.sin(x)).tobytes(), np.asarray(sdnp.pow(x, 1.5)).tobytes()))
        assert out[0] == out[1] == out[2]
    finally:
        sdnp.set_num_threads(before)
