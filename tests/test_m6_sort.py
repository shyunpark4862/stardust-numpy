"""M6: sort, argsort and unique_*."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust_numpy as sdnp
from conftest import NP_DTYPE, assert_same

REAL = [sdnp.bool, sdnp.int64, sdnp.float64]


def sample(dtype, shape=(4, 37), seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(-5, 6, shape)  # many ties
    if dtype is sdnp.bool:
        return n > 0
    if dtype is sdnp.int64:
        return n
    f = n * 0.5
    f[rng.random(shape) < 0.1] = np.nan
    f[rng.random(shape) < 0.05] = -0.0
    return f


def reference_argsort(lane, descending):
    """Stable, NaN last in both directions, ties in original order."""
    def key(i):
        v = lane[i]
        nan = isinstance(v, float) and math.isnan(v)
        return (nan, 0 if nan else (-v if descending else v))
    return sorted(range(len(lane)), key=key)


@pytest.mark.parametrize("dtype", REAL, ids=str)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("descending", [False, True])
def test_stable_argsort_and_sort(dtype, axis, descending):
    n = sample(dtype)
    x = sdnp.asarray(n)
    got = np.asarray(sdnp.argsort(x, axis=axis, descending=descending))
    moved = np.moveaxis(n, axis, -1)
    want = np.array([reference_argsort(lane.tolist(), descending) for lane in moved.reshape(-1, moved.shape[-1])])
    want = np.moveaxis(want.reshape(moved.shape), -1, axis)
    assert np.array_equal(got, want)
    assert_same(sdnp.sort(x, axis=axis, descending=descending), np.take_along_axis(n, want, axis=axis))
    if not descending:  # numpy's stable sort is the same ordering (-0.0 and 0.0 tie and keep their order)
        assert_same(sdnp.sort(x, axis=axis), np.sort(n, axis=axis, stable=True))
        assert np.array_equal(got, np.argsort(n, axis=axis, stable=True))


@pytest.mark.parametrize("dtype", REAL, ids=str)
@pytest.mark.parametrize("descending", [False, True])
def test_unstable_sort_gives_sorted_values(dtype, descending):
    n = sample(dtype, (3, 2000), seed=1)
    x = sdnp.asarray(n)
    stable = np.asarray(sdnp.sort(x, descending=descending))
    # Equal values, but tied -0.0/+0.0 may come out in either order: compare as numbers.
    assert np.array_equal(np.asarray(sdnp.sort(x, descending=descending, stable=False)), stable, equal_nan=True)
    idx = np.asarray(sdnp.argsort(x, descending=descending, stable=False))
    assert np.array_equal(np.take_along_axis(n, idx, axis=-1), stable, equal_nan=True)
    assert all(sorted(r) == list(range(n.shape[1])) for r in idx.tolist())  # a permutation


@pytest.mark.parametrize("pattern", ["sorted", "reversed", "equal", "sawtooth", "random"])
@pytest.mark.parametrize("stable", [True, False])
def test_adversarial_inputs(pattern, stable):
    n = {"sorted": np.arange(5000.0), "reversed": np.arange(5000.0)[::-1].copy(), "equal": np.ones(5000),
         "sawtooth": np.tile(np.arange(50.0), 100),
         "random": np.random.default_rng(2).standard_normal(5000)}[pattern]
    assert_same(sdnp.sort(sdnp.asarray(n), stable=stable), np.sort(n))


def test_layouts_and_rules():
    n = np.random.default_rng(3).standard_normal((6, 7))
    for view in (n.T, n[::-1, ::2], np.asfortranarray(n)):
        assert_same(sdnp.sort(sdnp.asarray(view), axis=0), np.sort(view, axis=0))
    with pytest.raises(TypeError):
        sdnp.sort(sdnp.asarray([1j]))  # S5
    with pytest.raises(TypeError):
        sdnp.sort(sdnp.asarray([1.0]), axis=None)  # an int axis only
    with pytest.raises(ValueError):
        sdnp.sort(sdnp.asarray([1.0]), axis=1)
    with pytest.raises(TypeError):
        sdnp.sort(sdnp.asarray([1.0]), -1)  # keyword-only
    assert sdnp.sort(sdnp.zeros((3, 0))).shape == (3, 0)
    assert sdnp.argsort(sdnp.zeros(0)).dtype is sdnp.int64


def test_descending_keeps_tie_order_and_nan_last():
    x = sdnp.asarray([1.0, math.nan, 3.0, 1.0, 3.0])
    assert sdnp.argsort(x, descending=True).tolist() == [2, 4, 0, 3, 1]


# ------------------------------------------------------------------ unique

def reference_unique(flat):
    """Sorted distinct values; every NaN on its own; first occurrences."""
    def key(i):
        v = flat[i]
        if isinstance(v, complex):
            nan = math.isnan(v.real) or math.isnan(v.imag)
            return (nan, 0, 0) if nan else (False, v.real, v.imag)
        nan = isinstance(v, float) and math.isnan(v)
        return (nan, 0 if nan else v)
    order = sorted(range(len(flat)), key=key)
    groups = []
    for i in order:
        v = flat[i]
        if groups and flat[groups[-1][0]] == v:  # nan == nan is False: a new group
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


@pytest.mark.parametrize("dtype", [sdnp.bool, sdnp.int64, sdnp.float64, sdnp.complex128], ids=str)
def test_unique_all(dtype):
    n = sample(sdnp.float64 if dtype is sdnp.complex128 else dtype, (5, 7))
    if dtype is sdnp.complex128:
        n = n + 1j * sample(sdnp.float64, (5, 7), seed=4)
    x = sdnp.asarray(n)
    r = sdnp.unique_all(x)
    assert r._fields == ("values", "indices", "inverse_indices", "counts")
    flat = n.ravel().tolist()
    groups = reference_unique(flat)
    assert_same(r.values, np.array([flat[g[0]] for g in groups], dtype=NP_DTYPE[dtype]))
    assert r.indices.tolist() == [g[0] for g in groups]
    assert r.counts.tolist() == [len(g) for g in groups]
    assert r.inverse_indices.shape == n.shape  # Array API: the shape of x
    rebuilt = np.asarray(r.values)[np.asarray(r.inverse_indices)]
    assert np.array_equal(rebuilt, n, equal_nan=True)
    assert_same(sdnp.unique_values(x), np.asarray(r.values))
    assert sdnp.unique_counts(x)._fields == ("values", "counts")
    assert sdnp.unique_inverse(x)._fields == ("values", "inverse_indices")


def test_unique_nan_and_signed_zero():
    r = sdnp.unique_counts(sdnp.asarray([math.nan, 0.0, -0.0, math.nan, 1.0]))
    v = r.values.tolist()
    assert v[:2] == [0.0, 1.0] and math.copysign(1, v[0]) == 1  # first occurrence (+0.0) is kept
    assert all(math.isnan(t) for t in v[2:]) and len(v) == 4  # each NaN is its own value (N3)
    assert r.counts.tolist() == [2, 1, 1, 1]
    assert sdnp.unique_values(sdnp.zeros(0)).shape == (0,)


@given(hnp.arrays(np.int64, hnp.array_shapes(min_dims=1, max_dims=3, max_side=6), elements=st.integers(-4, 4)))
def test_unique_matches_numpy_on_integers(n):
    r = sdnp.unique_all(sdnp.asarray(n))
    w = np.unique_all(n)
    assert r.values.tolist() == w.values.tolist()
    assert r.counts.tolist() == w.counts.tolist()
    assert r.indices.tolist() == w.indices.tolist()
    assert np.array_equal(np.asarray(r.inverse_indices), w.inverse_indices.reshape(n.shape))


# ------------------------------------------------ M7: the NEON sorting path

@pytest.mark.parametrize("n", [31, 32, 33, 47, 48, 64, 100, 1000, 4097])
@pytest.mark.parametrize("kind", ["int", "float", "float-ties", "extremes"])
def test_simd_sort_sizes_and_values(n, kind):
    # Sizes around the 16-element blocks and the padding; values that equal the
    # padding (inf, INT64_MAX) and many ties.
    rng = np.random.default_rng(n)
    v = {
        "int": rng.integers(-50, 50, n),
        "float": rng.standard_normal(n),
        "float-ties": rng.integers(-3, 3, n) * 0.5 + 0.25,
        "extremes": np.where(rng.random(n) < 0.2, 2**63 - 1, rng.integers(-(2**63), 2**63 - 1, n)),
    }[kind]
    if kind == "float":
        v[rng.random(n) < 0.1] = np.inf
        v[rng.random(n) < 0.05] = np.nan
    for stable in (True, False):
        assert_same(sdnp.sort(sdnp.asarray(v), stable=stable), np.sort(v, stable=True))
        got = np.asarray(sdnp.sort(sdnp.asarray(v), descending=True, stable=stable))
        assert np.array_equal(got[~np.isnan(got)] if got.dtype.kind == "f" else got,
                              np.sort(v[~np.isnan(v)] if v.dtype.kind == "f" else v)[::-1])
    assert_same(sdnp.unique_values(sdnp.asarray(v)), np.asarray(sdnp.unique_all(sdnp.asarray(v)).values))


def test_stable_float_sort_with_both_zero_signs_keeps_order():
    # -0.0 == +0.0: here the order of equal values is visible, so the stable
    # sort must not take the (unstable) NEON path.
    v = np.array([0.0, -0.0] * 40 + [1.0, -1.0] * 10)
    got = np.asarray(sdnp.sort(sdnp.asarray(v)))
    assert np.array_equal(got.view(np.int64), np.sort(v, stable=True).view(np.int64))
    u = sdnp.unique_values(sdnp.asarray(v))
    assert u.tolist() == [-1.0, 0.0, 1.0] and math.copysign(1, u.tolist()[1]) == 1  # first occurrence: +0.0


@given(st.lists(st.integers(-(2**63), 2**63 - 1), min_size=0, max_size=300))
def test_simd_sort_random_int64(values):
    n = np.array(values, dtype=np.int64)
    assert_same(sdnp.sort(sdnp.asarray(n)), np.sort(n))


def test_many_lanes_short_and_long():
    # Short lanes reuse per-thread scratch, long ones allocate (review fix,
    # 2026-11-10): 4096 Keyed pairs (64 KB) is the last that fits.
    rng = np.random.default_rng(3)
    for shape in [(20_000, 8), (3, 4096), (3, 4097), (2, 50_000)]:
        n = rng.integers(-5, 5, shape).astype(np.float64)
        n[::7, ::3] = np.nan
        x = sdnp.asarray(n)
        assert_same(sdnp.sort(x, axis=1), np.sort(n, axis=1, kind="stable"))
        assert_same(sdnp.argsort(x, axis=1), np.argsort(n, axis=1, kind="stable"))
        assert_same(sdnp.sort(x, axis=1, descending=True), -np.sort(-n, axis=1, kind="stable"))

