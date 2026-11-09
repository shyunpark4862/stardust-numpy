"""M6: advanced indexing (integer arrays, boolean masks): gather and scatter."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import stardust_numpy as sdnp
from conftest import assert_same

N = np.arange(60).reshape(3, 4, 5)


def keys():
    m = N > 30
    return [
        [0, 2], [-1, 0, -1], np.array([[0, 1], [2, 0]]), (slice(None), [1, 3]), ([0, 1], [1, 2]),
        ([[0], [2]], [1, 3]), (Ellipsis, [0, -1]), ([0, 2], Ellipsis), (1, [0, 2]), ([1, 2], 3),
        (slice(1, None), [0, 0], [4, 1]), (None, [0, 1]), ([0, 1], slice(None), slice(None)),
        m, (0, m[0]), (slice(None), m[0]), (m[:, :, 0],), (m[:, :, 0], slice(None)), ([0, 2], N[0, :, 0] > 6),
        [], ([], slice(None)), np.array([], dtype=np.int64), [True, False, True], (2, 3, [0, 4]),
    ]


@pytest.mark.parametrize("key", keys(), ids=lambda k: repr(k)[:40])
def test_getitem_matches_numpy(key):
    x = sdnp.asarray(N)
    want = N[key]
    sd_key = tuple(sdnp.asarray(k) if isinstance(k, np.ndarray) else k for k in key) if isinstance(key, tuple) \
        else (sdnp.asarray(key) if isinstance(key, np.ndarray) else key)
    got = x[sd_key]
    assert_same(got, want)
    assert not np.shares_memory(np.asarray(got), N)  # always a copy
    assert_same(x[key], want)  # numpy arrays work as keys too


def test_strided_source_and_indices():
    base = np.arange(100.0).reshape(10, 10)
    view = base[::-2, 1::3]
    idx = np.array([[4, 0], [1, 3]])[:, ::-1]
    assert_same(sdnp.asarray(view)[sdnp.asarray(idx)], view[idx])
    assert_same(sdnp.asarray(view)[:, sdnp.asarray([2, 0])], view[:, [2, 0]])


@pytest.mark.parametrize("key", [
    (0, slice(None), [0, 1]),          # int and array separated by a slice (M6 decision)
    ([0, 1], slice(None), 0),
    ([0, 1], slice(None), [0, 1]),
    ([0, 1], None, [0, 1]),
    ([0, 1], Ellipsis, [0, 1]),
], ids=repr)
def test_separated_advanced_indices_are_refused(key):
    with pytest.raises(IndexError):
        sdnp.asarray(N)[key]


@pytest.mark.parametrize("key", [True, False, (0, True), [1.0, 2.0], sdnp.asarray([0.5]), [0, 3], ([0], [9]),
                                 [-4], N[0] > 0, np.ones((3, 5), bool), "a"], ids=repr)
def test_index_errors(key):
    with pytest.raises(IndexError):
        sdnp.asarray(N)[key]


# ------------------------------------------------------------------ scatter

def test_setitem_matches_numpy():
    cases = [
        ([0, 2], 7), ((slice(None), [1, 3]), np.arange(30).reshape(3, 2, 5) * -1),
        (N > 40, -1), ((N[:, :, 0] > 20,), np.arange(5)), ([[0], [2]], 0), ((1, [0, 3]), [[9] * 5, [8] * 5]),
    ]
    for key, value in cases:
        want = N.copy()
        want[key] = value
        x = sdnp.asarray(N.copy())
        x[key] = sdnp.asarray(np.asarray(value)) if not np.isscalar(value) else value
        assert_same(x, want)


def test_repeated_indices_last_write_wins():
    x = sdnp.zeros(4)
    x[[1, 1, 3, 1]] = sdnp.asarray([1.0, 2.0, 3.0, 4.0])
    assert x.tolist() == [0.0, 4.0, 0.0, 3.0]


def test_scatter_rules():
    x = sdnp.zeros(4, dtype=sdnp.int64)
    with pytest.raises(TypeError):
        x[[0, 1]] = 1.5  # S6
    with pytest.raises(ValueError):
        x[[0, 1]] = sdnp.asarray([1, 2, 3])  # strict broadcasting
    with pytest.raises(ValueError):
        x[[0, 1]] = sdnp.asarray([[1, 2]])
    x[x == 0] = 5
    assert x.tolist() == [5, 5, 5, 5]
    ro = np.zeros(3)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        sdnp.asarray(ro)[[0]] = 1.0


def test_scatter_from_overlapping_value():
    n = np.arange(6.0)
    x = sdnp.asarray(n.copy())
    x[[0, 1, 2, 3, 4, 5]] = x[::-1]
    n[[0, 1, 2, 3, 4, 5]] = n[::-1].copy()
    assert_same(x, n)


@given(st.data())
def test_random_integer_indices(data):
    shape = data.draw(st.lists(st.integers(1, 4), min_size=1, max_size=3))
    n = np.arange(int(np.prod(shape))).reshape(shape)
    lead = data.draw(st.integers(0, len(shape) - 1))  # basic slices before the advanced run
    run = data.draw(st.integers(1, len(shape) - lead))
    ishape = data.draw(st.lists(st.integers(0, 3), min_size=1, max_size=2))
    key = [slice(None)] * lead
    for axis in range(lead, lead + run):
        key.append(np.array(data.draw(st.lists(st.integers(-shape[axis], shape[axis] - 1),
                                               min_size=int(np.prod(ishape)), max_size=int(np.prod(ishape)))),
                            dtype=np.int64).reshape(ishape))
    key = tuple(key)
    assert_same(sdnp.asarray(n)[key], n[key])
    value = data.draw(st.integers(-9, 9))
    want = n.copy()
    want[key] = value
    x = sdnp.asarray(n.copy())
    x[key] = value
    assert_same(x, want)


# ----------------------------------------- the fused one-index path (M9+)

def test_failed_assignment_writes_nothing():
    # Every index is checked before the first write (numpy behaves the same).
    for key in ([0, 1, 10, 2], (np.array([0, 1]), np.array([0, 99]))):
        n = np.arange(12.0).reshape(3, 4) if isinstance(key, tuple) else np.arange(5.0)
        x = sdnp.asarray(n.copy())
        with pytest.raises(IndexError):
            x[key] = -1.0
        assert_same(x, n)
    big = sdnp.zeros(100_000)
    idx = np.arange(100_000)
    idx[-1] = 100_000
    with pytest.raises(IndexError, match="index 100000 is out of bounds"):
        big[idx] = 1.0
    assert sdnp.sum(big) == 0.0


def test_first_bad_index_in_c_order_is_reported():
    x = sdnp.arange(10.0)
    with pytest.raises(IndexError, match="index -11 "):
        x[np.array([[0, 30], [-11, 2]]).T]  # C order of the transposed view: 0, -11, 30, 2


@pytest.mark.parametrize("idx", [
    np.array([3, -1, 0, 3, -10]), np.array([[1, 2, 3], [-4, -5, -6]]).T, np.arange(10)[::-3],
    np.zeros((0,), np.int64), np.array([[[7]]]),
])
def test_one_index_layouts(idx):
    n = np.arange(10.0) * 1.5
    assert_same(sdnp.asarray(n)[sdnp.asarray(idx)], n[idx])
    assert_same(sdnp.asarray(n[::-2])[sdnp.asarray(idx % 5)], n[::-2][idx % 5])
    want = n.copy()
    want[idx] = np.arange(idx.size).reshape(idx.shape) * 10.0
    x = sdnp.asarray(n.copy())
    x[sdnp.asarray(idx)] = sdnp.asarray(np.arange(idx.size).reshape(idx.shape) * 10.0)
    assert_same(x, want)  # repeated indices: the last write wins, in C order


def test_one_index_mixed_dtype_value_and_threads():
    n = np.zeros(10)
    x = sdnp.asarray(n.copy())
    x[sdnp.asarray([1, 2, 1])] = sdnp.asarray([5, 6, 7])  # int64 value into float64: the general path
    n[[1, 2, 1]] = [5, 6, 7]
    assert_same(x, n)
    src = np.random.default_rng(0).standard_normal(300_000)
    idx = np.random.default_rng(1).integers(-300_000, 300_000, 500_000)
    before = sdnp.get_num_threads()
    try:
        results = []
        for t in (1, 4, 10):
            sdnp.set_num_threads(t)
            results.append(np.asarray(sdnp.asarray(src)[sdnp.asarray(idx)]).tobytes())
        assert results[0] == results[1] == results[2] == src[idx].tobytes()
    finally:
        sdnp.set_num_threads(before)
