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


# M10+: one index array with whole blocks (one loop over the result shape),
# and boolean masks in fixed chunks. Results must equal numpy exactly.

@pytest.mark.parametrize("layout", ["C", "F", "reversed", "strided"])
def test_take_along_one_axis_layouts(layout):
    base = np.arange(6 * 7 * 8, dtype=np.float64).reshape(6, 7, 8) * 0.5
    a = {"C": base, "F": np.asfortranarray(base), "reversed": base[::-1, :, ::-1],
         "strided": np.arange(12 * 7 * 16.0).reshape(12, 7, 16)[::2, :, ::2]}[layout]
    x = sdnp.asarray(a)
    for key in [(slice(None), [3, -1, 0, 3]), (Ellipsis, np.array([[1, -8], [7, 0]])), ([5, -6, 2],),
                (slice(None), slice(None), [2]), (slice(None), [6] * 3, slice(1, 5))]:
        sd_key = tuple(sdnp.asarray(np.asarray(k)) if isinstance(k, (list, np.ndarray)) else k for k in key)
        assert_same(x[sd_key], a[key])
    with pytest.raises(IndexError, match="index 7 is out of bounds for axis 1"):
        x[:, sdnp.asarray([0, 7])]
    for dt in (np.int64, np.complex128, np.bool_):
        b = (base * 3).astype(dt)
        assert_same(sdnp.asarray(b)[:, sdnp.asarray([2, 0, -1])], b[:, [2, 0, -1]])


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 255, 256, 257, 16_383, 16_384, 16_385, 100_003])
@pytest.mark.parametrize("density", [0.0, 0.01, 0.5, 1.0])
def test_mask_chunks_match_numpy(n, density):
    rng = np.random.default_rng(n)
    mask = rng.random(n) < density
    for dt in (np.float64, np.int64, np.bool_, np.complex128):
        a = (rng.standard_normal(n) * 100).astype(dt)
        x = sdnp.asarray(a)
        assert_same(x[sdnp.asarray(mask)], a[mask])
        assert_same(sdnp.asarray(a[::-1])[sdnp.asarray(mask)], a[::-1][mask])
        assert_same(x[sdnp.asarray(mask[::-1])], a[mask[::-1]])  # a strided (reversed) mask
        want = a.copy()
        fill = (a[:1] if n else np.zeros(1, dt))[0]
        want[mask] = fill
        y = sdnp.asarray(a.copy())
        y[sdnp.asarray(mask)] = fill.item()
        assert_same(y, want)
        k = int(mask.sum())
        want = a.copy()
        want[mask] = a[:k][::-1]
        y = sdnp.asarray(a.copy())
        y[sdnp.asarray(mask)] = sdnp.asarray(a[:k][::-1])
        assert_same(y, want)


def test_mask_chunks_multi_axis_and_blocks():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((300, 200))
    m2 = rng.random((300, 200)) < 0.3
    assert_same(sdnp.asarray(a)[sdnp.asarray(m2)], a[m2])
    assert_same(sdnp.asarray(a.T)[sdnp.asarray(m2.T)], a.T[m2.T])
    rows = rng.random(300) < 0.5
    assert_same(sdnp.asarray(a)[sdnp.asarray(rows)], a[rows])  # blocks of 200 elements
    assert_same(sdnp.asarray(a)[:, sdnp.asarray(rows[:200])], a[:, rows[:200]])
    want = a.copy()
    want[m2] = -1.0
    x = sdnp.asarray(a.copy())
    x[sdnp.asarray(m2)] = -1.0
    assert_same(x, want)
    nan = np.array([np.nan, -0.0, 1.0] * 7000)
    y = sdnp.asarray(nan.copy())
    y[sdnp.asarray(nan == 1.0)] = 2.0  # untouched elements keep their bits (nan, -0.0)
    want = nan.copy()
    want[nan == 1.0] = 2.0
    assert np.asarray(y).tobytes() == want.tobytes()


def test_leading_int_fast_path_keeps_the_rules():
    a = np.arange(24.0).reshape(2, 3, 4)
    x = sdnp.asarray(a)
    assert x[1, 2, 3] == 23.0 and x[-1, -3, -4] == 12.0 and type(x[0, 0, 0]) is float
    assert_same(x[1], a[1])
    assert_same(x[-1, 1], a[-1, 1])
    assert_same(x[(0,)], a[0])
    v = x[1]
    assert v.base is not None
    v[0, 0] = -5.0
    assert x[1, 0, 0] == -5.0  # a view, not a copy
    for bad, exc in [((2, 0, 0), IndexError), ((0, -4), IndexError), ((0, 0, 0, 0), IndexError),
                     ((2**70,), IndexError), ((0, 0, 1.0), IndexError)]:
        with pytest.raises(exc):
            x[bad]
    b = sdnp.asarray(np.arange(3))
    assert_same(b[sdnp.asarray([True, False, True])], np.array([0, 2]))
    assert b[-1] == 2 and type(b[0]) is int
    with pytest.raises(IndexError, match="index 3 is out of bounds for axis 0 with size 3"):
        b[3]


def test_advanced_results_follow_the_input_layout():
    a = np.arange(30.0).reshape(5, 6)
    f = np.asfortranarray(a)
    rows = sdnp.asarray([4, 0, 2, 2])
    c_res, f_res = sdnp.asarray(a)[:, rows], sdnp.asarray(f)[:, rows]
    assert_same(c_res, a[:, [4, 0, 2, 2]])
    assert_same(f_res, a[:, [4, 0, 2, 2]])
    assert np.asarray(c_res).flags.c_contiguous and np.asarray(f_res).flags.f_contiguous
    assert np.asarray(sdnp.asarray(f)[rows % 5]).flags.f_contiguous
    # two index arrays and masks on an F-ordered input: same values, F layout
    two = sdnp.asarray(f)[sdnp.asarray([[0, 1], [4, 3]]), sdnp.asarray([5, 0])]
    assert_same(two, a[[[0, 1], [4, 3]], [5, 0]])
    t = np.asfortranarray(np.arange(60.0).reshape(3, 4, 5))
    m = np.array([True, False, True, True])
    got = sdnp.asarray(t)[:, sdnp.asarray(m)]
    assert_same(got, t[:, m])
    assert np.asarray(got).flags.f_contiguous
    g3 = sdnp.asarray(t)[:, sdnp.asarray([[3, 0], [1, 1]])]
    assert_same(g3, t[:, [[3, 0], [1, 1]]])
    # the B axes sit where axis 1 was (between axes 0 and 2 in memory), C order among themselves
    assert np.asarray(g3).strides == (8, 48, 24, 96)


# ------------------------------------------------- review fixes (2026-11-10)

def test_scatter_with_indices_inside_the_target():
    # The indices are read as they were before the assignment.
    for n0, key_of in [(np.array([1, 2, 3, 0]), lambda a: a),
                       (np.array([1, 0]), lambda a: a),
                       (np.arange(6)[::-1].copy(), lambda a: a[::2])]:
        for value in (2, 10**12):
            n, x = n0.copy(), sdnp.asarray(n0.copy())
            n[key_of(n).copy()] = value
            x[key_of(x)] = value
            assert_same(x, n)
    # two index arrays, one of them a view of x
    n = np.array([[0, 1], [1, 0]])
    x = sdnp.asarray(n.copy())
    want = n.copy()
    want[n[0].copy(), n[1].copy()] = 7
    x[x[0], x[1]] = 7
    assert_same(x, want)


def test_mask_assignment_with_mask_inside_the_target():
    n = np.random.default_rng(1).random(40_000) < 0.5
    x = sdnp.asarray(n.copy())
    x[x[::-1]] = False  # the mask is x reversed: writes land where it is still to be read
    want = n.copy()
    want[n[::-1].copy()] = False
    assert_same(x, want)


def test_dense_mask_assignment_ending_in_false():
    # Dense 256-element pieces whose last trues come before falses: the value
    # array is exactly as long as the selection.
    n = np.arange(70_000.0)
    m = np.ones(70_000, dtype=bool)
    m[-100:] = False
    m[1000:1100] = False
    v = -np.arange(float(m.sum()))
    x = sdnp.asarray(n.copy())
    x[sdnp.asarray(m)] = sdnp.asarray(v)
    n[m] = v
    assert_same(x, n)


def test_index_into_an_empty_axis():
    e = sdnp.zeros(0)
    with pytest.raises(IndexError, match="index 0 is out of bounds for axis 0 with size 0"):
        e[sdnp.asarray([0])]
    with pytest.raises(IndexError):
        e[sdnp.asarray([-1, 0])]
    assert_same(e[sdnp.asarray(np.zeros(0, dtype=np.int64))], np.zeros(0))
