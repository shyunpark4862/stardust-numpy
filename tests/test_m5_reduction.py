"""M5: reductions against numpy.

Integer, bool and min/max/arg* results must be exact. Floating sums are
compared with a tolerance: the summation order is deliberately not numpy's
(DESIGN.md §2.4), only deterministic.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust_numpy as sdnp
from conftest import ALL_DTYPES, NP_DTYPE, assert_same

INT_MAX, INT_MIN = 2**63 - 1, -(2**63)
AXES = [None, 0, 1, -1, (0, 2), (2, 0), (), (0, 1, 2)]


def data(dtype, shape=(3, 4, 5), seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(-3, 4, shape)
    if dtype is sdnp.bool:
        return n > 0
    if dtype is sdnp.int64:
        return n.astype(np.int64)
    if dtype is sdnp.float64:
        return rng.random(shape) + 0.5
    return (rng.random(shape) + 0.5) + 1j * (rng.random(shape) - 0.5)


def layouts(n):
    """The same values in several memory layouts."""
    yield n
    yield np.asfortranarray(n)
    t = np.ascontiguousarray(n.transpose(2, 0, 1)).transpose(1, 2, 0)  # permuted memory order
    yield t
    big = np.zeros(tuple(2 * s for s in n.shape), n.dtype)
    big[::2, ::2, ::2] = n
    yield big[::2, ::2, ::2]  # strided view


def compare(got, want, exact):
    want = np.asarray(want)
    if want.ndim == 0:
        assert not isinstance(got, sdnp.ndarray), "a full reduction must give a Python scalar (Z1)"
        got_np = np.asarray(got, dtype=want.dtype)
        assert type(got) is {"b": bool, "i": int, "f": float, "c": complex}[want.dtype.kind]
    else:
        assert isinstance(got, sdnp.ndarray) and got.shape == want.shape
        assert NP_DTYPE[got.dtype] == want.dtype, (got.dtype, want.dtype)
        got_np = np.asarray(got)
    if exact:
        assert_same(sdnp.asarray(got_np.reshape(-1)), want.reshape(-1))
    else:
        np.testing.assert_allclose(got_np, want, rtol=1e-13, atol=0)


@pytest.mark.parametrize("axis", AXES, ids=str)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("name", ["sum", "prod", "any", "all", "mean", "min", "max"])
def test_reductions_match_numpy(name, axis, keepdims, dtype):
    n = data(dtype)
    fn = getattr(sdnp, name)
    if name in ("min", "max") and dtype is sdnp.complex128:
        with pytest.raises(TypeError):
            fn(sdnp.asarray(n), axis=axis)
        return
    want = getattr(np, name)(n, axis=axis, keepdims=keepdims)
    exact = name in ("any", "all", "min", "max") or dtype in (sdnp.bool, sdnp.int64) and name != "mean"
    for layout in layouts(n):
        compare(fn(sdnp.asarray(layout), axis=axis, keepdims=keepdims), want, exact)


@pytest.mark.parametrize("axis", AXES, ids=str)
@pytest.mark.parametrize("correction", [0, 1, 0.5])
@pytest.mark.parametrize("name", ["var", "std"])
def test_var_std_match_numpy(name, axis, correction, dtype):
    n = data(dtype)
    if axis == () and correction >= 1:
        want = np.full(n.shape, np.nan)  # one element per output: N - correction <= 0 -> nan
    else:
        want = getattr(np, name)(n, axis=axis, ddof=correction)
    for layout in layouts(n):
        got = getattr(sdnp, name)(sdnp.asarray(layout), axis=axis, correction=correction)
        compare(got, np.asarray(want, dtype=np.float64), exact=False)


def test_methods_are_the_functions():
    n = data(sdnp.float64)
    x = sdnp.asarray(n)
    for name in ("sum", "prod", "min", "max", "mean", "var", "std", "argmin", "argmax", "any", "all"):
        assert_same(getattr(x, name)(axis=1), np.asarray(getattr(sdnp, name)(x, axis=1)))
    assert x.sum() == sdnp.sum(x)
    with pytest.raises(TypeError):
        x.sum(1)  # axis is keyword-only


# ----------------------------------------------------------- result dtypes

def test_result_dtypes():
    b, i, f, c = (sdnp.asarray(data(d)) for d in ALL_DTYPES)
    assert sdnp.sum(b, axis=0).dtype is sdnp.int64  # C13
    assert sdnp.prod(b, axis=0).dtype is sdnp.int64
    assert sdnp.sum(i, axis=0).dtype is sdnp.int64
    assert sdnp.max(b, axis=0).dtype is sdnp.bool
    for x in (b, i):
        assert sdnp.mean(x, axis=0).dtype is sdnp.float64
        assert sdnp.var(x, axis=0).dtype is sdnp.float64
    assert sdnp.mean(c, axis=0).dtype is sdnp.complex128
    assert sdnp.var(c, axis=0).dtype is sdnp.float64 and sdnp.std(c, axis=0).dtype is sdnp.float64
    assert sdnp.any(f, axis=0).dtype is sdnp.bool and sdnp.argmax(f, axis=0).dtype is sdnp.int64


def test_dtype_argument():
    f = sdnp.asarray([1.9, 2.9, -0.5])
    assert sdnp.sum(f, dtype=sdnp.int64) == 1 + 2 + 0  # explicit: each element truncated first
    assert sdnp.sum(sdnp.asarray([1, 2]), dtype=sdnp.float64) == 3.0
    assert type(sdnp.prod(sdnp.asarray([2, 3]), dtype=sdnp.complex128)) is complex
    with pytest.raises(TypeError):
        sdnp.sum(f, dtype=sdnp.bool)  # no bool accumulation (M5 decision)
    with pytest.raises(TypeError):
        sdnp.sum(sdnp.asarray([1j]), dtype=sdnp.float64)  # S4


def test_integer_sum_wraps():
    assert sdnp.sum(sdnp.asarray([INT_MAX, 1])) == INT_MIN
    assert sdnp.prod(sdnp.asarray([2**32, 2**32])) == 0


# ------------------------------------------------------------- empty arrays

def test_empty_rules():
    e = sdnp.zeros((0,))
    assert sdnp.sum(e) == 0 and sdnp.prod(e) == 1
    assert sdnp.any(e) is False and sdnp.all(e) is True
    for fn in (sdnp.mean, sdnp.var, sdnp.std):
        assert math.isnan(fn(e))
    for fn in (sdnp.min, sdnp.max, sdnp.argmin, sdnp.argmax):
        with pytest.raises(ValueError):
            fn(e)
    z = sdnp.zeros((3, 0))
    assert sdnp.sum(z, axis=1).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        sdnp.max(z, axis=1)  # every output would fold nothing
    assert sdnp.max(z, axis=0).shape == (0,)  # nothing to compute: fine
    with pytest.raises(ValueError):
        sdnp.argmax(z, axis=1)
    assert sdnp.argmax(z, axis=0).shape == (0,)


def test_sum_signed_zero_like_numpy():
    z = sdnp.asarray([-0.0, -0.0])
    assert math.copysign(1, sdnp.sum(z)) == 1  # starts from +0, as numpy
    assert [math.copysign(1, v) for v in sdnp.cumulative_sum(z).tolist()] == [-1, -1]


# ---------------------------------------------------------------- NaN rules

def test_nan_rules():
    x = sdnp.asarray([1.0, math.nan, 3.0, math.nan])
    assert math.isnan(sdnp.max(x)) and math.isnan(sdnp.min(x))
    assert sdnp.argmax(x) == 1 and sdnp.argmin(x) == 1  # the first NaN
    assert math.isnan(sdnp.sum(x)) and math.isnan(sdnp.mean(x))
    assert sdnp.any(sdnp.asarray([0.0, math.nan])) is True  # nan != 0
    m = sdnp.asarray([[1.0, math.nan], [2.0, 0.5]])
    assert sdnp.argmax(m, axis=1).tolist() == [1, 0]


def test_ties_give_the_first_index():
    x = sdnp.asarray([3, 1, 3, 1])
    assert sdnp.argmax(x) == 0 and sdnp.argmin(x) == 1
    t = sdnp.asarray(np.array([[1, 5], [5, 1]]).T)  # transposed: flat index is C order of the view
    assert sdnp.argmax(t) == int(np.argmax(np.asarray(t)))


@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmin_argmax_match_numpy(axis, keepdims, dtype):
    if dtype is sdnp.complex128:
        with pytest.raises(TypeError):
            sdnp.argmax(sdnp.asarray(data(dtype)))
        return
    n = data(dtype)  # small integers: many ties
    for layout in layouts(n):
        for name in ("argmin", "argmax"):
            got = getattr(sdnp, name)(sdnp.asarray(layout), axis=axis, keepdims=keepdims)
            compare(got, getattr(np, name)(n, axis=axis, keepdims=keepdims), exact=True)


def test_argmax_axis_must_be_int():
    with pytest.raises(TypeError):
        sdnp.argmax(sdnp.zeros((2, 2)), axis=(0, 1))  # C5


# ------------------------------------------------------------- var details

def test_var_degrees_of_freedom():
    one = sdnp.asarray([5.0])
    assert sdnp.var(one) == 0.0
    assert math.isnan(sdnp.var(one, correction=1))  # N - correction <= 0 -> nan (M5 decision)
    assert math.isnan(sdnp.std(sdnp.asarray([1.0, 2.0]), correction=2))
    assert sdnp.var(sdnp.asarray([1.0, 2.0, 3.0, 4.0]), correction=1) == pytest.approx(5 / 3)
    with pytest.raises(TypeError):
        sdnp.var(one, correction=True)


def test_var_complex_is_mean_squared_modulus():
    z = np.array([1 + 1j, -1 - 1j, 2j])
    assert sdnp.var(sdnp.asarray(z)) == pytest.approx(np.var(z))


# --------------------------------------------------------------- cumulative

@pytest.mark.parametrize("name,np_name", [("cumulative_sum", "cumulative_sum"), ("cumulative_prod", "cumulative_prod")])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("include_initial", [False, True])
def test_cumulative_match_numpy(name, np_name, axis, include_initial, dtype):
    n = data(dtype)
    want = getattr(np, np_name)(n, axis=axis, include_initial=include_initial)
    if dtype is sdnp.bool:
        want = want.astype(np.int64)
    for layout in layouts(n):
        got = getattr(sdnp, name)(sdnp.asarray(layout), axis=axis, include_initial=include_initial)
        compare(got, want, exact=dtype in (sdnp.bool, sdnp.int64))


def test_cumulative_axis_rules():
    v = sdnp.asarray([1, 2, 3])
    assert sdnp.cumulative_sum(v).tolist() == [1, 3, 6]  # 1-D: axis may be omitted
    with pytest.raises(ValueError):
        sdnp.cumulative_sum(sdnp.zeros((2, 2)))  # C4: ndim > 1 needs axis
    with pytest.raises(TypeError):
        sdnp.cumulative_sum(v, axis=(0,))
    assert sdnp.cumulative_sum(sdnp.zeros(0), include_initial=True).tolist() == [0.0]
    assert sdnp.cumulative_prod(sdnp.asarray([True, True])).dtype is sdnp.int64
    with pytest.raises(TypeError):
        sdnp.cumulative_sum(v, dtype=sdnp.bool)


# --------------------------------------------------------- axes and inputs

def test_axis_errors_and_inputs():
    x = sdnp.zeros((2, 3))
    with pytest.raises(ValueError):
        sdnp.sum(x, axis=2)
    with pytest.raises(ValueError):
        sdnp.sum(x, axis=(0, 0))
    with pytest.raises(TypeError):
        sdnp.sum([1, 2, 3])  # arrays only
    with pytest.raises(TypeError):
        sdnp.sum(x, 0)


def test_full_reduction_types():
    assert type(sdnp.sum(sdnp.asarray([1, 2]))) is int
    assert type(sdnp.max(sdnp.asarray([True]))) is bool
    assert type(sdnp.mean(sdnp.asarray([1, 2]))) is float
    assert type(sdnp.argmax(sdnp.asarray([1, 2]))) is int
    k = sdnp.sum(sdnp.ones((2, 3)), keepdims=True)
    assert isinstance(k, sdnp.ndarray) and k.shape == (1, 1)


def test_deterministic():
    n = np.random.default_rng(3).standard_normal((257, 129))
    x = sdnp.asarray(n)
    for fn in (sdnp.sum, sdnp.mean, sdnp.var):
        first = np.asarray(fn(x, axis=0))
        for _ in range(3):
            assert np.array_equal(np.asarray(fn(x, axis=0)).view(np.int64), first.view(np.int64))


@given(st.data())
def test_random_shapes_and_axes(data_):
    shape = data_.draw(hnp.array_shapes(min_dims=1, max_dims=4, min_side=0, max_side=4))
    n = data_.draw(hnp.arrays(np.int64, shape, elements=st.integers(-100, 100)))
    axis = data_.draw(st.one_of(st.none(), st.integers(-len(shape), len(shape) - 1),
                                st.lists(st.integers(0, len(shape) - 1), unique=True).map(tuple)))
    keepdims = data_.draw(st.booleans())
    x = sdnp.asarray(n)
    for name in ("sum", "prod", "any", "all"):
        compare(getattr(sdnp, name)(x, axis=axis, keepdims=keepdims),
                getattr(np, name)(n, axis=axis, keepdims=keepdims), exact=True)
    ax = axis if axis is None or isinstance(axis, tuple) else (axis,)
    reduced = range(len(shape)) if ax is None else [a % len(shape) for a in ax]
    if all(shape[a] > 0 for a in reduced):
        compare(sdnp.max(x, axis=axis, keepdims=keepdims), np.max(n, axis=axis, keepdims=keepdims), exact=True)


# ------------------------------------------------- M7 kernels: vector paths

@given(st.lists(st.sampled_from([0.0, -0.0, 1.0, 2.0, -3.0, math.inf, -math.inf, math.nan]), min_size=1, max_size=40))
def test_vector_folds_match_numpy(values):
    # Long enough lists reach the NEON paths (pairs of lanes, 8-wide folds);
    # ties, NaNs and signed zeros land in every lane position.
    n = np.array(values)
    x = sdnp.asarray(n)
    for name in ("argmax", "argmin"):
        assert getattr(sdnp, name)(x) == getattr(np, name)(n)
    for name in ("max", "min"):
        got, want = getattr(sdnp, name)(x), getattr(np, name)(n)
        assert (math.isnan(got) and math.isnan(want)) or (got == want and math.copysign(1, got) == math.copysign(1, want))


def test_pairwise_sum_accuracy():
    x = np.full(10**6, 0.1) + np.random.default_rng(7).random(10**6) * 1e-3
    exact = math.fsum(x.tolist())
    assert abs(sdnp.sum(sdnp.asarray(x)) - exact) / exact < 1e-15  # a running sum would be ~1e-14
    c = x + 1j * x[::-1]
    got = sdnp.sum(sdnp.asarray(c))
    assert abs(got.real - exact) / exact < 1e-15 and abs(got.imag - exact) / exact < 1e-15


@pytest.mark.parametrize("where", [0, 1, 4095, 4096, 4097, 99_999])
def test_any_all_stop_early_but_stay_right(where):
    z = np.zeros(100_000, bool)
    z[where] = True
    o = ~z
    for arr in (z, z[::-1], z.reshape(400, 250), z.reshape(400, 250).T):
        assert sdnp.any(sdnp.asarray(arr)) is True
        assert sdnp.all(sdnp.asarray(~arr)) is False
        assert np.array_equal(np.asarray(sdnp.any(sdnp.asarray(z.reshape(400, 250)), axis=1)), z.reshape(400, 250).any(axis=1))
    assert sdnp.any(sdnp.asarray(np.zeros(5000, bool))) is False and sdnp.all(sdnp.asarray(o | True)) is True
    assert sdnp.any(sdnp.asarray(np.where(z, 2.5, 0.0))) is True  # non-bool input: through the cast


@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32, 33, 1000])
def test_int_min_max_vector_fold_finds_every_position(n):
    # The NEON fold keeps 16 lanes; the extreme may sit in any lane or the tail.
    for pos in sorted({0, min(1, n - 1), n // 2, n - 1, min(16, n - 1)}):
        x = np.arange(n, dtype=np.int64) * 7 - 3 * n
        x[pos] = INT_MAX
        assert sdnp.max(sdnp.asarray(x)) == INT_MAX
        x[pos] = INT_MIN
        assert sdnp.min(sdnp.asarray(x)) == INT_MIN
        assert sdnp.max(sdnp.asarray(x)) == x.max()
