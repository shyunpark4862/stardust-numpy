"""M4: every elementwise function on every dtype, against numpy.

The expected loop dtype comes from the spec below (which dtypes each function
has a loop for); numpy then computes the reference on inputs already
converted to that dtype. Where stardust deliberately differs from numpy (D1
integer division and power, C13 bool arithmetic) the reference is plain
Python instead.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust_numpy as sdnp
from conftest import ALL_DTYPES, NP_DTYPE, assert_same, assert_ulp

B, I, F, C = sdnp.bool, sdnp.int64, sdnp.float64, sdnp.complex128
ORDER = [B, I, F, C]
INT_MAX, INT_MIN = 2**63 - 1, -(2**63)

_CF = [0.0, -0.0, 1.5, -2.5, math.inf, -math.inf, math.nan, 0.7, 1e-8, 2.0]
VALUES = {
    B: np.array([True, False]),
    I: np.array([0, 1, -1, 2, -7, 3, 5, INT_MAX, INT_MIN, 12345678901, 63, 64, -64]),
    F: np.array([0.0, -0.0, 1.5, -2.5, 0.5, 3.0, math.inf, -math.inf, math.nan, 1e308, 5e-324, -1e-310,
                 2.0, 100.0, 0.7, -0.3, 710.0, 1e-8, -1.0, 1.0]),
    C: np.array([complex(r, i) for r in _CF for i in _CF]),
}

# Which dtypes each function has a loop for (the spec; mirrors scalar_ops.hpp).
ALL, NUM, INEXACT, REAL, INT_FLOAT, BOOL_INT = set(ORDER), {I, F, C}, {F, C}, {B, I, F}, {I, F}, {B, I}
UNARY = {
    "negative": NUM, "positive": NUM, "abs": NUM, "square": NUM, "sign": NUM,
    "sqrt": INEXACT, "exp": INEXACT, "expm1": INEXACT, "log": INEXACT, "log1p": INEXACT, "log2": INEXACT,
    "log10": INEXACT, "sin": INEXACT, "cos": INEXACT, "tan": INEXACT, "asin": INEXACT, "acos": INEXACT,
    "atan": INEXACT, "sinh": INEXACT, "cosh": INEXACT, "tanh": INEXACT, "asinh": INEXACT, "acosh": INEXACT,
    "atanh": INEXACT,
    "floor": REAL, "ceil": REAL, "trunc": REAL, "round": ALL,
    "isnan": ALL, "isinf": ALL, "isfinite": ALL, "conj": ALL,
    "logical_not": {B}, "bitwise_invert": BOOL_INT,
}
BINARY = {
    "add": NUM, "subtract": NUM, "multiply": NUM, "divide": INEXACT, "pow": NUM,
    "floor_divide": INT_FLOAT, "remainder": INT_FLOAT, "atan2": {F},
    "maximum": REAL, "minimum": REAL,
    "equal": ALL, "not_equal": ALL, "less": REAL, "less_equal": REAL, "greater": REAL, "greater_equal": REAL,
    "logical_and": {B}, "logical_or": {B}, "logical_xor": {B},
    "bitwise_and": BOOL_INT, "bitwise_or": BOOL_INT, "bitwise_xor": BOOL_INT,
    "bitwise_left_shift": {I}, "bitwise_right_shift": {I},
}
# float64 functions that M9 moves to vForce: accurate to 8 ULP by contract
# (DESIGN.md §2.6). Everything else must match numpy bit for bit.
VFORCE = {"exp", "log", "log2", "log10", "sin", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
          "asinh", "acosh", "atanh", "pow"}


def loop_dtype(loops, d):
    return next((k for k in ORDER[ORDER.index(d):] if k in loops), None)


def wrap(v):
    return (v + 2**63) % 2**64 - 2**63


def py_int_binary(name, a, b):
    """D1 and the shift rule, as plain Python on Python ints."""
    if name == "floor_divide":
        return 0 if b == 0 else wrap(a // b)
    if name == "remainder":
        return a if b == 0 else a % b
    if name == "pow":
        if b < 0:
            return 1 if a == 1 else ((-1 if b % 2 else 1) if a == -1 else 0)
        return wrap(pow(a, b, 2**64))
    if name == "bitwise_left_shift":
        return 0 if not 0 <= b < 64 else wrap(a << b)
    if name == "bitwise_right_shift":
        return (-1 if a < 0 else 0) if not 0 <= b < 64 else a >> b
    raise AssertionError(name)


def reference_unary(name, n, loop):
    x = n.astype(NP_DTYPE[loop])
    if name == "conj" and loop != C:
        return x
    if name in ("floor", "ceil", "trunc", "round") and loop in (B, I):
        return x  # already whole; numpy returns float16 for bool
    return getattr(np, name)(x)


def reference_binary(name, a, b, loop):
    a, b = np.broadcast_arrays(a.astype(NP_DTYPE[loop]), b.astype(NP_DTYPE[loop]))
    if loop is I and name in ("floor_divide", "remainder", "pow", "bitwise_left_shift", "bitwise_right_shift"):
        out = [py_int_binary(name, x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist())]
        return np.array(out, dtype=np.int64).reshape(a.shape)
    return getattr(np, name)(a, b)


def check(got, want, name, loop):
    if name in VFORCE and loop is F:
        assert_ulp(got, want, 8)
    else:
        assert_same(got, want)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_all_dtypes(name, dtype):
    loop = loop_dtype(UNARY[name], dtype)
    n = VALUES[dtype]
    fn = getattr(sdnp, name)
    if loop is None:
        with pytest.raises(TypeError):
            fn(sdnp.asarray(n))
        return
    with np.errstate(all="ignore"):
        want = reference_unary(name, n, loop)
    check(fn(sdnp.asarray(n)), want, name, loop)  # contiguous
    check(fn(sdnp.asarray(n)[::-1]), want[::-1], name, loop)  # strided


@pytest.mark.parametrize("name", sorted(BINARY))
@pytest.mark.parametrize("da,db", list(itertools.product(ORDER, ORDER)), ids=lambda d: str(d))
def test_binary_all_dtype_pairs(name, da, db):
    a, b = VALUES[da][:, None], VALUES[db][None, :]
    fn = getattr(sdnp, name)
    loop = loop_dtype(BINARY[name], max(da, db, key=ORDER.index))
    if name.startswith("bitwise_") and "shift" in name and B in (da, db):
        loop = None  # shifts take integers only
    if loop is None:
        with pytest.raises(TypeError):
            fn(sdnp.asarray(a), sdnp.asarray(b))
        return
    with np.errstate(all="ignore"):
        want = reference_binary(name, a, b, loop)
    check(fn(sdnp.asarray(a), sdnp.asarray(b)), want, name, loop)  # scalar-broadcast inner loops
    # The same pairs as contiguous arrays and as strided views: each layout has
    # its own kernel path (M7), and all must give the same bits.
    fa, fb = (np.ascontiguousarray(v) for v in np.broadcast_arrays(a, b))
    check(fn(sdnp.asarray(fa), sdnp.asarray(fb)), want, name, loop)
    check(fn(sdnp.asarray(fa.T), sdnp.asarray(fb.T)), want.T, name, loop)


_ELEMENTS = {"allow_nan": True, "allow_infinity": True}


@given(st.data())
def test_random_strided_broadcast_operands(data):
    # Arbitrary layouts (reversed, stepped, transposed), broadcasting and mixed
    # dtypes all go through the same iterator: compare a few ops with numpy.
    da, db = data.draw(st.sampled_from(ORDER)), data.draw(st.sampled_from(ORDER))
    shape = data.draw(hnp.array_shapes(min_dims=1, max_dims=3, max_side=4))
    a = data.draw(hnp.arrays(NP_DTYPE[da], shape, elements=None if da in (B, I) else _ELEMENTS))
    b = data.draw(hnp.arrays(NP_DTYPE[db], shape[data.draw(st.integers(0, len(shape) - 1)):]))
    a = a[tuple(data.draw(st.sampled_from([slice(None), slice(None, None, -1), slice(None, None, 2)]))
                for _ in shape)]
    b = b.T if data.draw(st.booleans()) and b.ndim > 1 and len(set(b.shape)) == 1 else b
    if not all(x == y or x == 1 or y == 1 for x, y in zip(a.shape[::-1], b.shape[::-1])):
        return
    name = data.draw(st.sampled_from(["add", "multiply", "subtract", "equal", "maximum", "divide"]))
    loop = loop_dtype(BINARY[name], max(da, db, key=ORDER.index))
    if loop is None:
        return
    with np.errstate(all="ignore"):
        want = reference_binary(name, a, b, loop)
    check(getattr(sdnp, name)(sdnp.asarray(a), sdnp.asarray(b)), want, name, loop)


def test_complex_arithmetic_on_random_values():
    # Complex multiply/divide/square/abs use numpy's exact formulas, including
    # where its compiler fused a multiply-add (see scalar_ops.hpp).
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000) * 10.0 ** rng.integers(-5, 5, 5000) + 1j * rng.standard_normal(5000)
    b = rng.standard_normal(5000) + 1j * rng.standard_normal(5000) * 10.0 ** rng.integers(-5, 5, 5000)
    sa, sb = sdnp.asarray(a), sdnp.asarray(b)
    with np.errstate(all="ignore"):  # numpy warns on overflow; stardust never does
        for name in ("add", "subtract", "multiply", "divide", "pow"):
            assert_same(getattr(sdnp, name)(sa, sb), getattr(np, name)(a, b))
        for name in ("square", "abs", "sqrt", "exp", "expm1", "log", "log1p", "log2", "log10", "sign", "tanh"):
            assert_same(getattr(sdnp, name)(sa), getattr(np, name)(a))
    small = sdnp.asarray(np.arange(-5, 6) + 0j)
    assert_same(sdnp.pow(sa[:11], small), np.power(a[:11], np.arange(-5, 6) + 0j))  # integer-exponent path


# ------------------------------------------------------------- D1 and C13

def test_integer_division_never_fails():
    a = sdnp.asarray([7, -7, 0, INT_MIN, INT_MIN])
    b = sdnp.asarray([0, 0, 0, -1, 1])
    assert (a // b).tolist() == [0, 0, 0, INT_MIN, INT_MIN]
    assert (a % b).tolist() == [7, -7, 0, 0, 0]
    # a == (a // b) * b + a % b, even for b == 0
    assert ((a // b) * b + a % b == a).tolist() == [True] * 5


def test_floor_semantics_match_python():
    for x, y in itertools.product([7, -7, 6, -6, 1], [3, -3, 2, -2, 7]):
        assert sdnp.floor_divide(x, y) == x // y
        assert sdnp.remainder(x, y) == x % y


def test_integer_power():
    base = sdnp.asarray([2, -1, -1, 1, 0, 3, 2])
    exp = sdnp.asarray([10, -3, -4, -9, -2, -1, 64])
    assert (base**exp).tolist() == [1024, -1, 1, 1, 0, 0, 0]  # 2**64 wraps to 0
    assert (sdnp.asarray([3]) ** 40).tolist() == [wrap(3**40)]


def test_bool_arithmetic_is_int64():
    t = sdnp.asarray([True, False])
    for got, want in [(t + t, [2, 0]), (t - t, [0, 0]), (t * t, [1, 0]), (-t, [-1, 0]), (+t, [1, 0]),
                      (abs(t), [1, 0]), (sdnp.square(t), [1, 0]), (sdnp.sign(t), [1, 0]), (t**t, [1, 1])]:
        assert got.dtype is sdnp.int64 and got.tolist() == want
    assert (t / t).dtype is sdnp.float64
    # ... but logic and bit operations keep bool (C13)
    for got in (t & t, t | t, t ^ t, ~t, sdnp.maximum(t, t), t == t):
        assert got.dtype is sdnp.bool
    assert (~t).tolist() == [False, True]


def test_shift_counts_are_unsigned():
    x = sdnp.asarray([1, -8, 5])
    assert (x << sdnp.asarray([63, 1, 64])).tolist() == [INT_MIN, -16, 0]
    assert (x >> sdnp.asarray([-1, 70, 1])).tolist() == [0, -1, 2]


@pytest.mark.parametrize("fn", [sdnp.logical_and, sdnp.logical_or, sdnp.logical_xor])
def test_logical_takes_bool_only(fn):
    t = sdnp.asarray([True, False])
    assert fn(t, True).dtype is sdnp.bool
    with pytest.raises(TypeError):
        fn(sdnp.asarray([1, 0]), t)
    with pytest.raises(TypeError):
        fn(t, 1)


def test_complex_has_no_order():
    c = sdnp.asarray([1j])
    for fn in (sdnp.less, sdnp.maximum, sdnp.minimum, sdnp.floor_divide, sdnp.remainder, sdnp.atan2):
        with pytest.raises(TypeError):
            fn(c, c)
    with pytest.raises(TypeError):
        c < c
    with pytest.raises(TypeError):
        sdnp.floor(c)
    assert (c == c).tolist() == [True]


def test_maximum_minimum_signed_zero_and_nan():
    a = sdnp.asarray([-0.0, 0.0, math.nan, 1.0])
    b = sdnp.asarray([0.0, -0.0, 1.0, math.nan])
    assert_same(sdnp.maximum(a, b), np.maximum(np.asarray(a), np.asarray(b)))
    assert_same(sdnp.minimum(a, b), np.minimum(np.asarray(a), np.asarray(b)))
    assert [math.copysign(1, v) for v in sdnp.maximum(a, b).tolist()[:2]] == [1, 1]
    assert [math.copysign(1, v) for v in sdnp.minimum(a, b).tolist()[:2]] == [-1, -1]


# ----------------------------------------------------------- Python scalars

def test_python_scalar_promotion():
    i = sdnp.asarray([1, 2])
    assert (i + 1).dtype is sdnp.int64
    assert (i + 1.5).dtype is sdnp.float64 and (i + 1.5).tolist() == [2.5, 3.5]
    assert (i * 1j).dtype is sdnp.complex128
    assert (sdnp.asarray([True]) + True).tolist() == [2]
    assert (sdnp.asarray([1.0]) + 2**70).tolist() == [float(2**70)]  # Z4: float target is fine
    with pytest.raises(OverflowError):
        i + 2**70  # Z4: must become int64
    with pytest.raises(OverflowError):
        i < 2**70


def test_scalars_only_give_python_scalars():
    assert sdnp.add(1, 2) == 3 and type(sdnp.add(1, 2)) is int
    assert sdnp.add(2**62, 2**62) == INT_MIN  # int64 semantics, wraps (Z1)
    assert sdnp.add(True, True) == 2
    assert sdnp.divide(1, 2) == 0.5
    assert math.isnan(sdnp.sqrt(-1.0))
    assert sdnp.sqrt(-1 + 0j) == 1j
    assert sdnp.less(1, 2) is True
    assert sdnp.floor_divide(7, 0) == 0


def test_operands_must_be_arrays_or_python_scalars():
    x = sdnp.asarray([1, 2])
    for bad in ([1, 2], (1, 2), np.array([1, 2]), "1", None):
        with pytest.raises(TypeError):
            sdnp.add(x, bad)
    with pytest.raises(TypeError):
        x + [1, 2]
    with pytest.raises(TypeError):
        np.array([1, 2]) + x  # numpy steps aside (__array_ufunc__ = None)
    with pytest.raises(TypeError):
        x + np.array([1, 2])
    assert (x == None) is False  # noqa: E711 -- NotImplemented, then identity


# ------------------------------------------------------------------- out=

def test_out_basic_and_upcast():
    a, b = sdnp.asarray([1, 2, 3]), sdnp.asarray([10, 20, 30])
    out = sdnp.zeros(3, dtype=sdnp.int64)
    assert sdnp.add(a, b, out=out) is out and out.tolist() == [11, 22, 33]
    wide = sdnp.zeros(3, dtype=sdnp.complex128)
    sdnp.multiply(a, b, out=wide)  # int64 result widened into complex128
    assert wide.tolist() == [10, 40, 90]
    big = sdnp.zeros((2, 3))
    sdnp.add(a, 0.5, out=big)  # broadcast into a larger out
    assert big.tolist() == [[1.5, 2.5, 3.5]] * 2


def test_out_rules():
    a = sdnp.asarray([1.0, 2.0])
    with pytest.raises(TypeError):
        sdnp.add(a, a, out=sdnp.zeros(2, dtype=sdnp.int64))  # S6
    assert sdnp.less(a, a, out=sdnp.ones(2)).tolist() == [0.0, 0.0]  # bool -> float64 is upward: fine
    ro = np.zeros(2)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        sdnp.add(a, a, out=sdnp.asarray(ro))
    with pytest.raises(ValueError):
        sdnp.add(sdnp.zeros((2, 3)), 1.0, out=sdnp.zeros(3))  # result does not fit out
    with pytest.raises(TypeError):
        sdnp.add(a, a, sdnp.zeros(2))  # out is keyword-only


def test_out_overlapping_inputs():
    n = np.arange(10.0)
    a = sdnp.asarray(n.copy())
    sdnp.add(a[1:], a[:-1], out=a[1:])
    n[1:] = n[1:] + n[:-1]
    assert_same(a, n)
    b = sdnp.arange(6.0)
    sdnp.multiply(b, b[::-1], out=b)
    assert b.tolist() == [0.0, 4.0, 6.0, 6.0, 4.0, 0.0]


# ------------------------------------------------------------- operators

def test_operators_match_functions():
    x = sdnp.asarray([[5, -3], [7, 2]])
    y = sdnp.asarray([2, 3])
    pairs = [
        (x + y, sdnp.add), (x - y, sdnp.subtract), (x * y, sdnp.multiply), (x / y, sdnp.divide),
        (x // y, sdnp.floor_divide), (x % y, sdnp.remainder), (x**y, sdnp.pow), (x & y, sdnp.bitwise_and),
        (x | y, sdnp.bitwise_or), (x ^ y, sdnp.bitwise_xor), (x << y, sdnp.bitwise_left_shift),
        (x >> y, sdnp.bitwise_right_shift), (x == y, sdnp.equal), (x != y, sdnp.not_equal), (x < y, sdnp.less),
        (x <= y, sdnp.less_equal), (x > y, sdnp.greater), (x >= y, sdnp.greater_equal),
    ]
    for got, fn in pairs:
        assert_same(got, np.asarray(fn(x, y)))
    for got, fn in [(-x, sdnp.negative), (+x, sdnp.positive), (abs(x), sdnp.abs), (~x, sdnp.bitwise_invert)]:
        assert_same(got, np.asarray(fn(x)))


def test_reflected_operators():
    x = sdnp.asarray([1, 2, 4])
    assert (10 - x).tolist() == [9, 8, 6]
    assert (1 / x).tolist() == [1.0, 0.5, 0.25]
    assert (2**x).tolist() == [2, 4, 16]
    assert (7 // x).tolist() == [7, 3, 1]
    assert (1 << x).tolist() == [2, 4, 16]
    assert (3 > x).tolist() == [True, True, False]  # Python turns it into x < 3


def test_inplace_operators():
    x = sdnp.asarray([1.0, 2.0])
    before = x
    x += 1
    x *= sdnp.asarray([2.0, 3.0])
    x **= 2
    assert x is before and x.tolist() == [16.0, 81.0]
    i = sdnp.asarray([1, 2])
    with pytest.raises(TypeError):
        i += 0.5  # S6: float64 cannot go into int64
    with pytest.raises(TypeError):
        i /= 2  # divide gives float64
    b = sdnp.asarray([True])
    with pytest.raises(TypeError):
        b += True  # bool + bool is int64 (C13)
    a = sdnp.arange(5)
    a[1:] += a[:-1]
    assert a.tolist() == [0, 1, 3, 5, 7]


def test_result_layout_follows_input():
    f = sdnp.asarray(np.asfortranarray(np.ones((3, 4))))
    assert (f + 1)._flags()["f_contiguous"]
    assert sdnp.sqrt(f)._flags()["f_contiguous"]


# ------------------------------------------------------------ where, clip

def test_where():
    c = sdnp.asarray([True, False, True])
    assert_same(sdnp.where(c, sdnp.asarray([1, 2, 3]), 0.5), np.array([1.0, 0.5, 3.0]))
    assert sdnp.where(True, 1, 2) == 1
    got = sdnp.where(sdnp.asarray([[True], [False]]), sdnp.arange(3), sdnp.asarray([1j]))
    assert_same(got, np.where([[True], [False]], np.arange(3), 1j))
    with pytest.raises(TypeError):
        sdnp.where(sdnp.asarray([1, 0]), 1, 2)  # condition must be bool
    out = sdnp.zeros(3)
    assert sdnp.where(c, 1.0, 2.0, out=out) is out and out.tolist() == [1.0, 2.0, 1.0]


def test_clip():
    x = sdnp.asarray([-5, 0, 5, 10])
    assert sdnp.clip(x, min=0, max=6).tolist() == [0, 0, 5, 6]
    assert sdnp.clip(x, min=1).tolist() == [1, 1, 5, 10]
    assert sdnp.clip(x, max=1).tolist() == [-5, 0, 1, 1]
    y = sdnp.clip(x)
    assert y.tolist() == x.tolist() and y is not x
    assert sdnp.clip(x, min=sdnp.asarray([0, 1, 2, 3]), max=4).tolist() == [0, 1, 4, 4]
    assert sdnp.clip(x, min=5, max=1).tolist() == [1, 1, 1, 1]  # min > max: max wins, like numpy
    f = sdnp.asarray([math.nan, 1.0, 3.0, -0.0])
    got = sdnp.clip(f, min=0.0, max=2.0).tolist()
    assert math.isnan(got[0]) and got[1:] == [1.0, 2.0, 0.0]
    # Signed zeros follow maximum/minimum (IEEE: -0 < +0). numpy itself is not
    # consistent here: scalar bounds keep x, array bounds pick the bound.
    assert math.copysign(1, got[3]) == 1
    assert math.isnan(sdnp.clip(sdnp.asarray([1.0]), min=math.nan).tolist()[0])
    with pytest.raises(TypeError):
        sdnp.clip(x, min=0.5)  # keeps x's dtype: float bound into int64 (S6)
    with pytest.raises(TypeError):
        sdnp.clip(sdnp.asarray([1j]), min=0)
    with pytest.raises(TypeError):
        sdnp.clip(x, 0, 1)  # bounds are keyword-only
