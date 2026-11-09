"""M3: shape manipulation. Views share memory; numpy is the oracle for values."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust_numpy as sdnp
from conftest import assert_same


def shares(view, owner_np):
    return np.shares_memory(np.asarray(view), owner_np)


def grid(*shape):
    n = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    return n, sdnp.asarray(n)


# --------------------------------------------------------------------- reshape

@pytest.mark.parametrize("src,dst", [
    ((12,), (3, 4)), ((3, 4), (12,)), ((3, 4), (2, 6)), ((2, 3, 4), (6, 4)),
    ((2, 3, 4), (2, -1)), ((3, 4), (-1,)), ((1, 12, 1), (3, 1, 4)), ((0, 3), (3, 0)), ((0, 3), (-1, 5)),
])
def test_reshape_contiguous_is_a_view(src, dst):
    n, a = grid(*src)
    r = sdnp.reshape(a, dst)
    assert_same(r, n.reshape(dst))
    assert shares(r, n) or n.size == 0


def test_reshape_strided_view_when_strides_allow():
    n = np.arange(48.0).reshape(4, 12)[:, ::2]  # (4, 6), rows not contiguous
    r = sdnp.reshape(sdnp.asarray(n), (4, 2, 3))  # splits only the inner axis
    assert_same(r, n.reshape(4, 2, 3))
    assert shares(r, n)


def test_reshape_needs_copy_for_transpose():
    n = np.arange(12.0).reshape(3, 4).T
    a = sdnp.asarray(n)
    r = sdnp.reshape(a, (12,))
    assert_same(r, n.reshape(12))
    assert not shares(r, n) and r.base is None
    with pytest.raises(ValueError):
        sdnp.reshape(a, (12,), copy=False)


def test_reshape_copy_true_always_copies():
    n, a = grid(3, 4)
    r = sdnp.reshape(a, (4, 3), copy=True)
    assert_same(r, n.reshape(4, 3))
    assert not shares(r, n)


@pytest.mark.parametrize("bad,exc", [
    ((5,), ValueError), ((-1, -1), ValueError), ((-1, 5), ValueError), ((), ValueError), ((-2, 6), ValueError),
    (3.0, TypeError), ([3, 4], TypeError),
])
def test_reshape_errors(bad, exc):
    _, a = grid(3, 4)
    with pytest.raises(exc):
        sdnp.reshape(a, bad)


def test_reshape_unknown_extent_of_empty_array():
    _, a = grid(0, 3)
    with pytest.raises(ValueError):
        sdnp.reshape(a, (-1, 0))  # any extent would do: ambiguous


def test_reshape_method_and_signature():
    n, a = grid(2, 3)
    assert_same(a.reshape((3, 2)), n.reshape(3, 2))
    assert_same(a.reshape(6), n.reshape(6))
    with pytest.raises(TypeError):
        a.reshape(3, 2)  # C2: one shape argument
    with pytest.raises(TypeError):
        sdnp.reshape(a, (6,), False)  # copy is keyword-only


@given(st.data())
def test_reshape_matches_numpy_on_random_views(data):
    base = np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5)
    index = tuple(data.draw(st.sampled_from([slice(None), slice(None, None, -1), slice(None, None, 2), slice(1, 2)]))
                  for _ in range(4))
    view = base[index]
    if data.draw(st.booleans()):
        view = view.transpose(data.draw(st.permutations(range(4))))
    # A random factorisation of the size into a random number of axes.
    size = view.size
    shape = []
    while size > 1:
        f = data.draw(st.sampled_from([d for d in range(2, size + 1) if size % d == 0]))
        shape.append(f)
        size //= f
    shape = shape or [1]
    shape.insert(data.draw(st.integers(0, len(shape))), 1)
    r = sdnp.reshape(sdnp.asarray(view), tuple(shape))
    want = view.reshape(shape)
    assert_same(r, want)
    # View exactly when numpy can view (same algorithm).
    assert shares(r, base) == np.shares_memory(want, base)


def test_ravel():
    n, a = grid(3, 4)
    r = sdnp.ravel(a)
    assert_same(r, n.ravel())
    assert shares(r, n)
    t = sdnp.ravel(a.T)
    assert_same(t, n.T.ravel())
    assert not shares(t, n)
    # A 1-D strided array is already "flat": ravel views it (numpy copies).
    s = sdnp.ravel(a[0, ::2])
    assert_same(s, n[0, ::2])
    assert shares(s, n)


# ----------------------------------------------------------------- transposes

def test_T_and_mT():
    n, a = grid(3, 4)
    assert_same(a.T, n.T)
    assert shares(a.T, n)
    n3, a3 = grid(2, 3, 4)
    assert_same(a3.mT, np.swapaxes(n3, -1, -2))
    assert_same(sdnp.matrix_transpose(a3), np.swapaxes(n3, -1, -2))
    with pytest.raises(ValueError):
        a3.T  # C10: only 2-D
    with pytest.raises(ValueError):
        sdnp.arange(3).T
    with pytest.raises(ValueError):
        sdnp.arange(3).mT


def test_permute_dims():
    n, a = grid(2, 3, 4)
    for axes in [(0, 1, 2), (2, 0, 1), (1, 2, 0), (-1, 0, 1)]:
        assert_same(sdnp.permute_dims(a, axes), n.transpose(axes))
    for bad in [(0, 1), (0, 0, 1), (0, 1, 3)]:
        with pytest.raises(ValueError):
            sdnp.permute_dims(a, bad)
    assert_same(sdnp.permute_dims(sdnp.arange(3), 0), np.arange(3))


@pytest.mark.parametrize("src,dst", [(0, 2), (-1, 0), ((0, 1), (2, 0)), ((0, 2), (-1, -3)), ((), ())])
def test_moveaxis(src, dst):
    n, a = grid(2, 3, 4)
    assert_same(sdnp.moveaxis(a, src, dst), np.moveaxis(n, src, dst))


def test_moveaxis_errors():
    _, a = grid(2, 3, 4)
    with pytest.raises(ValueError):
        sdnp.moveaxis(a, (0, 1), 2)
    with pytest.raises(ValueError):
        sdnp.moveaxis(a, (0, 0), (1, 2))
    with pytest.raises(ValueError):
        sdnp.moveaxis(a, 3, 0)


# --------------------------------------------------- adding and removing axes

def test_squeeze():
    n, a = grid(1, 3, 1, 4)
    assert_same(sdnp.squeeze(a, axis=0), n.squeeze(0))
    assert_same(sdnp.squeeze(a, axis=(0, -2)), n.squeeze((0, 2)))
    with pytest.raises(ValueError):
        sdnp.squeeze(a, axis=1)  # extent 3
    with pytest.raises(ValueError):
        sdnp.squeeze(sdnp.zeros((1, 1)), axis=(0, 1))  # would be 0-d (Z2)
    with pytest.raises(TypeError):
        sdnp.squeeze(a)  # axis is required


def test_expand_dims():
    n, a = grid(2, 3)
    for axis in [0, 1, 2, -1, -3, (0, 2), (0, -1), (1, 2, 3)]:
        got = sdnp.expand_dims(a, axis=axis)
        assert_same(got, np.expand_dims(n, axis))
        assert shares(got, n)
    for bad in [3, -4, (0, 0)]:
        with pytest.raises(ValueError):
            sdnp.expand_dims(a, axis=bad)
    with pytest.raises(ValueError):
        sdnp.expand_dims(sdnp.zeros((1,) * 31), axis=(0, 1))


# --------------------------------------------------------------- broadcasting

def test_broadcast_to_is_a_readonly_view():
    n = np.arange(3.0)
    b = sdnp.broadcast_to(sdnp.asarray(n), (2, 3))
    assert_same(b, np.broadcast_to(n, (2, 3)))
    assert b.strides == (0, 8)
    assert not b._flags()["writeable"]
    with pytest.raises(ValueError):
        b[0, 0] = 1.0
    assert shares(b, n)


@pytest.mark.parametrize("shape", [(2,), (3, 2), (), (-1, 3)])
def test_broadcast_to_errors(shape):
    with pytest.raises(ValueError):
        sdnp.broadcast_to(sdnp.arange(3), shape)


def test_broadcast_to_is_one_directional():
    with pytest.raises(ValueError):
        sdnp.broadcast_to(sdnp.zeros((2, 3)), (1, 3))


def test_broadcast_arrays():
    x, y = np.arange(3.0), np.arange(2).reshape(2, 1)
    got = sdnp.broadcast_arrays(sdnp.asarray(x), sdnp.asarray(y))
    want = np.broadcast_arrays(x, y)
    assert isinstance(got, tuple) and len(got) == 2
    for g, w in zip(got, want):
        assert_same(g, w)
        assert not g._flags()["writeable"]
    assert sdnp.broadcast_arrays() == ()
    with pytest.raises(ValueError):
        sdnp.broadcast_arrays(sdnp.zeros(2), sdnp.zeros(3))
    with pytest.raises(TypeError):
        sdnp.broadcast_arrays(sdnp.zeros(2), [1, 2])


# --------------------------------------------------------------- real / imag

def test_real_imag_of_complex_are_writeable_views():
    n = np.array([1 + 2j, 3 - 4j, -0.0 + 5j])
    a = sdnp.asarray(n)
    re, im = sdnp.real(a), sdnp.imag(a)
    assert_same(re, n.real)
    assert_same(im, n.imag)
    assert re.strides == (16,) and im.strides == (16,)
    im[1] = 9.0
    assert n[1] == 3 + 9j  # writes go through to the complex array


def test_real_imag_of_strided_complex():
    n = (np.arange(24.0) + 1j * np.arange(24.0)[::-1]).reshape(4, 6)[::-1, ::2]
    a = sdnp.asarray(n)
    assert_same(sdnp.real(a), n.real)
    assert_same(sdnp.imag(a).T, n.imag.T)


def test_real_imag_of_real_dtypes(dtype):
    if dtype is sdnp.complex128:
        return
    a = sdnp.ones((2, 3), dtype=dtype)
    assert sdnp.real(a) is a
    im = sdnp.imag(a)
    assert im.dtype is dtype and im.tolist() == [[0] * 3] * 2
    assert not im._flags()["writeable"]


def test_real_imag_out():
    a = sdnp.asarray([1 + 2j, 3 + 4j])
    out = sdnp.zeros((2, 2))
    assert sdnp.imag(a, out=out) is out
    assert out.tolist() == [[2, 4], [2, 4]]  # broadcast into out
    with pytest.raises(TypeError):
        sdnp.real(a, out=sdnp.zeros(2, dtype=sdnp.int64))  # S6: float64 -> int64
    with pytest.raises(TypeError):
        sdnp.real(a, sdnp.zeros(2))  # out is keyword-only


# ------------------------------------------------------------------- base

def test_base_is_the_final_owner():
    a = sdnp.zeros((4, 4))
    v = a[1:][:, 2:].T[None].reshape((2, 3))
    assert v.base is a
    n = np.zeros((4, 4))
    w = sdnp.asarray(n)[1:]
    assert w.base is n  # imported memory: numpy's array owns it
