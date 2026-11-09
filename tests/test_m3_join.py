"""M3: concat and stack."""

import numpy as np
import pytest

import stardust_numpy as sdnp
from conftest import NP_DTYPE, assert_same


def arrays(*shapes, dtype=np.float64):
    ns = [np.arange(int(np.prod(s))).astype(dtype).reshape(s) + 100 * k for k, s in enumerate(shapes)]
    return ns, [sdnp.asarray(n) for n in ns]


@pytest.mark.parametrize("shapes,axis", [
    ([(2, 3), (4, 3)], 0), ([(2, 3), (2, 1), (2, 0)], 1), ([(2, 3), (2, 2)], -1), ([(5,)], 0),
    ([(2, 2, 2), (2, 3, 2)], 1),
])
def test_concat(shapes, axis):
    ns, xs = arrays(*shapes)
    assert_same(sdnp.concat(xs, axis=axis), np.concatenate(ns, axis=axis))
    assert_same(sdnp.concat(tuple(xs), axis=axis), np.concatenate(ns, axis=axis))


def test_concat_axis_none_flattens():
    ns, xs = arrays((2, 3), (4,), (1, 2, 2))
    xs[0] = xs[0].T  # layout must not matter: C index order
    ns[0] = ns[0].T
    assert_same(sdnp.concat(xs, axis=None), np.concatenate(ns, axis=None))


def test_concat_promotes():
    got = sdnp.concat([sdnp.asarray([True]), sdnp.asarray([2]), sdnp.asarray([0.5])])
    assert_same(got, np.array([1.0, 2.0, 0.5]))
    got = sdnp.concat([sdnp.asarray([1j]), sdnp.asarray([2])])
    assert got.dtype is sdnp.complex128


def test_concat_from_strided_views():
    n = np.arange(24.0).reshape(4, 6)
    a = sdnp.asarray(n)
    assert_same(sdnp.concat([a[::2, ::-1], a.T[:2, :2]], axis=1),
                np.concatenate([n[::2, ::-1], n.T[:2, :2]], axis=1))


@pytest.mark.parametrize("bad,exc", [
    ([], ValueError), ([sdnp.zeros((2, 3)), sdnp.zeros((2, 4))], ValueError),
    ([sdnp.zeros((2, 3)), sdnp.zeros(3)], ValueError), ([sdnp.zeros(2), [1, 2]], TypeError),
    (sdnp.zeros(2), TypeError), ([np.zeros(2)], TypeError),
])
def test_concat_errors(bad, exc):
    with pytest.raises(exc):
        sdnp.concat(bad)


def test_concat_axis_out_of_bounds():
    with pytest.raises(ValueError):
        sdnp.concat([sdnp.zeros(2)], axis=1)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_stack(axis, dtype):
    ns, xs = arrays((2, 3), (2, 3), (2, 3), dtype=NP_DTYPE[dtype])
    assert_same(sdnp.stack(xs, axis=axis), np.stack(ns, axis=axis))


def test_stack_errors():
    with pytest.raises(ValueError):
        sdnp.stack([sdnp.zeros(2), sdnp.zeros(3)])
    with pytest.raises(ValueError):
        sdnp.stack([sdnp.zeros(2)], axis=2)
    with pytest.raises(ValueError):
        sdnp.stack([])
    with pytest.raises(TypeError):
        sdnp.stack([sdnp.zeros(2)], axis=None)


def test_results_are_fresh_c_contiguous():
    a = sdnp.zeros((2, 2))
    for r in (sdnp.concat([a, a]), sdnp.stack([a, a], axis=2)):
        assert r.base is None and r._flags()["c_contiguous"]
