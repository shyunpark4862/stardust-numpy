"""M1: dtype objects, ndarray attributes and Python protocols."""

import operator

import numpy as np
import pytest

import stardust_numpy as sdnp


def test_dtype_objects_are_four_distinct_singletons():
    dts = [sdnp.bool, sdnp.int64, sdnp.float64, sdnp.complex128]
    assert len({id(d) for d in dts}) == 4
    assert [repr(d) for d in dts] == ["bool", "int64", "float64", "complex128"]
    assert sdnp.asarray([1.0]).dtype is sdnp.float64


@pytest.mark.parametrize("bad", [float, int, "float64", np.float64, np.dtype("float64")])
def test_dtype_argument_accepts_only_dtype_objects(bad):
    with pytest.raises(TypeError):
        sdnp.zeros(3, dtype=bad)


def test_ndarray_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        sdnp.ndarray()


def test_attributes(dtype):
    a = sdnp.zeros((2, 3, 4), dtype=dtype)
    item = {sdnp.bool: 1, sdnp.int64: 8, sdnp.float64: 8, sdnp.complex128: 16}[dtype]
    assert a.shape == (2, 3, 4)
    assert a.ndim == 3
    assert a.size == 24
    assert a.itemsize == item
    assert a.nbytes == 24 * item
    assert a.strides == (12 * item, 4 * item, item)
    assert a.base is None
    assert a._flags() == {"c_contiguous": True, "f_contiguous": False, "writeable": True}


def test_shape_one_axes_do_not_break_contiguity():
    a = sdnp.zeros((1, 5, 1))
    assert a._flags()["c_contiguous"] and a._flags()["f_contiguous"]


@pytest.mark.parametrize("convert", [bool, int, float, complex, operator.index])
def test_arrays_never_convert_to_scalars(convert):
    with pytest.raises(TypeError):
        convert(sdnp.asarray([1]))  # Q4: even a single element


def test_arrays_are_unhashable():
    with pytest.raises(TypeError):
        hash(sdnp.zeros(2))


def test_max_ndim_is_32():
    assert sdnp.zeros((1,) * 32).ndim == 32
    with pytest.raises(ValueError):
        sdnp.zeros((1,) * 33)


def test_no_reference_leaks():
    """Stands in for nanobind's (disabled) shutdown leak check."""
    import gc
    import sys

    dts = [sdnp.bool, sdnp.int64, sdnp.float64, sdnp.complex128]
    src = np.arange(6.0).reshape(2, 3)

    def work():
        for d in dts:
            a = sdnp.zeros((3, 2), dtype=d)
            a.dtype, a.base, repr(a), a.tolist()
            np.asarray(sdnp.asarray(np.asarray(a)))
        sdnp.asarray(src)
        try:
            sdnp.asarray([1j], dtype=sdnp.float64)
        except TypeError:
            pass

    work()
    gc.collect()
    before = [sys.getrefcount(o) for o in [*dts, src]]
    for _ in range(2000):
        work()
    gc.collect()
    assert [sys.getrefcount(o) for o in [*dts, src]] == before


def test_memory_is_cache_line_aligned():
    for n in (1, 3, 1000):
        addr = np.asarray(sdnp.empty(n)).__array_interface__["data"][0]
        assert addr % 128 == 0
