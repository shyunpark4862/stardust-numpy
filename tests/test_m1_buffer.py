"""M1: buffer protocol in both directions, tolist, repr/str."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust_numpy as sdnp
from conftest import assert_same

# --------------------------------------------------------- import (numpy -> sdnp)

@pytest.mark.parametrize("np_dtype", [np.bool_, np.int64, np.float64, np.complex128])
def test_exact_dtypes_are_shared_without_copy(np_dtype):
    src = np.arange(6).astype(np_dtype).reshape(2, 3)
    a = sdnp.asarray(src)
    assert a.base is src
    src[0, 0] = 1
    assert a.tolist()[0][0] == np_dtype(1).item()


def test_strided_numpy_views_are_shared_with_their_strides():
    src = np.arange(24.0).reshape(4, 6)[::2, ::-3]
    a = sdnp.asarray(src)
    assert a.strides == src.strides and a.base is src
    assert_same(a, src)


@pytest.mark.parametrize(
    "np_dtype,target",
    [(np.int8, sdnp.int64), (np.int16, sdnp.int64), (np.int32, sdnp.int64), (np.uint8, sdnp.int64),
     (np.uint16, sdnp.int64), (np.uint32, sdnp.int64), (np.float16, sdnp.float64), (np.float32, sdnp.float64),
     (np.complex64, sdnp.complex128)],
)
def test_lossless_widening_copies(np_dtype, target):
    src = np.array([[1, 2], [3, 250]]).astype(np_dtype)
    a = sdnp.asarray(src)
    assert a.dtype is target and a.base is None
    assert a.tolist() == src.astype(np.complex128 if target is sdnp.complex128 else src.dtype).tolist()
    with pytest.raises(ValueError):
        sdnp.asarray(src, copy=False)


def test_lossy_or_foreign_formats_are_rejected():
    with pytest.raises(TypeError):
        sdnp.asarray(np.array([1], dtype=np.uint64))
    with pytest.raises(TypeError):
        sdnp.asarray(np.array([1.0], dtype=">f8"))
    with pytest.raises(TypeError):
        sdnp.asarray(np.array(["a"]))


def test_zero_dim_buffers_are_rejected():
    with pytest.raises(ValueError):
        sdnp.asarray(np.float64(1.0).reshape(()))


def test_readonly_buffers_stay_readonly():
    src = np.arange(3.0)
    src.flags.writeable = False
    a = sdnp.asarray(src)
    assert a._flags()["writeable"] is False
    assert not np.asarray(a).flags.writeable


def test_bytes_are_unsigned_bytes():
    assert_same(sdnp.asarray(b"\x00\x7f\xff"), np.array([0, 127, 255], dtype=np.int64))


def test_explicit_dtype_on_buffer_import():
    assert sdnp.asarray(np.array([1.7, -2.2], dtype=np.float32), dtype=sdnp.int64).tolist() == [1, -2]
    with pytest.raises(TypeError):
        sdnp.asarray(np.array([1j]), dtype=sdnp.float64)


def test_copy_true_on_shareable_buffer():
    src = np.arange(3.0)
    a = sdnp.asarray(src, copy=True)
    assert a.base is None
    src[0] = 99
    assert a.tolist()[0] == 0.0


def test_misaligned_buffer_is_copied():
    raw = np.zeros(8 * 4 + 1, dtype=np.uint8)
    src = raw[1:].view(np.float64)  # data pointer is 1 byte off
    a = sdnp.asarray(src)
    assert a.base is None and a.tolist() == [0.0] * 4


# --------------------------------------------------------- export (sdnp -> numpy)

def test_numpy_sees_our_memory():
    a = sdnp.zeros((2, 3), dtype=sdnp.int64)
    view = np.asarray(a)
    view[1, 2] = 42  # writes through to stardust's buffer
    assert a.tolist() == [[0, 0, 0], [0, 0, 42]]
    assert memoryview(a).format == "q"
    assert memoryview(sdnp.zeros(1, dtype=sdnp.complex128)).format == "Zd"


def test_exported_view_keeps_array_alive():
    view = np.asarray(sdnp.full(4, 7))
    import gc
    gc.collect()
    assert view.tolist() == [7, 7, 7, 7]


@given(hnp.arrays(
    dtype=st.sampled_from([np.bool_, np.int64, np.float64, np.complex128]),
    shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=0, max_side=5),
))
def test_roundtrip_any_numpy_array(src):
    assert_same(sdnp.asarray(src), src)


# --------------------------------------------------------- tolist / repr / str

def test_tolist_gives_python_scalars(dtype):
    out = sdnp.ones((2, 2), dtype=dtype).tolist()
    assert type(out[0][0]) is {sdnp.bool: bool, sdnp.int64: int, sdnp.float64: float, sdnp.complex128: complex}[dtype]


def test_repr_and_str():
    a = sdnp.asarray([[1, 22], [-3, 4]])
    assert repr(a) == "array([[ 1, 22],\n       [-3,  4]], dtype=int64)"
    assert str(a) == "[[ 1 22]\n [-3  4]]"
    assert repr(sdnp.asarray([True, False])) == "array([ True, False], dtype=bool)"
    assert repr(sdnp.zeros(0)) == "array([], dtype=float64)"
    assert repr(sdnp.zeros((2, 0))) == "array([], shape=(2, 0), dtype=float64)"
    assert str(sdnp.zeros((2, 0))) == "[]"


def test_repr_of_3d_separates_blocks_with_blank_lines():
    assert str(sdnp.zeros((2, 1, 2), dtype=sdnp.int64)) == "[[[0 0]]\n\n [[0 0]]]"


def test_large_arrays_are_summarized():
    r = repr(sdnp.arange(10_000))
    assert r == "array([   0,    1,    2, ..., 9997, 9998, 9999], dtype=int64)"
    s = str(sdnp.zeros((100, 100), dtype=sdnp.int64))
    assert s.count("...") == 7 and len(s.splitlines()) == 7  # 6 elided columns + 1 elided row


def test_long_rows_wrap_at_75_columns():
    r = repr(sdnp.arange(1000, 1030))
    assert all(len(line) <= 75 for line in r.splitlines())
    assert r.splitlines()[1].startswith("       ")
