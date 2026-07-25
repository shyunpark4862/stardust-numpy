import sys
import warnings

import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, assert_matches
from hypothesis import given
from strategies import shaped_values

VALUES = {
    bool: [[False, True, False], [True, False, True]],
    int: [[0, 1, -2], [3, -4, 5]],
    float: [[0.0, 1.5, -2.0], [3.25, -4.5, 5.0]],
    complex: [
        [0j, 1 + 2j, -2 + 0.5j],
        [3 - 1j, -4 + 2j, 5 + 0j],
    ],
}


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_array_properties_and_list_conversion(dtype):
    values = VALUES[dtype]
    array = sdnp.array(values, dtype=dtype)

    assert array.shape == [2, 3]
    assert array.strides == [3, 1]
    assert array.ndim == 2
    assert array.size == 6
    assert array.dtype is dtype
    assert array.to_list() == values
    assert len(array) == 2
    assert_matches(array, np.asarray(values, dtype=NP_DTYPES[dtype]))


@given(shaped_values(max_dims=3, max_side=3))
@pytest.mark.property
def test_properties_match_numpy_for_generated_arrays(case):
    dtype, shape, values = case
    expected = np.asarray(values, dtype=NP_DTYPES[dtype]).reshape(shape)
    array = (
        sdnp.zeros(shape, dtype=dtype)
        if expected.size == 0
        else sdnp.array(values, dtype=dtype)
    )

    assert tuple(array.shape) == expected.shape
    assert array.ndim == expected.ndim
    assert array.size == expected.size
    assert array.dtype is dtype
    assert_matches(array, expected)


def test_empty_array_properties_and_element_strides():
    array = sdnp.zeros((2, 0, 3), dtype=int)
    assert array.shape == [2, 0, 3]
    assert array.strides == [0, 3, 1]
    assert array.ndim == 3
    assert array.size == 0
    assert array.to_list() == [[], []]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([True, False], "array([True, False], dtype=bool)"),
        ([1, -2], "array([1, -2], dtype=int64)"),
        ([1.5, float("inf")], "array([1.5, inf], dtype=float64)"),
        ([1 + 2j, 3 + 0j], "array([(1+2j), (3+0j)], dtype=complex128)"),
    ],
)
def test_repr_and_str_include_values_and_fixed_width_dtype(values, expected):
    array = sdnp.array(values)
    assert repr(array) == expected
    assert str(array) == expected


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_flat_and_axis_zero_iteration(dtype):
    expected = np.asarray(VALUES[dtype], dtype=NP_DTYPES[dtype])
    array = sdnp.array(VALUES[dtype], dtype=dtype)

    assert list(array.flat) == expected.ravel().tolist()
    rows = list(array)
    assert len(rows) == 2
    assert all(isinstance(row, sdnp.Array) for row in rows)
    for row, expected_row in zip(rows, expected, strict=True):
        assert_matches(row, expected_row)


def test_copy_is_c_contiguous_and_independent():
    source = sdnp.arange(6).reshape((2, 3)).transpose()
    copied = source.copy()

    assert copied.shape == [3, 2]
    assert copied.strides == [2, 1]
    assert_matches(copied, np.arange(6).reshape(2, 3).T)

    copied[0, 0] = 99
    assert source[0, 0] == 0
    source[1, 0] = 77
    assert copied[1, 0] == 1


@pytest.mark.parametrize(
    "make_view",
    [
        pytest.param(lambda a: a.reshape((3, 2)), id="reshape"),
        pytest.param(lambda a: a.transpose(), id="transpose"),
        pytest.param(lambda a: a.permute_axes((1, 0)), id="permute_axes"),
    ],
)
def test_views_use_copy_on_write(make_view):
    source = sdnp.arange(6).reshape((2, 3))
    view = make_view(source)
    original_source = source.to_list()

    view[0, 0] = 100
    assert source.to_list() == original_source
    assert view[0, 0] == 100

    view_snapshot = view.to_list()
    source[0, 0] = 200
    assert view.to_list() == view_snapshot


def test_array_from_array_is_a_copy_on_write_alias():
    source = sdnp.array([[1, 2], [3, 4]])
    alias = sdnp.array(source)

    alias[0, 0] = 9
    assert source[0, 0] == 1
    source[1, 1] = 8
    assert alias[1, 1] == 4


@pytest.mark.parametrize("source_dtype", [bool, int, float, complex])
@pytest.mark.parametrize("target_dtype", [bool, int, float, complex])
def test_astype_supports_every_dtype_pair(source_dtype, target_dtype):
    source_values = {
        bool: [False, True],
        int: [0, 1],
        float: [0.0, 1.0],
        complex: [0j, 1 + 0j],
    }[source_dtype]
    source = sdnp.array(source_values, dtype=source_dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        expected = np.asarray(
            source_values, dtype=NP_DTYPES[source_dtype]
        ).astype(NP_DTYPES[target_dtype])
    result = source.astype(target_dtype)

    assert result.dtype is target_dtype
    assert result.strides == [1]
    assert_matches(result, expected)

    result[0] = target_dtype(1)
    assert source[0] == source_values[0]


@pytest.mark.parametrize("dtype", [str, object, "float64", np.float64])
def test_astype_rejects_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="unsupported dtype"):
        sdnp.array([1]).astype(dtype)


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        (6, (6,)),
        ((3, 2), (3, 2)),
        ([1, 6], (1, 6)),
        ((-1, 3), (2, 3)),
        ((2, -1, 1), (2, 3, 1)),
    ],
)
def test_reshape_signatures_and_inferred_dimension(shape, expected_shape):
    result = sdnp.arange(6).reshape(shape)
    expected = np.arange(6).reshape(expected_shape)
    assert_matches(result, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (4, 2),
        (-1, -1),
        (-2, 3),
        (0, -1),
        (1.5, 4),
        (sys.maxsize, 3),
    ],
)
def test_reshape_rejects_invalid_shapes_and_overflow(shape):
    with pytest.raises((ValueError, OverflowError)):
        sdnp.arange(6).reshape(shape)


def test_reshape_of_noncontiguous_view_preserves_c_order_values():
    transposed = sdnp.arange(6).reshape((2, 3)).T
    reshaped = transposed.reshape((2, 3))
    expected = np.arange(6).reshape(2, 3).T.reshape(2, 3)
    assert reshaped.strides == [3, 1]
    assert_matches(reshaped, expected)


def test_squeeze_without_axis_and_with_int_or_sequence_axes():
    source = sdnp.arange(6).reshape((1, 2, 1, 3, 1))
    expected = np.arange(6).reshape((1, 2, 1, 3, 1))

    assert_matches(source.squeeze(), np.squeeze(expected))
    assert_matches(source.squeeze(0), np.squeeze(expected, axis=0))
    assert_matches(source.squeeze(-1), np.squeeze(expected, axis=-1))
    assert_matches(
        source.squeeze((0, 2, -1)), np.squeeze(expected, axis=(0, 2, -1))
    )


def test_squeeze_that_removes_every_axis_returns_a_scalar():
    result = sdnp.array(7, shape=(1, 1)).squeeze()
    assert type(result) is int
    assert result == 7


@pytest.mark.parametrize("axis", [1, (0, 0), (), 5, "bad"])
def test_squeeze_rejects_invalid_axes(axis):
    with pytest.raises((TypeError, ValueError, IndexError)):
        sdnp.zeros((1, 2, 1)).squeeze(axis)


def test_transpose_T_and_permute_axes_match_numpy_layout():
    source = sdnp.arange(24).reshape((2, 3, 4))
    expected = np.arange(24).reshape((2, 3, 4))

    for result in (source.transpose(), source.T):
        assert result.shape == [4, 3, 2]
        assert result.strides == [1, 4, 12]
        assert_matches(result, expected.transpose())

    permuted = source.permute_axes((1, -1, 0))
    assert permuted.shape == [3, 4, 2]
    assert permuted.strides == [4, 1, 12]
    assert_matches(permuted, np.transpose(expected, (1, 2, 0)))


@pytest.mark.parametrize("axes", [(0, 1), (0, 1, 1), (0, 1, 3), (), "bad"])
def test_permute_axes_rejects_non_permutations(axes):
    with pytest.raises((TypeError, ValueError, IndexError)):
        sdnp.zeros((2, 3, 4)).permute_axes(axes)


def test_one_dimensional_transpose_is_a_copy_on_write_view():
    source = sdnp.arange(3)
    view = source.T
    assert view.shape == [3]
    assert view.strides == [1]

    view[0] = 9
    assert source[0] == 0


def test_method_argument_counts_and_keywords_follow_declared_signatures():
    array = sdnp.arange(3)
    with pytest.raises(TypeError):
        array.copy(1)
    with pytest.raises(TypeError):
        array.transpose((0,))
    assert array.reshape(shape=(3,)).shape == [3]
    with pytest.raises(TypeError):
        array.permute_axes()
