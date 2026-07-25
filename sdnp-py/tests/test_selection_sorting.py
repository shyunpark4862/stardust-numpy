import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, assert_matches, numpy_call

VALUES = {
    bool: [[True, False, True], [False, False, True]],
    int: [[3, -1, 3], [0, 2, -4]],
    float: [[3.5, -1.0, 3.5], [0.0, 2.25, -4.0]],
    complex: [[3 + 1j, -1 + 2j, 3 + 1j], [0j, 2 - 1j, -4j]],
}


def make_array(values, dtype):
    return sdnp.array(values, dtype=dtype)


@pytest.mark.parametrize("dtype", (bool, int, float, complex))
def test_where_broadcasts_and_promotes_supported_dtypes(dtype):
    condition = sdnp.array([[True], [False]], dtype=bool)
    x_values = VALUES[dtype][:2]
    x = make_array(x_values, dtype)
    y = sdnp.array([False, True, False], dtype=bool)

    expected = np.where(
        np.array([[True], [False]]),
        np.asarray(x_values, dtype=NP_DTYPES[dtype]),
        np.array([False, True, False]),
    )
    assert_matches(sdnp.where(condition, x, y), expected)


def test_where_accepts_array_like_branches_and_scalar_broadcasting():
    condition = sdnp.array([[True, False], [False, True]], dtype=bool)
    assert_matches(
        sdnp.where(condition, 2, [[0.5, 1.5], [2.5, 3.5]]),
        np.where(
            [[True, False], [False, True]],
            2,
            [[0.5, 1.5], [2.5, 3.5]],
        ),
    )


@pytest.mark.parametrize("dtype", (bool, int, float, complex))
def test_nonzero_matches_numpy_for_all_dtypes(dtype):
    values = np.asarray(VALUES[dtype], dtype=NP_DTYPES[dtype])
    actual = sdnp.nonzero(make_array(values.tolist(), dtype))
    expected = np.nonzero(values)

    assert isinstance(actual, tuple)
    assert len(actual) == values.ndim
    for actual_axis, expected_axis in zip(actual, expected, strict=True):
        assert_matches(actual_axis, expected_axis)


def test_nonzero_empty_array_has_one_empty_coordinate_per_axis():
    actual = sdnp.nonzero(sdnp.zeros((2, 0, 3), dtype=float))
    assert len(actual) == 3
    for coordinate in actual:
        assert_matches(coordinate, np.array([], dtype=np.int64))


@pytest.mark.parametrize(
    ("dtype", "minimum", "maximum"),
    [
        (bool, False, True),
        (int, -1, 2),
        (float, -1.25, 2.5),
    ],
)
def test_clip_matches_numpy_and_preserves_dtype(dtype, minimum, maximum):
    values = np.asarray(VALUES[dtype], dtype=NP_DTYPES[dtype])
    assert_matches(
        sdnp.clip(make_array(values.tolist(), dtype), minimum, maximum),
        np.clip(values, minimum, maximum),
    )


@pytest.mark.parametrize(
    ("minimum", "maximum"), [(None, 1), (0, None), (None, None)]
)
def test_clip_supports_open_bounds(minimum, maximum):
    values = np.array([-3.5, -0.5, 2.0, 7.0])
    assert_matches(
        sdnp.clip(sdnp.array(values.tolist()), minimum, maximum),
        np.clip(values, minimum, maximum),
    )


@pytest.mark.parametrize("dtype", (bool, int, float))
@pytest.mark.parametrize("axis", (None, 0, 1, -1))
def test_sort_and_argsort_match_stable_numpy(dtype, axis):
    values = np.asarray(VALUES[dtype], dtype=NP_DTYPES[dtype])
    array = make_array(values.tolist(), dtype)

    assert_matches(
        sdnp.sort(array, axis=axis),
        numpy_call(np.sort, values, axis=axis, kind="stable"),
    )
    assert_matches(
        sdnp.argsort(array, axis=axis),
        numpy_call(np.argsort, values, axis=axis, kind="stable"),
    )


def test_argsort_is_stable_for_duplicate_values():
    values = [2, 1, 2, 1, 2, 1]
    actual = sdnp.argsort(sdnp.array(values, dtype=int))
    assert_matches(actual, np.array([1, 3, 5, 0, 2, 4], dtype=np.int64))


def test_float_sort_argsort_and_unique_group_nan_last():
    values = np.array([np.nan, 2.0, np.nan, -1.0, 2.0, np.nan])
    array = sdnp.array(values.tolist(), dtype=float)

    assert_matches(sdnp.sort(array), np.sort(values, kind="stable"))
    assert_matches(sdnp.argsort(array), np.argsort(values, kind="stable"))
    assert_matches(sdnp.unique(array), np.unique(values))


@pytest.mark.parametrize("dtype", (bool, int, float, complex))
def test_unique_flattens_and_sorts_all_supported_dtypes(dtype):
    values = np.asarray(VALUES[dtype], dtype=NP_DTYPES[dtype])
    assert_matches(
        sdnp.unique(make_array(values.tolist(), dtype)),
        np.unique(values),
    )


def test_complex_unique_groups_all_nan_components():
    values = np.array(
        [1 + 2j, complex(np.nan, 1), 1 + 2j, complex(3, np.nan), -1j],
        dtype=np.complex128,
    )
    actual = sdnp.unique(sdnp.array(values.tolist(), dtype=complex))
    expected = np.array([-1j, 1 + 2j, complex(np.nan, 1)])
    assert_matches(actual, expected)


def test_selection_and_sorting_handle_strided_inputs():
    np_base = np.arange(30).reshape(5, 6)
    sd_base = sdnp.arange(30).reshape((5, 6))
    np_view = np_base[::-2, 1::2]
    sd_view = sd_base[::-2, 1::2]

    condition = sdnp.array([[True, False, True]], dtype=bool)
    assert_matches(
        sdnp.where(condition, sd_view, -1),
        np.where([[True, False, True]], np_view, -1),
    )
    actual_nonzero = sdnp.nonzero(sd_view)
    for actual, expected in zip(
        actual_nonzero, np.nonzero(np_view), strict=True
    ):
        assert_matches(actual, expected)
    assert_matches(sdnp.clip(sd_view, 8, 24), np.clip(np_view, 8, 24))
    assert_matches(
        sdnp.sort(sd_view, axis=0),
        np.sort(np_view, axis=0, kind="stable"),
    )
    assert_matches(
        sdnp.argsort(sd_view, axis=-1),
        np.argsort(np_view, axis=-1, kind="stable"),
    )
    assert_matches(sdnp.unique(sd_view), np.unique(np_view))


def test_sorting_and_unique_empty_inputs():
    array = sdnp.zeros((2, 0, 3), dtype=float)
    assert_matches(sdnp.sort(array, axis=1), np.zeros((2, 0, 3)))
    assert_matches(
        sdnp.argsort(array, axis=1),
        np.zeros((2, 0, 3), dtype=np.int64),
    )
    assert_matches(sdnp.sort(array), np.array([], dtype=np.float64))
    assert_matches(sdnp.unique(array), np.array([], dtype=np.float64))


def test_where_rejects_non_bool_condition_and_incompatible_shapes():
    with pytest.raises(TypeError, match="bool"):
        sdnp.where(sdnp.array([0, 1], dtype=int), [1, 2], [3, 4])
    with pytest.raises(ValueError, match="broadcast"):
        sdnp.where(
            sdnp.array([[True, False]], dtype=bool),
            sdnp.zeros((2, 3)),
            sdnp.zeros((4,)),
        )


def test_clip_rejects_complex_array_or_bounds_and_nonscalar_bounds():
    with pytest.raises(ValueError, match="complex"):
        sdnp.clip(sdnp.array([1 + 1j], dtype=complex), 0, 1)
    with pytest.raises(TypeError, match="real scalar"):
        sdnp.clip(sdnp.array([1.0]), 0j, 1)
    with pytest.raises(ValueError, match="scalar"):
        sdnp.clip(sdnp.array([1.0]), sdnp.array([0.0]), 1)


@pytest.mark.parametrize("operation", (sdnp.sort, sdnp.argsort))
def test_sort_operations_reject_complex_and_invalid_axis(operation):
    with pytest.raises(ValueError, match="complex"):
        operation(sdnp.array([1 + 1j], dtype=complex))
    with pytest.raises(IndexError, match="axis"):
        operation(sdnp.array([[2, 1]], dtype=int), axis=2)
    with pytest.raises(TypeError):
        operation(sdnp.array([[2, 1]], dtype=int), axis=0.5)
