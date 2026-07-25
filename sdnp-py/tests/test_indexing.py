import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, assert_matches


def _array(values, dtype=int):
    return sdnp.array(
        np.asarray(values, dtype=NP_DTYPES[dtype]).tolist(), dtype=dtype
    )


def _sample(dtype=int):
    values = np.arange(3 * 4 * 5, dtype=np.int64).reshape(3, 4, 5)
    if dtype is bool:
        values = values % 3 == 0
    else:
        values = values.astype(NP_DTYPES[dtype])
    return _array(values, dtype), values


@pytest.mark.parametrize(
    "index",
    [
        1,
        -1,
        (1, 2),
        (-1, -2, -3),
        slice(None),
        slice(1, None),
        slice(None, None, 2),
        slice(None, None, -1),
        (slice(None, None, -1), slice(3, 0, -2), slice(None, None, -2)),
        (Ellipsis, 2),
        (1, Ellipsis),
        (None, slice(None), 2, None, Ellipsis),
        (slice(0, 0), Ellipsis),
    ],
    ids=[
        "integer",
        "negative-integer",
        "integer-tuple",
        "negative-tuple",
        "full-slice",
        "bounded-slice",
        "strided-slice",
        "reverse",
        "mixed-reverse-slices",
        "leading-ellipsis",
        "trailing-ellipsis",
        "newaxis-and-ellipsis",
        "empty-slice",
    ],
)
def test_basic_indexing_matches_numpy(dtype, index):
    actual, expected = _sample(dtype)
    assert_matches(actual[index], expected[index])


@pytest.mark.parametrize(
    ("sdnp_index", "numpy_index"),
    [
        (
            lambda: _array([True, False, True], bool),
            lambda: np.array([True, False, True]),
        ),
        (
            lambda: _array(
                [
                    [True, False, False, True],
                    [False, True, False, False],
                    [True, False, True, False],
                ],
                bool,
            ),
            lambda: np.array(
                [
                    [True, False, False, True],
                    [False, True, False, False],
                    [True, False, True, False],
                ]
            ),
        ),
        (
            lambda: (
                slice(None),
                _array([[True, False, True, False, False]] * 4, bool),
            ),
            lambda: (
                slice(None),
                np.array([[True, False, True, False, False]] * 4),
            ),
        ),
    ],
    ids=["axis-mask", "multi-axis-mask", "mixed-mask"],
)
def test_boolean_indexing_matches_numpy(sdnp_index, numpy_index):
    actual, expected = _sample()
    assert_matches(actual[sdnp_index()], expected[numpy_index()])


@pytest.mark.parametrize(
    ("sdnp_index", "numpy_index"),
    [
        (
            lambda: _array([2, 0, -1]),
            lambda: np.array([2, 0, -1]),
        ),
        (
            lambda: (_array([2, 0]), _array([3, 1])),
            lambda: (np.array([2, 0]), np.array([3, 1])),
        ),
        (
            lambda: (_array([[0], [2]]), slice(None), _array([[1, 4, 0]])),
            lambda: (
                np.array([[0], [2]]),
                slice(None),
                np.array([[1, 4, 0]]),
            ),
        ),
        (
            lambda: (slice(None), _array([3, 1]), _array([4, 0])),
            lambda: (slice(None), np.array([3, 1]), np.array([4, 0])),
        ),
    ],
    ids=[
        "single-fancy",
        "adjacent-fancy",
        "broadcast-separated-fancy",
        "mixed-slice-fancy",
    ],
)
def test_fancy_and_mixed_indexing_matches_numpy(sdnp_index, numpy_index):
    actual, expected = _sample()
    assert_matches(actual[sdnp_index()], expected[numpy_index()])


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_fully_integer_index_unwraps_python_scalar(dtype):
    values = np.asarray([[0, 1], [2, 3]], dtype=NP_DTYPES[dtype])
    actual = _array(values, dtype)[-1, -1]

    assert not isinstance(actual, sdnp.Array)
    assert_matches(actual, values[-1, -1])


@pytest.mark.parametrize(
    ("index", "value"),
    [
        ((slice(None), 1, slice(None, None, 2)), -7),
        ((Ellipsis, -1), 99),
        ((_array([2, 0]), slice(None), 1), -3),
        (_array([True, False, True], bool), 42),
    ],
    ids=["basic", "ellipsis", "fancy", "boolean"],
)
def test_scalar_assignment_matches_numpy(index, value):
    actual, expected = _sample()
    actual[index] = value
    numpy_index = _numpy_index(index)
    expected[numpy_index] = value
    assert_matches(actual, expected)


@pytest.mark.parametrize(
    ("index", "values"),
    [
        ((1, slice(None), slice(None)), np.arange(20).reshape(4, 5) + 100),
        (
            (slice(None), slice(1, 3), slice(None)),
            np.arange(10).reshape(1, 2, 5),
        ),
        (
            (slice(None), slice(None), slice(None, None, 2)),
            -np.arange(1, 5).reshape(4, 1),
        ),
        ((_array([2, 0]), slice(None), 1), np.arange(8).reshape(2, 4)),
    ],
    ids=["exact-shape", "leading-broadcast", "trailing-broadcast", "fancy"],
)
def test_array_and_broadcast_assignment_match_numpy(index, values):
    actual, expected = _sample()
    actual[index] = _array(values)
    expected[_numpy_index(index)] = values
    assert_matches(actual, expected)


def test_overlapping_assignment_copies_source_before_write():
    actual = _array(np.arange(12).reshape(3, 4))
    expected = np.arange(12).reshape(3, 4)

    actual[:, 1:] = actual[:, :-1]
    expected[:, 1:] = expected[:, :-1]

    assert_matches(actual, expected)


def test_basic_view_write_uses_copy_on_write():
    original, expected_original = _sample()
    view = original[::-1, :, ::-2]
    expected_view = expected_original[::-1, :, ::-2].copy()

    view[...] = -1
    expected_view[...] = -1

    assert_matches(view, expected_view)
    assert_matches(original, expected_original)


def test_fancy_result_is_an_independent_copy():
    original, expected_original = _sample()
    selected = original[_array([2, 0])]
    expected_selected = expected_original[[2, 0]].copy()

    selected[...] = -1
    expected_selected[...] = -1

    assert_matches(selected, expected_selected)
    assert_matches(original, expected_original)


def test_read_only_meshgrid_rejects_assignment():
    grid, _ = sdnp.meshgrid(sdnp.arange(3), sdnp.arange(2))

    with pytest.raises(ValueError, match="read-only"):
        grid[...] = 0


@pytest.mark.parametrize(
    ("index", "error"),
    [
        (3, IndexError),
        (-4, IndexError),
        ((0, 0, 0, 0), IndexError),
        ((Ellipsis, Ellipsis), IndexError),
        ("bad", TypeError),
        (1.5, TypeError),
        (slice(None, None, 0), ValueError),
    ],
    ids=[
        "positive-oob",
        "negative-oob",
        "too-many",
        "duplicate-ellipsis",
        "string",
        "float",
        "zero-step",
    ],
)
def test_invalid_basic_indices_raise(index, error):
    array, _ = _sample()
    with pytest.raises(error):
        _ = array[index]


def test_invalid_fancy_index_dtype_raises():
    array, _ = _sample()
    with pytest.raises(TypeError, match="integer or boolean"):
        _ = array[_array([0.0, 1.0], float)]


def test_boolean_index_shape_mismatch_raises():
    array, _ = _sample()
    with pytest.raises(IndexError, match="boolean index shape"):
        _ = array[_array([True, False], bool)]


@pytest.mark.parametrize(
    "values",
    [
        np.ones((2, 2), dtype=np.int64),
        np.ones((6,), dtype=np.int64),
    ],
)
def test_assignment_shape_mismatch_raises(values):
    array, _ = _sample()
    with pytest.raises(ValueError):
        array[:, 1:3, :] = _array(values)


def test_invalid_scalar_assignment_dtype_raises():
    array = _array([True, False], bool)
    with pytest.raises(ValueError, match="bool array"):
        array[:] = 1 + 2j


def _numpy_index(index):
    if isinstance(index, sdnp.Array):
        return np.asarray(index.to_list(), dtype=NP_DTYPES[index.dtype])
    if isinstance(index, tuple):
        return tuple(_numpy_index(item) for item in index)
    return index
