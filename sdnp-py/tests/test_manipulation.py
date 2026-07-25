import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, assert_matches


def _values(shape, dtype=int, *, start=0):
    size = int(np.prod(shape, dtype=np.int64))
    values = np.arange(start, start + size, dtype=np.int64).reshape(shape)
    if dtype is bool:
        return (values % 3 == 0).astype(np.bool_)
    return values.astype(NP_DTYPES[dtype])


def _array(values, dtype=int):
    values = np.asarray(values, dtype=NP_DTYPES[dtype])
    if values.size == 0:
        return sdnp.zeros(values.shape, dtype=dtype)
    return sdnp.array(values.tolist(), dtype=dtype)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_concatenate_across_axes_matches_numpy(dtype, axis):
    left = _values((2, 3, 4), dtype)
    right_shape = list(left.shape)
    right_shape[axis % left.ndim] = 2
    right = _values(tuple(right_shape), dtype, start=100)

    actual = sdnp.concatenate(
        [_array(left, dtype), _array(right, dtype)], axis=axis
    )
    expected = np.concatenate([left, right], axis=axis)

    assert_matches(actual, expected)


def test_concatenate_defaults_to_axis_zero_and_accepts_tuple():
    left = np.arange(6).reshape(2, 3)
    right = np.arange(9).reshape(3, 3) + 20

    actual = sdnp.concatenate((_array(left), _array(right)))

    assert_matches(actual, np.concatenate((left, right)))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concatenate_noncontiguous_inputs(axis):
    base = np.arange(4 * 6 * 8).reshape(4, 6, 8)
    left_np = base[::-1, ::2, 1::2]
    right_np = (base + 1000)[::-1, ::2, 1::2]
    left = _array(base)[::-1, ::2, 1::2]
    right = _array(base + 1000)[::-1, ::2, 1::2]

    assert_matches(
        sdnp.concatenate([left, right], axis=axis),
        np.concatenate([left_np, right_np], axis=axis),
    )


def test_concatenate_transposed_inputs():
    left_np = np.arange(12).reshape(3, 4).T
    right_np = (np.arange(6).reshape(3, 2) + 50).T
    left = _array(np.arange(12).reshape(3, 4)).T
    right = _array(np.arange(6).reshape(3, 2) + 50).T

    assert_matches(
        sdnp.concatenate([left, right], axis=0),
        np.concatenate([left_np, right_np], axis=0),
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_concatenate_empty_dimensions(axis):
    shape = [2, 3, 4]
    shape[axis] = 0
    empty = np.empty(shape, dtype=np.int64)
    full_shape = shape.copy()
    full_shape[axis] = 2
    full = _values(tuple(full_shape))

    assert_matches(
        sdnp.concatenate([_array(empty), _array(full)], axis=axis),
        np.concatenate([empty, full], axis=axis),
    )
    assert_matches(
        sdnp.concatenate([_array(empty), _array(empty)], axis=axis),
        np.concatenate([empty, empty], axis=axis),
    )


@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -4])
def test_stack_across_every_insert_axis(dtype, axis):
    first = _values((2, 3, 4), dtype)
    second = _values((2, 3, 4), dtype, start=100)

    assert_matches(
        sdnp.stack([_array(first, dtype), _array(second, dtype)], axis=axis),
        np.stack([first, second], axis=axis),
    )


def test_stack_default_axis_and_single_input():
    values = np.arange(6).reshape(2, 3)
    assert_matches(sdnp.stack([_array(values)]), np.stack([values]))


def test_stack_empty_and_noncontiguous_inputs():
    empty = np.empty((2, 0, 3), dtype=np.int64)
    assert_matches(
        sdnp.stack([_array(empty), _array(empty)], axis=2),
        np.stack([empty, empty], axis=2),
    )

    base = np.arange(30).reshape(5, 6)
    first_np = base[::-2, ::-2]
    second_np = (base + 100)[::-2, ::-2]
    first = _array(base)[::-2, ::-2]
    second = _array(base + 100)[::-2, ::-2]
    assert_matches(
        sdnp.stack([first, second], axis=-1),
        np.stack([first_np, second_np], axis=-1),
    )


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_vstack_promotes_vectors_to_rows(dtype):
    first = _values((4,), dtype)
    second = _values((4,), dtype, start=10)

    assert_matches(
        sdnp.vstack([_array(first, dtype), _array(second, dtype)]),
        np.vstack([first, second]),
    )


def test_vstack_combines_vector_and_matrix_after_rank_promotion():
    vector = np.arange(4)
    matrix = np.arange(8).reshape(2, 4) + 10
    assert_matches(
        sdnp.vstack([_array(vector), _array(matrix)]),
        np.vstack([vector, matrix]),
    )


def test_vstack_handles_empty_vectors_and_noncontiguous_rows():
    empty = np.empty((0,), dtype=np.int64)
    assert_matches(
        sdnp.vstack([_array(empty), _array(empty)]),
        np.vstack([empty, empty]),
    )

    base = np.arange(20).reshape(4, 5)
    first_np = base[::2, ::-1]
    second_np = (base + 100)[::2, ::-1]
    assert_matches(
        sdnp.vstack([_array(base)[::2, ::-1], _array(base + 100)[::2, ::-1]]),
        np.vstack([first_np, second_np]),
    )


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_hstack_keeps_vectors_one_dimensional(dtype):
    first = _values((3,), dtype)
    second = _values((2,), dtype, start=10)

    assert_matches(
        sdnp.hstack([_array(first, dtype), _array(second, dtype)]),
        np.hstack([first, second]),
    )


def test_hstack_joins_matrices_on_second_axis():
    left = np.arange(6).reshape(3, 2)
    right = np.arange(12).reshape(3, 4) + 20
    assert_matches(
        sdnp.hstack([_array(left), _array(right)]),
        np.hstack([left, right]),
    )


def test_hstack_handles_empty_and_noncontiguous_inputs():
    empty = np.empty((3, 0), dtype=np.int64)
    full = np.arange(6).reshape(3, 2)
    assert_matches(
        sdnp.hstack([_array(empty), _array(full)]),
        np.hstack([empty, full]),
    )

    base = np.arange(24).reshape(4, 6)
    left_np = base[::-1, ::2]
    right_np = (base + 100)[::-1, 1::2]
    assert_matches(
        sdnp.hstack([_array(base)[::-1, ::2], _array(base + 100)[::-1, 1::2]]),
        np.hstack([left_np, right_np]),
    )


@pytest.mark.parametrize(
    "operation",
    [sdnp.concatenate, sdnp.stack, sdnp.vstack, sdnp.hstack],
    ids=["concatenate", "stack", "vstack", "hstack"],
)
def test_join_operations_reject_empty_input(operation):
    with pytest.raises(ValueError, match="at least one array"):
        operation([])


@pytest.mark.parametrize(
    "operation",
    [sdnp.concatenate, sdnp.stack, sdnp.vstack, sdnp.hstack],
    ids=["concatenate", "stack", "vstack", "hstack"],
)
def test_join_operations_reject_non_array_elements(operation):
    with pytest.raises(TypeError, match="sdnp.Array"):
        operation([_array([1, 2]), [3, 4]])


@pytest.mark.parametrize(
    "operation",
    [sdnp.concatenate, sdnp.stack, sdnp.vstack, sdnp.hstack],
    ids=["concatenate", "stack", "vstack", "hstack"],
)
def test_join_operations_reject_mixed_dtypes(operation):
    with pytest.raises(ValueError, match="same dtype|dtype mismatch"):
        operation([_array([1, 2], int), _array([3.0, 4.0], float)])


@pytest.mark.parametrize("axis", [2, -3])
def test_concatenate_rejects_out_of_bounds_axis(axis):
    arrays = [_array(np.ones((2, 3), dtype=np.int64))] * 2
    with pytest.raises(IndexError, match="axis"):
        sdnp.concatenate(arrays, axis=axis)


@pytest.mark.parametrize("axis", [3, -4])
def test_stack_rejects_out_of_bounds_axis(axis):
    arrays = [_array(np.ones((2, 3), dtype=np.int64))] * 2
    with pytest.raises(IndexError, match="axis"):
        sdnp.stack(arrays, axis=axis)


def test_axis_must_be_an_integer():
    arrays = [_array(np.ones((2, 3), dtype=np.int64))] * 2
    with pytest.raises(TypeError):
        sdnp.concatenate(arrays, axis=1.5)
    with pytest.raises(TypeError):
        sdnp.stack(arrays, axis="0")


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (sdnp.concatenate, np.ones((2, 3)), np.ones((2, 4))),
        (sdnp.concatenate, np.ones((2, 3)), np.ones((2, 3, 1))),
        (sdnp.stack, np.ones((2, 3)), np.ones((2, 4))),
        (sdnp.vstack, np.ones((2, 3)), np.ones((1, 4))),
        (sdnp.hstack, np.ones((2, 3)), np.ones((3, 2))),
        (sdnp.hstack, np.ones((3,)), np.ones((1, 3))),
    ],
    ids=[
        "concatenate-dimension",
        "concatenate-rank",
        "stack-shape",
        "vstack-width",
        "hstack-height",
        "hstack-rank",
    ],
)
def test_join_operations_reject_incompatible_shapes(operation, left, right):
    with pytest.raises(ValueError, match="shape|rank|dimensions"):
        operation([_array(left, float), _array(right, float)])
