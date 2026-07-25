"""Iteration protocols and stable human-readable array formatting."""

import pytest
import sdnp


def exhaust(iterator):
    values = list(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    return values


def test_array_axis0_iteration_returns_rows_and_then_exhausts():
    array = sdnp.arange(6).reshape((2, 3))
    iterator = iter(array)

    assert iter(iterator) is iterator
    first = next(iterator)
    second = next(iterator)
    assert isinstance(first, sdnp.Array)
    assert first.to_list() == [0, 1, 2]
    assert second.to_list() == [3, 4, 5]
    assert exhaust(iterator) == []


def test_one_dimensional_axis0_iteration_unwraps_items_to_scalars():
    array = sdnp.array([True, False, True])

    values = list(array)

    assert values == [True, False, True]
    assert all(not isinstance(value, sdnp.Array) for value in values)
    assert len(array) == 3


def test_axis0_iteration_of_empty_array_is_empty():
    array = sdnp.zeros((0, 3))

    assert len(array) == 0
    assert exhaust(iter(array)) == []


def test_flat_iteration_uses_logical_c_order_for_noncontiguous_views():
    view = sdnp.arange(12).reshape((3, 4))[::-1, ::2].T
    iterator = view.flat

    assert iter(iterator) is iterator
    assert exhaust(iterator) == [8, 4, 0, 10, 6, 2]


def test_flat_iteration_preserves_scalar_types_and_handles_empty_arrays():
    assert list(sdnp.array([True, False]).flat) == [True, False]
    assert list(sdnp.array([1.5, -2.0]).flat) == [1.5, -2.0]
    assert list(sdnp.array([1 + 2j, 3 - 4j]).flat) == [1 + 2j, 3 - 4j]
    assert exhaust(sdnp.zeros((2, 0)).flat) == []


def test_ndindex_len_tracks_remaining_items_and_exhaustion():
    iterator = sdnp.ndindex((2, 3))

    assert iter(iterator) is iterator
    assert len(iterator) == 6
    assert next(iterator) == (0, 0)
    assert len(iterator) == 5
    assert exhaust(iterator) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert len(iterator) == 0


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3), (0, 0)])
def test_ndindex_empty_shapes_have_zero_length(shape):
    iterator = sdnp.ndindex(shape)

    assert len(iterator) == 0
    assert exhaust(iterator) == []


def test_ndindex_accepts_single_integer_shape():
    assert list(sdnp.ndindex(3)) == [(0,), (1,), (2,)]


def test_ndenumerate_noncontiguous_order_and_exhaustion():
    view = sdnp.arange(6).reshape((2, 3))[:, ::-1]
    iterator = sdnp.ndenumerate(view)

    assert iter(iterator) is iterator
    assert next(iterator) == ((0, 0), 2)
    assert exhaust(iterator) == [
        ((0, 1), 1),
        ((0, 2), 0),
        ((1, 0), 5),
        ((1, 1), 4),
        ((1, 2), 3),
    ]
    with pytest.raises(TypeError):
        len(iterator)


def test_ndenumerate_empty_array_is_empty():
    assert exhaust(sdnp.ndenumerate(sdnp.zeros((2, 0)))) == []


def test_nditer_single_operand_noncontiguous_and_empty():
    iterator = sdnp.nditer((sdnp.arange(6).reshape((2, 3))[:, ::-1],))

    assert iter(iterator) is iterator
    assert exhaust(iterator) == [2, 1, 0, 5, 4, 3]
    assert exhaust(sdnp.nditer((sdnp.zeros((0, 2)),))) == []
    with pytest.raises(TypeError):
        len(iterator)


def test_nditer_broadcasts_two_operands_in_logical_order():
    left = sdnp.array([[1], [2]])
    right = sdnp.array([[10, 20, 30]])

    assert list(sdnp.nditer((left, right))) == [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 20),
        (2, 30),
    ]


def test_repr_and_str_include_shape_values_and_dtype():
    array = sdnp.arange(6).reshape((2, 3))
    expected = "array([[0, 1, 2],[3, 4, 5]], dtype=int64)"

    assert repr(array) == expected
    assert str(array) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([True, False], "array([True, False], dtype=bool)"),
        (
            [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
            "array([0, -0, nan, inf, -inf], dtype=float64)",
        ),
        (
            [1 + 2j, 1 - 2j, complex(float("nan"), 0)],
            "array([(1+2j), (1-2j), (nan+0j)], dtype=complex128)",
        ),
    ],
)
def test_repr_special_values(values, expected):
    array = sdnp.array(values)

    assert repr(array) == expected
    assert str(array) == expected


def test_repr_abbreviates_large_one_dimensional_arrays():
    text = repr(sdnp.arange(1001))

    assert text == "array([0, 1, 2, ..., 998, 999, 1000], dtype=int64)"
