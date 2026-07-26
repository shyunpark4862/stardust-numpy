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


def assert_format_width(text):
    assert text
    assert all(len(line) <= 80 for line in text.splitlines())


@pytest.mark.parametrize(
    ("values", "dtype", "data", "dtype_name", "display"),
    [
        (
            [True, False],
            bool,
            "[True, False]",
            "bool",
            "[0] True False",
        ),
        ([1, -2], int, "[1, -2]", "int64", "[0] 1 -2"),
        (
            [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
            float,
            "[0, -0, nan, inf, -inf]",
            "float64",
            "[0] 0 -0 nan inf -inf",
        ),
        (
            [1 + 2j, 1 - 2j, 3 + 0j, complex(float("nan"), 0)],
            complex,
            "[1+2j, 1-2j, 3+0j, nan+0j]",
            "complex128",
            "[0] 1+2j 1-2j 3+0j nan+0j",
        ),
    ],
)
def test_repr_and_str_scalar_formatting(
    values, dtype, data, dtype_name, display
):
    array = sdnp.array(values, dtype=dtype)

    assert repr(array).splitlines() == [
        f"sdnp-array at 0x{id(array):x}",
        f"  @ data: {data}",
        f"  @ shape: [{len(values)}]",
        "  @ ndim: 1",
        f"  @ size: {len(values)}",
        f"  @ dtype: {dtype_name}",
    ]
    assert str(array) == display
    assert "(" not in data
    assert_format_width(repr(array))
    assert_format_width(str(array))


def test_str_formats_two_dimensional_matrix_with_zero_based_labels():
    array = sdnp.arange(6).reshape((2, 3))

    assert str(array) == (
        "     [,0] [,1] [,2]\n"
        "[0,]    0    1    2\n"
        "[1,]    3    4    5"
    )


def test_str_paginates_three_dimensions_by_leading_axis():
    array = sdnp.arange(24).reshape((2, 3, 4))

    assert str(array) == (
        "[0, ,]\n"
        "     [,0] [,1] [,2] [,3]\n"
        "[0,]    0    1    2    3\n"
        "[1,]    4    5    6    7\n"
        "[2,]    8    9   10   11\n\n"
        "[1, ,]\n"
        "     [,0] [,1] [,2] [,3]\n"
        "[0,]   12   13   14   15\n"
        "[1,]   16   17   18   19\n"
        "[2,]   20   21   22   23"
    )


def test_str_paginates_four_dimensions_in_leading_axis_order():
    array = sdnp.arange(16).reshape((2, 2, 2, 2))
    labels = [
        line
        for line in str(array).splitlines()
        if line.startswith("[") and line.endswith(", ,]")
    ]

    assert labels == ["[0, 0, ,]", "[0, 1, ,]", "[1, 0, ,]", "[1, 1, ,]"]


def test_repr_and_str_follow_logical_order_for_non_contiguous_view():
    array = sdnp.arange(6).reshape((2, 3)).T

    assert repr(array).splitlines()[1] == "  @ data: [0, 3, 1, 4, 2, 5]"
    assert str(array) == (
        "     [,0] [,1]\n"
        "[0,]    0    3\n"
        "[1,]    1    4\n"
        "[2,]    2    5"
    )


def test_repr_and_vector_str_abbreviate_both_edges_within_eighty_columns():
    array = sdnp.arange(1001)
    repr_text = repr(array)
    str_text = str(array)

    assert repr_text.splitlines()[1].startswith("  @ data: [0, 1, 2")
    assert "..." in repr_text.splitlines()[1]
    assert repr_text.splitlines()[1].endswith("1000]")
    assert str_text.startswith("[0] 0 1 2")
    assert "..." in str_text
    assert str_text.endswith("1000")
    assert_format_width(repr_text)
    assert_format_width(str_text)


def test_str_abbreviates_large_matrix_rows_and_columns():
    text = str(sdnp.arange(400).reshape((20, 20)))
    lines = text.splitlines()

    assert lines[0].startswith("      [,0] [,1]")
    assert " ... " in lines[0]
    assert lines[1].lstrip().startswith("[0,]")
    assert lines[3].lstrip().startswith("[2,]")
    assert lines[4] == "..."
    assert lines[-1].startswith("[19,]")
    assert_format_width(text)


def test_str_abbreviates_many_pages_at_both_edges():
    text = str(sdnp.arange(40).reshape((10, 2, 2)))
    blocks = text.split("\n\n")

    assert [block.splitlines()[0] for block in blocks] == [
        "[0, ,]",
        "[1, ,]",
        "[2, ,]",
        "...",
        "[7, ,]",
        "[8, ,]",
        "[9, ,]",
    ]
    assert_format_width(text)


def test_long_shape_and_long_complex_values_remain_width_bounded():
    max_float = float.fromhex("0x1.fffffffffffffp+1023")
    arrays = [
        sdnp.ones((1,) * 64),
        sdnp.full((10, 10), complex(max_float, -max_float)),
    ]

    for array in arrays:
        assert_format_width(repr(array))
        assert_format_width(str(array))

    assert "..." in repr(arrays[0]).splitlines()[2]
    assert "(" not in repr(arrays[1])
    assert "(" not in str(arrays[1])


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(sdnp.array([], dtype=int), id="empty-vector"),
        pytest.param(sdnp.zeros((2, 0)), id="zero-columns"),
        pytest.param(sdnp.zeros((0, 2)), id="zero-rows"),
        pytest.param(sdnp.zeros((2, 0, 3)), id="empty-page"),
    ],
)
def test_empty_array_formats_are_stable_and_width_bounded(array):
    assert "  @ data: []" in repr(array)
    assert_format_width(repr(array))
    assert all(len(line) <= 80 for line in str(array).splitlines())
