import pytest

from numpy import arange, array, full, zeros


def test_repr_scalar():
    scalar = full((), 42)

    assert repr(scalar) == "array(shape=(), dtype=int, value=42)"


def test_str_scalar():
    scalar = full((), 42)

    assert str(scalar) == "42"


def test_repr_empty():
    empty = array([])

    assert repr(empty) == "array(shape=(0,), dtype=float)"


def test_str_empty():
    empty = array([])

    assert str(empty) == "[1]"


def test_repr_small_1d():
    values = array([1, 2, 3])

    assert repr(values) == "array(shape=(3,), dtype=int, data=[1, 2, 3])"


def test_str_small_1d():
    values = array([1, 20, 300])

    assert str(values) == "[1] 1 20 300"


def test_repr_small_2d():
    matrix = array([[1, 2, 3], [4, 5, 6]])

    assert (
        repr(matrix)
        == "array(shape=(2, 3), dtype=int, data=[[1, 2, 3], [4, 5, 6]])"
    )


def test_str_small_2d():
    matrix = array([[1, 20, 300], [4, 5, 6]])

    assert str(matrix) == (
        "     [,1] [,2] [,3]\n[1,]    1   20  300\n[2,]    4    5    6"
    )


def test_repr_large_uses_preview():
    values = arange(20)

    assert repr(values).startswith("array(shape=(20,), dtype=int, preview=[")


def test_str_uses_ellipsis_when_line_exceeds_width():
    values = arange(2000)

    text = str(values)

    assert text.startswith("[1]")
    assert "..." in text


def test_str_3d_uses_last_two_dims_as_matrix():
    tensor = arange(40).reshape(2, 2, 10)
    text = str(tensor)

    assert "1, ," in text
    assert "2, ," in text
    assert "[1,]" in text
    assert "[,10]" in text


def test_str_3d_blocks_are_separated():
    tensor = arange(40).reshape(2, 2, 10)
    text = str(tensor)

    assert "\n\n" in text


def test_str_line_width_is_at_most_80():
    tensor = arange(2000).reshape(10, 10, 20)

    for line in str(tensor).splitlines():
        assert len(line) <= 80


def test_repr_and_str_differ_for_matrix():
    matrix = array([[1, 2], [3, 4]])

    assert repr(matrix) != str(matrix)


def test_str_string_array():
    values = array(["a", "bb"])

    assert str(values) == "[1] 'a' 'bb'"


def test_repr_string_array():
    values = array(["a", "bb"])

    assert repr(values) == "array(shape=(2,), dtype=str, data=['a', 'bb'])"


def test_str_float_array():
    values = array([1.0, 2.5])

    assert str(values) == "[1] 1. 2.5"


def test_str_complex_array():
    values = array([1 + 2j, 3j])

    assert str(values) == "[1] (1.+2.j) (0.+3.j)"


def test_str_bool_array():
    values = array([True, False])

    assert str(values) == "[1] True False"


def test_repr_empty_2d():
    empty = zeros((0, 3))

    assert repr(empty) == "array(shape=(0, 3), dtype=int)"


def test_str_empty_2d_with_rows():
    empty = zeros((2, 0))

    assert str(empty) == "\n"


def test_str_long_values_stay_within_line_width():
    cases = [
        array([10**75]),
        array([["x" * 100]]),
        array([[10**50]]),
        array([1 + 2e20j, 3e-15 + 4e15j, 5e200 + 6e-200j]),
        array([[1e-30, 1e30], [1e300, 1e-300]]),
        array([[10**15, 10**16], [10**17, 10**18]]),
    ]

    for values in cases:
        for line in str(values).splitlines():
            assert len(line) <= 80


def test_str_row_label_is_separated_from_first_cell():
    import re

    matrix = array([[10**17, 10**18], [10**19, 10**20]])
    glued = [
        line
        for line in str(matrix).splitlines()
        if re.search(r"\[\d+,\]\S", line)
    ]

    assert glued == []


def test_str_long_string_is_truncated():
    text = str(array([["x" * 100]]))

    assert "..." in text
    assert all(len(line) <= 80 for line in text.splitlines())


def test_str_many_rows_are_summarized():
    values = arange(500).reshape(100, 5)
    lines = str(values).splitlines()

    assert len(lines) <= 10
    assert any("..." in line for line in lines)
    assert all(len(line) <= 80 for line in lines)


def test_str_many_columns_are_summarized():
    values = arange(500).reshape(5, 100)
    text = str(values)

    assert "..." in text
    assert "[,100]" in text
    assert all(len(line) <= 80 for line in text.splitlines())


def test_str_many_outer_blocks_are_summarized():
    values = arange(5000).reshape(50, 5, 20)
    lines = str(values).splitlines()

    assert len(lines) < 200
    assert "1, ," in lines[0]
    assert any(line == "..." for line in lines)
    assert any(line.startswith("50, ,") for line in lines)
    assert all(len(line) <= 80 for line in lines)


def test_str_huge_4d_tensor_stays_compact():
    values = arange(10000).reshape(10, 10, 10, 10)
    lines = str(values).splitlines()

    assert len(lines) < 200
    assert all(len(line) <= 80 for line in lines)
