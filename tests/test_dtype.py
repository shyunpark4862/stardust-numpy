"""Dtype inference, casting, homogeneous array rules, and operation validation."""

from __future__ import annotations

import pytest

from numpy import *  # noqa: F403


def test_array_infers_int_dtype():
    a = array([1, 2, 3])

    assert a.dtype is int
    assert a.tolist() == [1, 2, 3]


def test_array_infers_float_dtype():
    a = array([1.0, 2.5, 3.0])

    assert a.dtype is float
    assert a.tolist() == [1.0, 2.5, 3.0]


def test_array_infers_complex_dtype():
    a = array([1 + 2j, 3j])

    assert a.dtype is complex
    assert a.tolist() == [1 + 2j, 3j]


def test_array_promotes_numeric_to_complex():
    a = array([1, 2.0, 3 + 0j])

    assert a.dtype is complex
    assert a.tolist() == [1 + 0j, 2 + 0j, 3 + 0j]


def test_array_promotes_int_and_float_to_float():
    a = array([1, 2.0, 3])

    assert a.dtype is float
    assert a.tolist() == [1.0, 2.0, 3.0]


def test_array_promotes_bool_and_int_to_int():
    a = array([True, 2, 3])

    assert a.dtype is int
    assert a.tolist() == [1, 2, 3]


def test_array_rejects_mixed_incompatible_dtypes():
    with pytest.raises(ValueError):
        array([1, "hello"])


def test_array_accepts_builtin_type_as_dtype():
    a = array([1, 2, 3], dtype=float)

    assert a.dtype is float
    assert a.tolist() == [1.0, 2.0, 3.0]

    assert array([1, 2], dtype=int).dtype is int
    assert array([True, False], dtype=bool).dtype is bool
    assert array(["a"], dtype=str).dtype is str
    assert array([1 + 2j], dtype=complex).dtype is complex


def test_array_accepts_explicit_dtype():
    a = array([1.9, 2.1, 3.8], dtype=int)

    assert a.dtype is int
    assert a.tolist() == [1, 2, 3]


def test_full_uses_fill_value_dtype():
    a = full(3, "x")

    assert a.dtype is str
    assert a.tolist() == ["x", "x", "x"]


def test_zeros_and_ones_default_to_int():
    assert zeros(3).dtype is int
    assert ones(3).dtype is int


def test_empty_array_defaults_to_float():
    assert array([]).dtype is float


def test_astype_casts_to_target_dtype():
    a = array([1, 2, 3])

    assert a.astype(float).dtype is float
    assert a.astype(float).tolist() == [1.0, 2.0, 3.0]
    assert a.astype(bool).tolist() == [True, True, True]
    assert a.astype(complex).tolist() == [1 + 0j, 2 + 0j, 3 + 0j]


def test_astype_rejects_invalid_cast():
    a = array(["a", "b"])

    with pytest.raises(TypeError):
        a.astype(int)

    c = array([1 + 2j, 3j])

    with pytest.raises(TypeError):
        c.astype(int)


def test_comparison_results_are_bool_arrays():
    a = array([1, 2, 3])
    result = a > 1

    assert result.dtype is bool
    assert result.tolist() == [False, True, True]


def test_arange_respects_explicit_dtype():
    assert arange(3, dtype=float).dtype is float
    assert arange(3, dtype=float).tolist() == [0.0, 1.0, 2.0]


def test_concatenate_promotes_dtypes():
    left = array([1, 2], dtype=int)
    right = array([3.0, 4.0], dtype=float)

    result = concatenate([left, right])

    assert result.dtype is float
    assert result.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_complex_arithmetic_and_division():
    a = array([1 + 2j, 3j])

    assert (a + 1).dtype is complex
    assert (a / 2).dtype is complex
    assert (a / 2).tolist() == [0.5 + 1j, 1.5j]


def test_complex_sqrt_uses_complex_result():
    a = array([-1 + 0j])
    result = sqrt(a)

    assert result.dtype is complex
    assert result.tolist() == [1j]


def test_sum_rejects_string_dtype():
    a = array(["a", "b"])

    with pytest.raises(
        TypeError, match="operation 'sum' is not supported for dtype str"
    ):
        a.sum()


def test_mean_rejects_string_dtype():
    a = array(["a", "b"])

    with pytest.raises(
        TypeError, match="operation 'mean' is not supported for dtype str"
    ):
        a.mean()


def test_min_and_sort_reject_complex_dtype():
    a = array([1 + 2j, 3j])

    with pytest.raises(
        TypeError, match="operation 'min' is not supported for dtype complex"
    ):
        a.min()

    with pytest.raises(
        TypeError, match="operation 'sort' is not supported for dtype complex"
    ):
        a.sort()


def test_dot_rejects_string_dtype():
    a = array(["a", "b"])
    b = array(["c", "d"])

    with pytest.raises(
        TypeError, match="operation 'dot' is not supported for dtype str"
    ):
        a.dot(b)


def test_logical_and_rejects_string_dtype():
    a = array(["a", "b"])

    with pytest.raises(
        TypeError, match="operation 'logical' is not supported for dtype str"
    ):
        a & True


def test_arithmetic_rejects_string_dtype():
    a = array(["a", "b"])

    with pytest.raises(
        TypeError, match="operation 'arithmetic' is not supported for dtype str"
    ):
        a + 1


def test_comparison_rejects_string_and_int_mix():
    a = array(["a", "b"])

    with pytest.raises(
        TypeError, match="operation 'comparison' does not support operands"
    ):
        a > 1


def test_as_dtype_accepts_string_aliases():
    from numpy._dtype import as_dtype

    assert as_dtype("float") is float
    assert as_dtype("INT") is int
    assert as_dtype("complex") is complex
    assert as_dtype("str") is str


def test_as_dtype_rejects_invalid_values():
    from numpy._dtype import as_dtype

    with pytest.raises(TypeError, match="is not a supported dtype"):
        as_dtype("not_a_dtype")

    with pytest.raises(TypeError, match="is not a supported dtype"):
        as_dtype(object)


def test_can_cast_covers_promotion_rules():
    from numpy._dtype import can_cast

    assert can_cast(int, float)
    assert can_cast(bool, int)
    assert can_cast(float, complex)
    assert can_cast(int, str)
    assert not can_cast(float, int)
    assert not can_cast(int, bool)


def test_can_cast_same_dtype_and_unknown_target():
    from numpy._dtype import can_cast

    assert can_cast(int, int) is True
    assert can_cast(int, list) is False
    assert can_cast(list, int) is False


def test_cast_value_supports_str_and_rejects_invalid_casts():
    from numpy._dtype import cast_value

    assert cast_value(str, 123) == "123"
    assert cast_value(complex, True) == 1 + 0j
    assert cast_value(float, True) == 1.0

    with pytest.raises(TypeError, match="cannot cast"):
        cast_value(int, "x")

    with pytest.raises(TypeError, match="cannot cast"):
        cast_value(float, 1 + 2j)

    with pytest.raises(TypeError, match="cannot cast"):
        cast_value(complex, "x")


def test_cast_value_rejects_unsupported_dtype():
    from numpy._dtype import cast_value

    with pytest.raises(TypeError, match="unsupported dtype"):
        cast_value(list, 1)


def test_dtype_helpers_and_defaults():
    from numpy._dtype import (
        default_fill_value,
        infer_dtype_from_value,
        is_float_dtype,
        is_str_dtype,
        require_orderable_dtype,
        result_dtype_for_mean,
    )

    assert is_float_dtype(float)
    assert is_str_dtype(str)
    assert result_dtype_for_mean(complex) is complex
    assert default_fill_value(str) == ""
    assert default_fill_value(bool) is False

    with pytest.raises(TypeError):
        infer_dtype_from_value([])

    with pytest.raises(TypeError):
        default_fill_value(list)

    with pytest.raises(TypeError):
        require_orderable_dtype(complex, operation="sort")


def test_empty_array_infers_float_dtype():
    assert array([]).dtype is float


def test_infer_dtype_from_values_empty_sequence():
    from numpy._dtype import infer_dtype_from_values

    assert infer_dtype_from_values([]) is float


def test_contiguous_ndarray_infers_dtype_when_none():
    from numpy._factory import contiguous_ndarray

    result = contiguous_ndarray([1, 2, 3], (3,))

    assert result.dtype is int
    assert result.tolist() == [1, 2, 3]


def test_astype_same_dtype_returns_a_copy():
    original = array([1, 2, 3])
    copied = original.astype(int)

    assert copied is not original
    assert copied.tolist() == original.tolist()
