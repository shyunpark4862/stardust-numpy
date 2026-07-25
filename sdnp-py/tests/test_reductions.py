import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, as_numpy, assert_matches, numpy_call

DTYPES = (bool, int, float, complex)
NUMERIC_REDUCTIONS = ("sum", "prod", "min", "max", "mean", "var", "std")


def array(values, dtype):
    return sdnp.array(
        np.asarray(values, dtype=NP_DTYPES[dtype]).tolist(), dtype=dtype
    )


def numpy_reduction(
    name, values, *, axis=None, keepdims=False, nan_policy="propagate"
):
    prefix = "nan" if nan_policy == "ignore" else ""
    func = getattr(np, f"{prefix}{name}")
    return numpy_call(func, values, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize("name", NUMERIC_REDUCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_numeric_reductions_all_dtypes(name, dtype):
    if dtype is complex and name in {"min", "max", "var", "std"}:
        with pytest.raises(ValueError, match="not supported"):
            getattr(sdnp, name)(array([[1 + 2j, 3j]], dtype))
        return

    values = [[1, 2, 3], [4, 5, 6]]
    operand = array(values, dtype)
    expected = getattr(np, name)(np.asarray(values, dtype=NP_DTYPES[dtype]))
    assert_matches(getattr(sdnp, name)(operand), expected)


@pytest.mark.parametrize("name", NUMERIC_REDUCTIONS)
@pytest.mark.parametrize(
    "keyword,axis",
    [
        ("axis", 0),
        ("axis", -1),
        ("axes", (0, 2)),
        ("axes", (-1, -3)),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_numeric_reduction_axes_negative_multi_and_keepdims(
    name, keyword, axis, keepdims
):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 1
    operand = sdnp.array(values.tolist(), dtype=float)
    actual = getattr(sdnp, name)(operand, **{keyword: axis}, keepdims=keepdims)
    expected = getattr(np, name)(values, axis=axis, keepdims=keepdims)
    assert_matches(actual, expected)


@pytest.mark.parametrize("name", NUMERIC_REDUCTIONS)
def test_reductions_accept_axis_sequence_as_axis_alias(name):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 1
    actual = getattr(sdnp, name)(sdnp.array(values.tolist()), axis=(0, 2))
    assert_matches(actual, getattr(np, name)(values, axis=(0, 2)))


@pytest.mark.parametrize("name", NUMERIC_REDUCTIONS)
def test_reductions_handle_noncontiguous_arrays(name):
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 1
    operand = sdnp.array(values.tolist()).permute_axes((2, 0, 1))
    actual = getattr(sdnp, name)(operand, axes=(0, 2), keepdims=True)
    expected = getattr(np, name)(
        values.transpose(2, 0, 1), axis=(0, 2), keepdims=True
    )
    assert_matches(actual, expected)


@pytest.mark.parametrize(
    "name", ["sum", "prod", "min", "max", "mean", "var", "std"]
)
def test_float_nan_policy_matches_numpy(name):
    values = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 7.0]])
    operand = sdnp.array(values.tolist())
    assert_matches(
        getattr(sdnp, name)(operand, axis=1, nan_policy="propagate"),
        numpy_reduction(name, values, axis=1),
    )
    assert_matches(
        getattr(sdnp, name)(operand, axis=1, nan_policy="ignore"),
        numpy_reduction(name, values, axis=1, nan_policy="ignore"),
    )


@pytest.mark.parametrize(
    "name", ["sum", "prod", "min", "max", "mean", "var", "std"]
)
def test_all_nan_reductions_with_ignore_return_nan(name):
    operand = sdnp.array([[np.nan, np.nan], [1.0, np.nan]])
    result = getattr(sdnp, name)(operand, axis=1, nan_policy="ignore")
    actual = as_numpy(result)
    assert np.isnan(actual[0])
    if name == "sum":
        assert actual[1] == 1.0
    elif name == "prod":
        assert actual[1] == 1.0


@pytest.mark.parametrize("name", ["sum", "prod", "mean"])
def test_complex_nan_policy(name):
    values = np.array([1 + 2j, complex(np.nan, 0), 3 + 4j], dtype=np.complex128)
    operand = sdnp.array(values.tolist(), dtype=complex)
    assert_matches(getattr(sdnp, name)(operand), getattr(np, name)(values))
    assert_matches(
        getattr(sdnp, name)(operand, nan_policy="ignore"),
        getattr(np, f"nan{name}")(values),
    )


@pytest.mark.parametrize("name", ["any", "all"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [None, 0, -1, (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_logical_reductions(name, dtype, axis, keepdims):
    values = np.array(
        [[[0, 1], [2, 0]], [[0, 0], [3, 4]]],
        dtype=NP_DTYPES[dtype],
    )
    operand = sdnp.array(values.tolist(), dtype=dtype)
    actual = getattr(sdnp, name)(operand, axis=axis, keepdims=keepdims)
    expected = getattr(np, name)(values, axis=axis, keepdims=keepdims)
    assert_matches(actual, expected)


@pytest.mark.parametrize("name", ["any", "all"])
def test_logical_reductions_handle_noncontiguous_complex(name):
    values = np.array([[0j, 1j, 0j], [2 + 0j, 0j, 3j]])
    operand = sdnp.array(values.tolist(), dtype=complex).T
    assert_matches(
        getattr(sdnp, name)(operand, axis=0),
        getattr(np, name)(values.T, axis=0),
    )


@pytest.mark.parametrize("name", ["argmin", "argmax"])
@pytest.mark.parametrize("dtype", (bool, int, float))
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_arg_reductions_axes_dtype_and_scalar_unwrap(name, dtype, axis):
    values = np.array([[3, 1, 2], [0, 5, 4]], dtype=NP_DTYPES[dtype])
    operand = sdnp.array(values.tolist(), dtype=dtype)
    result = getattr(sdnp, name)(operand, axis=axis)
    assert_matches(result, getattr(np, name)(values, axis=axis))
    if axis is None:
        assert isinstance(result, int)


@pytest.mark.parametrize("name", ["argmin", "argmax"])
def test_arg_reductions_nan_policies(name):
    values = np.array([[3.0, np.nan, 2.0], [4.0, 1.0, 5.0]])
    operand = sdnp.array(values.tolist())
    assert_matches(
        getattr(sdnp, name)(operand, axis=1), getattr(np, name)(values, axis=1)
    )
    assert_matches(
        getattr(sdnp, name)(operand, axis=1, nan_policy="ignore"),
        getattr(np, f"nan{name}")(values, axis=1),
    )


@pytest.mark.parametrize("name", ["argmin", "argmax"])
def test_arg_reductions_reject_complex_and_all_nan_slices(name):
    with pytest.raises(ValueError, match="complex"):
        getattr(sdnp, name)(array([1 + 1j, 2j], complex))
    with pytest.raises(ValueError, match="all-NaN"):
        getattr(sdnp, name)(
            sdnp.array([[np.nan, np.nan], [1.0, 2.0]]),
            axis=1,
            nan_policy="ignore",
        )


@pytest.mark.parametrize("name", ["argmin", "argmax"])
def test_arg_reductions_use_c_order_for_noncontiguous_input(name):
    values = np.array([[9, 1, 7], [3, 8, 2]])
    operand = sdnp.array(values.tolist()).T
    assert_matches(getattr(sdnp, name)(operand), getattr(np, name)(values.T))
    assert_matches(
        getattr(sdnp, name)(operand, axis=0),
        getattr(np, name)(values.T, axis=0),
    )


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_cumulative_functions_axes_dtypes_and_flattening(name, dtype, axis):
    values = np.array([[1, 2, 3], [4, 1, 2]], dtype=NP_DTYPES[dtype])
    operand = sdnp.array(values.tolist(), dtype=dtype)
    actual = getattr(sdnp, name)(operand, axis=axis)
    expected = getattr(np, name)(values, axis=axis)
    assert_matches(actual, expected)


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
def test_cumulative_functions_noncontiguous_and_negative_axis(name):
    values = np.arange(1, 13).reshape(3, 4)
    operand = sdnp.array(values.tolist()).T
    assert_matches(getattr(sdnp, name)(operand), getattr(np, name)(values.T))
    assert_matches(
        getattr(sdnp, name)(operand, axis=-1),
        getattr(np, name)(values.T, axis=-1),
    )


def nan_cumulative(name, values, axis):
    identity = 0.0 if name == "cumsum" else 1.0
    replaced = np.where(np.isnan(values), identity, values)
    result = getattr(np, name)(replaced, axis=axis)
    if axis is None:
        flat = values.ravel()
        result = np.asarray(result)
        first_valid = np.maximum.accumulate(~np.isnan(flat))
        result[~first_valid] = np.nan
        return result
    seen = np.maximum.accumulate(~np.isnan(values), axis=axis)
    result[~seen] = np.nan
    return result


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_cumulative_nan_policies(name, axis):
    values = np.array([[np.nan, 2.0, np.nan], [3.0, np.nan, 4.0]])
    operand = sdnp.array(values.tolist())
    assert_matches(
        getattr(sdnp, name)(operand, axis=axis),
        getattr(np, name)(values, axis=axis),
    )
    assert_matches(
        getattr(sdnp, name)(operand, axis=axis, nan_policy="ignore"),
        nan_cumulative(name, values, axis),
    )


@pytest.mark.parametrize(
    "name,expected_dtype",
    [
        ("sum", int),
        ("prod", int),
        ("min", bool),
        ("max", bool),
        ("mean", float),
        ("var", float),
        ("std", float),
        ("any", bool),
        ("all", bool),
        ("argmin", int),
        ("argmax", int),
    ],
)
def test_reduction_output_dtype_and_scalar_unwrap(name, expected_dtype):
    operand = sdnp.array([[True, False], [True, True]], dtype=bool)
    result = getattr(sdnp, name)(operand)
    assert not isinstance(result, sdnp.Array)
    assert type(result) is expected_dtype


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
def test_bool_cumulative_output_is_int_array(name):
    result = getattr(sdnp, name)(sdnp.array([True, False, True], dtype=bool))
    assert isinstance(result, sdnp.Array)
    assert result.dtype is int


@pytest.mark.parametrize("name", ["sum", "prod"])
def test_empty_reductions_use_identity(name):
    operand = sdnp.zeros((2, 0, 3), dtype=int)
    identity = 0 if name == "sum" else 1
    assert_matches(
        getattr(sdnp, name)(operand, axis=1),
        np.full((2, 3), identity, dtype=np.int64),
    )
    assert_matches(getattr(sdnp, name)(operand), identity)


@pytest.mark.parametrize("name", ["any", "all"])
def test_empty_logical_reductions_use_identity(name):
    operand = sdnp.zeros((2, 0), dtype=float)
    expected = (
        np.any(np.empty((2, 0)), axis=1)
        if name == "any"
        else np.all(np.empty((2, 0)), axis=1)
    )
    assert_matches(getattr(sdnp, name)(operand, axis=1), expected)


@pytest.mark.parametrize("name", ["min", "max", "mean", "var", "std"])
def test_empty_numeric_reductions_raise(name):
    with pytest.raises(ValueError, match="empty"):
        getattr(sdnp, name)(sdnp.zeros((2, 0)), axis=1)


@pytest.mark.parametrize("name", ["argmin", "argmax"])
def test_empty_arg_reductions_raise(name):
    with pytest.raises(ValueError, match="empty"):
        getattr(sdnp, name)(sdnp.zeros((0,), dtype=int))
    with pytest.raises(ValueError, match="empty"):
        getattr(sdnp, name)(sdnp.zeros((2, 0), dtype=int), axis=1)


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
def test_empty_cumulative_functions_preserve_expected_shape(name):
    operand = sdnp.zeros((2, 0, 3), dtype=float)
    assert_matches(getattr(sdnp, name)(operand, axis=1), np.empty((2, 0, 3)))
    assert_matches(getattr(sdnp, name)(operand), np.empty((0,)))


def test_integer_reductions_wrap():
    maximum = np.iinfo(np.int64).max
    minimum = np.iinfo(np.int64).min
    assert sdnp.sum(array([maximum, 1], int)) == minimum
    assert sdnp.prod(array([maximum, 2], int)) == -2
    assert_matches(
        sdnp.cumsum(array([maximum, 1], int)),
        np.array([maximum, minimum], dtype=np.int64),
    )
    assert_matches(
        sdnp.cumprod(array([maximum, 2], int)),
        np.array([maximum, -2], dtype=np.int64),
    )


@pytest.mark.parametrize("name", NUMERIC_REDUCTIONS)
def test_numeric_reduction_invalid_axis_arguments(name):
    operand = sdnp.ones((2, 3, 4))
    func = getattr(sdnp, name)
    with pytest.raises(ValueError, match="both axis and axes"):
        func(operand, axis=0, axes=(1,))
    with pytest.raises(ValueError, match="non-empty"):
        func(operand, axes=())
    with pytest.raises(ValueError, match="duplicates"):
        func(operand, axes=(0, -3))
    with pytest.raises(IndexError, match="out of bounds"):
        func(operand, axis=3)
    with pytest.raises(TypeError, match="axis"):
        func(operand, axis="bad")
    with pytest.raises(ValueError, match="nan_policy"):
        func(operand, nan_policy="omit")


@pytest.mark.parametrize("name", ["any", "all"])
def test_logical_reduction_invalid_arguments(name):
    operand = sdnp.ones((2, 3))
    func = getattr(sdnp, name)
    with pytest.raises(IndexError, match="out of bounds"):
        func(operand, axis=2)
    with pytest.raises(ValueError, match="duplicates"):
        func(operand, axis=(0, -2))
    with pytest.raises(TypeError, match="unexpected keyword"):
        func(operand, axes=(0,))
    with pytest.raises(TypeError):
        func([True, False])


@pytest.mark.parametrize("name", ["argmin", "argmax", "cumsum", "cumprod"])
def test_single_axis_functions_reject_invalid_arguments(name):
    operand = sdnp.ones((2, 3))
    func = getattr(sdnp, name)
    with pytest.raises(IndexError, match="out of bounds"):
        func(operand, axis=2)
    with pytest.raises(TypeError, match="axis"):
        func(operand, axis=(0, 1))
    with pytest.raises(ValueError, match="nan_policy"):
        func(operand, nan_policy="omit")
    with pytest.raises(TypeError):
        func([[1, 2], [3, 4]])
