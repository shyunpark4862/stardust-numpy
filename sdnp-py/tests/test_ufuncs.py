import operator

import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, as_numpy, assert_matches, numpy_call

DTYPES = (bool, int, float, complex)
PROMOTION_CASES = [
    (left, right, max((left, right), key=DTYPES.index))
    for left in DTYPES
    for right in DTYPES
]


def array(values, dtype):
    return sdnp.array(
        np.asarray(values, dtype=NP_DTYPES[dtype]).tolist(), dtype=dtype
    )


def trunc_divide(left, right):
    quotient = np.divide(left, right)
    if np.iscomplexobj(quotient):
        return np.trunc(quotient.real) + 1j * np.trunc(quotient.imag)
    return np.trunc(quotient)


def trunc_remainder(left, right):
    quotient = trunc_divide(left, right)
    return left - quotient * right


@pytest.mark.parametrize(
    "left_dtype,right_dtype,expected_dtype", PROMOTION_CASES
)
def test_add_dtype_promotion_matrix(left_dtype, right_dtype, expected_dtype):
    if left_dtype is bool and right_dtype is bool:
        with pytest.raises(ValueError, match="bool"):
            sdnp.add(array([True, False], bool), array([True, True], bool))
        return

    left = array([1, 2], left_dtype)
    right = array([3, 4], right_dtype)
    result = sdnp.add(left, right)
    expected = np.add(
        np.asarray([1, 2], dtype=NP_DTYPES[left_dtype]),
        np.asarray([3, 4], dtype=NP_DTYPES[right_dtype]),
    ).astype(NP_DTYPES[expected_dtype])
    assert_matches(result, expected)


@pytest.mark.parametrize(
    "name,np_func",
    [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("divide", np.divide),
        ("trunc_divide", trunc_divide),
        ("remainder", trunc_remainder),
        ("power", np.power),
    ],
)
@pytest.mark.parametrize("dtype", (int, float, complex))
def test_numeric_binary_ufuncs_arrays(name, np_func, dtype):
    left_values = [[2, 3, 4], [5, 6, 7]]
    right_values = [[1, 2, 2], [2, 3, 2]]
    left = array(left_values, dtype)
    right = array(right_values, dtype)
    expected = numpy_call(
        np_func,
        np.asarray(left_values, dtype=NP_DTYPES[dtype]),
        np.asarray(right_values, dtype=NP_DTYPES[dtype]),
    )
    if dtype is int and name in {"divide", "trunc_divide", "remainder"}:
        expected = expected.astype(np.int64)
    assert_matches(getattr(sdnp, name)(left, right), expected)


@pytest.mark.parametrize(
    "name,np_func", [("add", np.add), ("multiply", np.multiply)]
)
def test_binary_ufunc_accepts_scalars_lists_and_broadcasts(name, np_func):
    left = [[1], [2], [3]]
    right = [10.5, 20.5]
    assert_matches(
        getattr(sdnp, name)(left, right),
        np_func(np.asarray(left), np.asarray(right)),
    )
    assert_matches(
        getattr(sdnp, name)(sdnp.array(left), 2.5), np_func(left, 2.5)
    )
    assert_matches(
        getattr(sdnp, name)(2.5, sdnp.array(right)), np_func(2.5, right)
    )
    assert_matches(getattr(sdnp, name)(2, 3.5), np_func(2, 3.5))


@pytest.mark.parametrize(
    "name,np_func",
    [("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply)],
)
def test_binary_ufunc_handles_noncontiguous_operands(name, np_func):
    base = sdnp.array([[1, 2, 3], [4, 5, 6]])
    left = base.T
    right = sdnp.array([[10, 20], [30, 40], [50, 60]])
    assert_matches(
        getattr(sdnp, name)(left, right),
        np_func(as_numpy(left), as_numpy(right)),
    )


def test_broadcast_failure_is_reported():
    with pytest.raises(ValueError, match="broadcast"):
        sdnp.add(sdnp.ones((2, 3)), sdnp.ones((2, 2)))


@pytest.mark.parametrize(
    "name,np_func",
    [
        ("equal", np.equal),
        ("not_equal", np.not_equal),
        ("less", np.less),
        ("less_equal", np.less_equal),
        ("greater", np.greater),
        ("greater_equal", np.greater_equal),
    ],
)
@pytest.mark.parametrize("dtype", (int, float))
def test_comparison_ufuncs(name, np_func, dtype):
    left = array([[1, 2], [3, 4]], dtype)
    right = array([2, 3], dtype)
    assert_matches(
        getattr(sdnp, name)(left, right),
        np_func(as_numpy(left), as_numpy(right)),
    )


@pytest.mark.parametrize("name", ["equal", "not_equal"])
def test_equality_supports_bool_and_complex(name):
    np_func = getattr(np, name)
    for dtype, values in [(bool, [True, False]), (complex, [1 + 2j, 0j])]:
        operand = array(values, dtype)
        assert_matches(
            getattr(sdnp, name)(operand, operand), np_func(values, values)
        )


@pytest.mark.parametrize(
    "name", ["less", "less_equal", "greater", "greater_equal"]
)
@pytest.mark.parametrize(
    "dtype,values", [(bool, [True, False]), (complex, [1 + 1j, 2j])]
)
def test_ordering_rejects_bool_and_complex(name, dtype, values):
    with pytest.raises(ValueError, match="ordering|bool"):
        getattr(sdnp, name)(array(values, dtype), array(values, dtype))


@pytest.mark.parametrize(
    "name,np_func",
    [("logical_and", np.logical_and), ("logical_or", np.logical_or)],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_logical_binary_ufuncs_use_truthiness(name, np_func, dtype):
    values = [0, 1, -2]
    left = array(values, dtype)
    right = array([1, 0, 3], dtype)
    assert_matches(
        getattr(sdnp, name)(left, right),
        np_func(as_numpy(left), as_numpy(right)),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_logical_not_uses_truthiness(dtype):
    operand = array([0, 1, -2], dtype)
    assert_matches(sdnp.logical_not(operand), np.logical_not(as_numpy(operand)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_negative_and_absolute(dtype):
    values = [True, False] if dtype is bool else [-2, 0, 3]
    operand = array(values, dtype)
    if dtype is bool:
        assert_matches(
            sdnp.negative(operand), np.asarray(values, dtype=np.bool_)
        )
    else:
        assert_matches(sdnp.negative(operand), np.negative(as_numpy(operand)))
    assert_matches(sdnp.absolute(operand), np.absolute(as_numpy(operand)))
    assert_matches(-operand, as_numpy(sdnp.negative(operand)))
    assert_matches(abs(operand), as_numpy(sdnp.absolute(operand)))


@pytest.mark.parametrize(
    "name,np_func",
    [("isnan", np.isnan), ("isinf", np.isinf), ("isfinite", np.isfinite)],
)
@pytest.mark.parametrize("dtype", (float, complex))
def test_float_classification_including_nan_and_inf(name, np_func, dtype):
    values = [0.0, np.nan, np.inf, -np.inf]
    if dtype is complex:
        values = [
            0j,
            complex(np.nan, 0),
            complex(0, np.inf),
            complex(np.inf, np.nan),
        ]
    operand = array(values, dtype)
    assert_matches(getattr(sdnp, name)(operand), np_func(as_numpy(operand)))


@pytest.mark.parametrize(
    "name", ["isnan", "isinf", "isfinite", "conj", "real", "imag"]
)
@pytest.mark.parametrize("dtype", (bool, int))
def test_special_unary_ufuncs_reject_unsupported_real_dtypes(name, dtype):
    with pytest.raises(ValueError, match="unsupported unary"):
        getattr(sdnp, name)(array([0, 1], dtype))


def test_complex_component_ufuncs():
    operand = array([1 + 2j, -3 - 4j, complex(np.nan, np.inf)], complex)
    for name, np_func in [
        ("conj", np.conj),
        ("real", np.real),
        ("imag", np.imag),
    ]:
        assert_matches(getattr(sdnp, name)(operand), np_func(as_numpy(operand)))


@pytest.mark.parametrize("name", ["negative", "absolute", "logical_not"])
def test_unary_ufuncs_unwrap_scalar_results(name):
    result = getattr(sdnp, name)(-3.0)
    assert not isinstance(result, sdnp.Array)


ARITHMETIC_DUNDERS = [
    ("add", operator.add, np.add),
    ("sub", operator.sub, np.subtract),
    ("mul", operator.mul, np.multiply),
    ("truediv", operator.truediv, np.divide),
    ("floordiv", operator.floordiv, trunc_divide),
    ("mod", operator.mod, trunc_remainder),
    ("pow", operator.pow, np.power),
]


@pytest.mark.parametrize("_name,op,np_func", ARITHMETIC_DUNDERS)
def test_array_arithmetic_and_reverse_dunders(_name, op, np_func):
    operand = sdnp.array([2.0, 3.0, 4.0])
    assert_matches(op(operand, 2.0), np_func(np.array([2.0, 3.0, 4.0]), 2.0))
    assert_matches(op(12.0, operand), np_func(12.0, np.array([2.0, 3.0, 4.0])))


@pytest.mark.parametrize(
    "op,np_func",
    [
        (operator.eq, np.equal),
        (operator.ne, np.not_equal),
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
    ],
)
def test_array_comparison_dunders_and_reflected_dispatch(op, np_func):
    operand = sdnp.array([1, 2, 3])
    assert_matches(op(operand, 2), np_func(np.array([1, 2, 3]), 2))
    assert_matches(op(2, operand), np_func(2, np.array([1, 2, 3])))


def test_power_dunder_rejects_modulus():
    with pytest.raises(TypeError, match="modular"):
        pow(sdnp.array([2, 3]), 2, 5)


def test_integer_arithmetic_wraps_at_i64_boundaries():
    maximum = np.iinfo(np.int64).max
    minimum = np.iinfo(np.int64).min
    assert_matches(
        sdnp.add(array([maximum], int), 1), np.array([minimum], dtype=np.int64)
    )
    assert_matches(
        sdnp.subtract(array([minimum], int), 1),
        np.array([maximum], dtype=np.int64),
    )
    assert_matches(
        sdnp.multiply(array([maximum], int), 2), np.array([-2], dtype=np.int64)
    )
    assert_matches(
        sdnp.negative(array([minimum], int)),
        np.array([minimum], dtype=np.int64),
    )
    assert_matches(
        sdnp.absolute(array([minimum], int)),
        np.array([minimum], dtype=np.int64),
    )
    assert_matches(
        sdnp.power(array([2], int), 63), np.array([minimum], dtype=np.int64)
    )


@pytest.mark.parametrize("name", ["divide", "trunc_divide"])
def test_integer_division_errors(name):
    with pytest.raises(ZeroDivisionError):
        getattr(sdnp, name)(array([1], int), 0)
    with pytest.raises(ValueError, match="overflow"):
        getattr(sdnp, name)(array([np.iinfo(np.int64).min], int), -1)


def test_integer_remainder_errors():
    with pytest.raises(ZeroDivisionError):
        sdnp.remainder(array([1], int), 0)
    with pytest.raises(ValueError, match="overflow"):
        sdnp.remainder(array([np.iinfo(np.int64).min], int), -1)


@pytest.mark.parametrize("exponent", [-1, 2**32])
def test_integer_power_rejects_invalid_exponents(exponent):
    with pytest.raises(ValueError, match="exponent"):
        sdnp.power(array([2], int), exponent)


def test_float_division_remainder_power_follow_ieee_semantics():
    left = array([1.0, 0.0, -1.0], float)
    zero = array([0.0, 0.0, 0.0], float)
    assert_matches(
        sdnp.divide(left, zero), numpy_call(np.divide, as_numpy(left), 0.0)
    )
    assert_matches(
        sdnp.remainder(left, zero), numpy_call(np.fmod, as_numpy(left), 0.0)
    )
    assert_matches(
        sdnp.power(array([-1.0, 0.0], float), 0.5), np.array([np.nan, 0.0])
    )


@pytest.mark.parametrize(
    "name",
    [
        "add",
        "subtract",
        "multiply",
        "divide",
        "trunc_divide",
        "remainder",
        "power",
    ],
)
def test_bool_bool_arithmetic_is_rejected(name):
    with pytest.raises(ValueError, match="bool"):
        getattr(sdnp, name)(array([True], bool), array([True], bool))
