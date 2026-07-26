import sys

import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, assert_matches
from hypothesis import given
from hypothesis import strategies as st

SAMPLES = {
    bool: [False, True, False],
    int: [-2, 0, 3],
    float: [-2.5, 0.0, 3.25],
    complex: [-2 + 1j, 0j, 3.25 - 4j],
}
NUMERIC_DTYPES = (int, float, complex)


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_array_constructs_nested_lists_and_tuples_for_every_dtype(dtype):
    values = SAMPLES[dtype]
    nested = (values, tuple(reversed(values)))
    result = sdnp.array(nested)
    expected = np.asarray(nested, dtype=NP_DTYPES[dtype])

    assert result.dtype is dtype
    assert result.shape == [2, 3]
    assert_matches(result, expected)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([False, True], bool),
        ([False, 2], int),
        ([False, 2, 3.5], float),
        ([False, 2, 3.5, 4 + 1j], complex),
    ],
)
def test_array_infers_the_promoted_dtype(values, dtype):
    result = sdnp.array(values)
    assert result.dtype is dtype
    assert_matches(result, np.asarray(values, dtype=NP_DTYPES[dtype]))


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_array_explicit_dtype_accepts_values_of_that_dtype(dtype):
    result = sdnp.array(SAMPLES[dtype], dtype=dtype)
    assert result.dtype is dtype
    assert_matches(result, np.asarray(SAMPLES[dtype], dtype=NP_DTYPES[dtype]))


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_array_can_cast_bool_nested_values_to_every_dtype(dtype):
    result = sdnp.array([False, True], dtype=dtype)
    expected = np.asarray([False, True], dtype=NP_DTYPES[dtype])
    assert_matches(result, expected)


@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_array_scalar_fill_shape_and_dtype(dtype):
    result = sdnp.array(True, shape=(2, 3), dtype=dtype)
    expected = np.full((2, 3), True, dtype=NP_DTYPES[dtype])
    assert_matches(result, expected)


def test_array_scalar_fill_accepts_an_integer_shape():
    assert_matches(sdnp.array(2.5, shape=3), np.full(3, 2.5))


def test_array_from_array_preserves_or_casts_dtype():
    source = sdnp.array([0, 1, 2])
    preserved = sdnp.array(source)
    cast = sdnp.array(source, dtype=complex)
    assert preserved.dtype is int
    assert_matches(preserved, np.array([0, 1, 2], dtype=np.int64))
    assert_matches(cast, np.array([0, 1, 2], dtype=np.complex128))


def test_empty_nested_input_has_the_binding_default_bool_dtype():
    result = sdnp.array([])
    assert result.dtype is bool
    assert result.shape == [0]
    assert result.to_list() == []


@pytest.mark.parametrize(
    "obj",
    [
        [[1], [2, 3]],
        [1, [2]],
        [object()],
        {"a": 1},
    ],
)
def test_array_rejects_ragged_or_unsupported_inputs(obj):
    with pytest.raises((TypeError, ValueError)):
        sdnp.array(obj)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([0, 1], bool),
        ([1.5], int),
        ([1 + 2j], float),
    ],
)
def test_array_rejects_unsupported_explicit_nested_narrowing(values, dtype):
    with pytest.raises(ValueError):
        sdnp.array(values, dtype=dtype)


@pytest.mark.parametrize("factory", [sdnp.zeros, sdnp.ones])
@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_zeros_and_ones_support_all_dtypes(factory, dtype):
    result = factory((2, 3), dtype=dtype)
    np_factory = np.zeros if factory is sdnp.zeros else np.ones
    expected = np_factory((2, 3), dtype=NP_DTYPES[dtype])
    assert_matches(result, expected)


def test_zeros_and_ones_default_to_float():
    assert sdnp.zeros(2).dtype is float
    assert sdnp.ones(2).dtype is float


@pytest.mark.parametrize("fill_value", [False, True, -3, 2.5, 1.5 - 2j])
def test_full_infers_dtype_from_fill_value(fill_value):
    result = sdnp.full((2, 2), fill_value)
    expected = np.full((2, 2), fill_value, dtype=NP_DTYPES[type(fill_value)])
    assert_matches(result, expected)


@pytest.mark.parametrize("shape", [0, (0,), (2, 0), [1, 2, 0], (2, 3)])
def test_basic_factories_accept_nonnegative_shapes(shape):
    for factory, fill in (
        (sdnp.zeros, 0.0),
        (sdnp.ones, 1.0),
        (lambda s: sdnp.full(s, 4), 4),
    ):
        result = factory(shape)
        expected_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        assert tuple(result.shape) == expected_shape
        assert result.size == int(np.prod(expected_shape, dtype=np.int64))
        if result.size:
            assert all(value == fill for value in result.flat)


@pytest.mark.parametrize("shape", [-1, (2, -1), (1.5,), "23"])
def test_basic_factories_reject_invalid_shapes(shape):
    with pytest.raises((TypeError, ValueError)):
        sdnp.zeros(shape)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sdnp.zeros((sys.maxsize, 3)),
        lambda: sdnp.ones((sys.maxsize, 3)),
        lambda: sdnp.full((sys.maxsize, 3), 1),
    ],
)
def test_basic_factories_reject_shape_size_overflow_without_allocating(factory):
    with pytest.raises((ValueError, OverflowError), match="overflow"):
        factory()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((5,), np.arange(5, dtype=np.int64)),
        ((2, 9), np.arange(2, 9, dtype=np.int64)),
        ((9, 2, -2), np.arange(9, 2, -2, dtype=np.int64)),
        ((2, 9, -1), np.arange(2, 9, -1, dtype=np.int64)),
        ((9, 2, 2), np.arange(9, 2, 2, dtype=np.int64)),
        ((0, 0), np.arange(0, 0, dtype=np.int64)),
    ],
)
def test_arange_matches_numpy(args, expected):
    result = sdnp.arange(*args)
    assert result.dtype is int
    assert_matches(result, expected)


def test_arange_handles_i64_endpoint_boundaries():
    maximum = 2**63 - 1
    minimum = -(2**63)
    assert sdnp.arange(maximum - 2, maximum).to_list() == [
        maximum - 2,
        maximum - 1,
    ]
    assert sdnp.arange(minimum + 2, minimum, -1).to_list() == [
        minimum + 2,
        minimum + 1,
    ]


def test_arange_rejects_zero_step_when_a_stop_is_given():
    with pytest.raises(ValueError, match="step"):
        sdnp.arange(0, 3, 0)


def test_single_argument_arange_ignores_step_per_declared_binding_semantics():
    assert sdnp.arange(4, step=0).to_list() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "call",
    [
        lambda: sdnp.arange(2**63),
        lambda: sdnp.arange(-(2**63) - 1),
        lambda: sdnp.arange(-(2**63), 2**63 - 1),
    ],
)
def test_arange_rejects_integer_or_allocation_overflow(call):
    with pytest.raises((OverflowError, ValueError)):
        call()


@pytest.mark.parametrize("num", [0, 1, 2, 7])
@pytest.mark.parametrize("endpoint", [True, False])
def test_linspace_matches_numpy(num, endpoint):
    assert_matches(
        sdnp.linspace(-2.5, 4.0, num, endpoint=endpoint),
        np.linspace(-2.5, 4.0, num, endpoint=endpoint),
    )


@pytest.mark.parametrize("num", [0, 1, 2, 7])
@pytest.mark.parametrize("endpoint", [True, False])
@pytest.mark.parametrize("base", [0.5, 2.0, 10.0])
def test_logspace_matches_numpy(num, endpoint, base):
    assert_matches(
        sdnp.logspace(-2.0, 3.0, num, endpoint=endpoint, base=base),
        np.logspace(-2.0, 3.0, num, endpoint=endpoint, base=base),
    )


@pytest.mark.parametrize(
    ("start", "stop"), [(1.0, 1000.0), (-1.0, -1000.0), (0.25, 16.0)]
)
@pytest.mark.parametrize("num", [0, 1, 2, 7])
@pytest.mark.parametrize("endpoint", [True, False])
def test_geomspace_matches_numpy(start, stop, num, endpoint):
    assert_matches(
        sdnp.geomspace(start, stop, num, endpoint=endpoint),
        np.geomspace(start, stop, num, endpoint=endpoint),
    )


@pytest.mark.parametrize("function", [sdnp.linspace, sdnp.logspace])
@pytest.mark.parametrize(
    "bad_bound", [float("nan"), float("inf"), -float("inf")]
)
def test_linear_space_factories_require_finite_bounds(function, bad_bound):
    with pytest.raises(ValueError, match="finite"):
        function(bad_bound, 1.0, 3)
    with pytest.raises(ValueError, match="finite"):
        function(1.0, bad_bound, 3)


@pytest.mark.parametrize("base", [0.0, -1.0, float("nan"), float("inf")])
def test_logspace_rejects_invalid_base(base):
    with pytest.raises(ValueError, match="base"):
        sdnp.logspace(0.0, 1.0, 3, base=base)


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
    ],
)
def test_geomspace_rejects_invalid_bounds(start, stop):
    with pytest.raises(ValueError):
        sdnp.geomspace(start, stop, 3)


@pytest.mark.parametrize(
    "function",
    [sdnp.linspace, sdnp.logspace, sdnp.geomspace],
)
def test_space_factories_reject_negative_or_impossible_num(function):
    with pytest.raises(OverflowError):
        function(1.0, 2.0, -1)
    with pytest.raises((ValueError, OverflowError)):
        function(1.0, 2.0, sys.maxsize)


@given(
    start=st.integers(-50, 50),
    stop=st.integers(-50, 50),
    step=st.integers(-10, 10).filter(lambda value: value != 0),
)
@pytest.mark.property
def test_arange_property_matches_numpy(start, stop, step):
    assert_matches(
        sdnp.arange(start, stop, step),
        np.arange(start, stop, step, dtype=np.int64),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("n", [0, 1, 4])
def test_eye_matches_numpy_for_every_supported_dtype(dtype, n):
    assert_matches(
        sdnp.eye(n, dtype=dtype),
        np.eye(n, dtype=NP_DTYPES[dtype]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (2, 4)])
@pytest.mark.parametrize("k", [-4, -1, 0, 1, 4])
def test_eye_with_matches_numpy(dtype, shape, k):
    n, m = shape
    assert_matches(
        sdnp.eye_with(n, m, k=k, dtype=dtype),
        np.eye(n, m, k=k, dtype=NP_DTYPES[dtype]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("n", [0, 1, 4])
def test_tri_matches_numpy_for_every_supported_dtype(dtype, n):
    assert_matches(
        sdnp.tri(n, dtype=dtype),
        np.tri(n, dtype=NP_DTYPES[dtype]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (2, 4)])
@pytest.mark.parametrize("k", [-4, -1, 0, 1, 4])
def test_tri_with_matches_numpy(dtype, shape, k):
    n, m = shape
    assert_matches(
        sdnp.tri_with(n, m, k, dtype=dtype),
        np.tri(n, m, k=k, dtype=NP_DTYPES[dtype]),
    )


@pytest.mark.parametrize(
    "function", [sdnp.eye, sdnp.eye_with, sdnp.tri, sdnp.tri_with]
)
def test_eye_and_tri_factories_reject_bool_dtype(function):
    args = (2, 3) if function in (sdnp.eye_with, sdnp.tri_with) else (2,)
    with pytest.raises(ValueError, match="bool"):
        function(*args, dtype=bool)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sdnp.eye(-1),
        lambda: sdnp.tri(-1),
        lambda: sdnp.eye_with(2, -1),
        lambda: sdnp.tri_with(-1, 2),
        lambda: sdnp.eye_with(sys.maxsize, 3),
        lambda: sdnp.tri_with(sys.maxsize, 3),
    ],
)
def test_eye_and_tri_reject_negative_dimensions_and_overflow(call):
    with pytest.raises((OverflowError, ValueError)):
        call()


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    ],
)
@pytest.mark.parametrize("k", [-2, 0, 2])
def test_tril_and_triu_match_numpy(dtype, values, k):
    array = sdnp.array(values, dtype=dtype)
    expected = np.asarray(values, dtype=NP_DTYPES[dtype])
    assert_matches(sdnp.tril(array, k), np.tril(expected, k))
    assert_matches(sdnp.triu(array, k), np.triu(expected, k))


@pytest.mark.parametrize("function", [sdnp.tril, sdnp.triu])
def test_tril_and_triu_validate_array_and_dtype(function):
    with pytest.raises(TypeError):
        function([[1, 2]])
    with pytest.raises(ValueError, match="bool"):
        function(sdnp.array([[True, False]]))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("values", [[1, 2, 3], [[1, 2], [3, 4]]])
@pytest.mark.parametrize("k", [-3, -1, 0, 1, 3])
def test_diag_matches_numpy(dtype, values, k):
    expected = np.asarray(values, dtype=NP_DTYPES[dtype])
    assert_matches(
        sdnp.diag(sdnp.array(values, dtype=dtype), k), np.diag(expected, k)
    )


@pytest.mark.parametrize(
    "values", [[True, False, True], [[True, False], [False, True]]]
)
def test_diag_rejects_bool_array(values):
    with pytest.raises(ValueError, match="does not support boolean"):
        sdnp.diag(sdnp.array(values))


def test_diag_validates_rank_input_type_and_overflow():
    with pytest.raises(TypeError):
        sdnp.diag([1, 2])
    with pytest.raises(ValueError, match="1-D or 2-D"):
        sdnp.diag(sdnp.zeros((1, 1, 1)))
    with pytest.raises((ValueError, OverflowError)):
        sdnp.diag(sdnp.array([1]), sys.maxsize)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_meshgrid_matches_numpy_for_all_supported_dtypes(dtype, indexing):
    inputs = (
        sdnp.array([1, 2], dtype=dtype),
        sdnp.array([10, 20, 30], dtype=dtype),
        sdnp.array([-1, 1], dtype=dtype),
    )
    expected_inputs = [
        np.asarray(value.to_list(), dtype=NP_DTYPES[dtype]) for value in inputs
    ]
    actual = sdnp.meshgrid(*inputs, indexing=indexing)
    expected = np.meshgrid(*expected_inputs, indexing=indexing)

    assert isinstance(actual, tuple)
    assert len(actual) == 3
    for result, expected_result in zip(actual, expected, strict=True):
        assert_matches(result, expected_result)


def test_meshgrid_empty_call_returns_empty_tuple():
    assert sdnp.meshgrid() == ()


@pytest.mark.parametrize(
    "call",
    [
        lambda: sdnp.meshgrid(sdnp.array([1]), indexing="bad"),
        lambda: sdnp.meshgrid([1, 2]),
        lambda: sdnp.meshgrid(sdnp.array([[1, 2]])),
        lambda: sdnp.meshgrid(sdnp.array([1]), sdnp.array([1.0])),
        lambda: sdnp.meshgrid(sdnp.array([True, False])),
    ],
)
def test_meshgrid_rejects_invalid_inputs(call):
    with pytest.raises((TypeError, ValueError)):
        call()
