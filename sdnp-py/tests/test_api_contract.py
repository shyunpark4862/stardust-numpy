import importlib
import inspect

import pytest
import sdnp

PUBLIC_API = [
    "Array",
    "array",
    "zeros",
    "ones",
    "full",
    "arange",
    "linspace",
    "logspace",
    "geomspace",
    "meshgrid",
    "eye",
    "eye_with",
    "tri",
    "tri_with",
    "tril",
    "triu",
    "diag",
    "add",
    "subtract",
    "multiply",
    "divide",
    "trunc_divide",
    "remainder",
    "power",
    "negative",
    "absolute",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_not",
    "isnan",
    "isinf",
    "isfinite",
    "conj",
    "real",
    "imag",
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "var",
    "std",
    "any",
    "all",
    "argmin",
    "argmax",
    "cumsum",
    "cumprod",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "where",
    "nonzero",
    "clip",
    "sort",
    "argsort",
    "unique",
    "dot",
    "matmul",
    "vdot",
    "outer",
    "diagonal",
    "trace",
    "ndindex",
    "ndenumerate",
    "nditer",
]


def test_all_is_the_complete_ordered_public_api():
    assert list(sdnp.__all__) == PUBLIC_API
    assert len(sdnp.__all__) == len(set(sdnp.__all__))
    for name in PUBLIC_API:
        assert hasattr(sdnp, name), name
        assert callable(getattr(sdnp, name)), name


def test_test_suite_uses_an_optimized_extension_build():
    native = importlib.import_module("sdnp.sdnp")
    assert native.__optimized__ is True
    assert native.__build_profile__ == "release"


def test_public_types_report_the_sdnp_module():
    assert sdnp.Array.__module__ == "sdnp"
    array = sdnp.array([1, 2, 3])
    assert type(array) is sdnp.Array
    assert type(array.flat).__module__ == "sdnp"
    assert type(iter(array)).__module__ == "sdnp"


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (sdnp.array, ["obj", "dtype", "shape"]),
        (sdnp.zeros, ["shape", "dtype"]),
        (sdnp.ones, ["shape", "dtype"]),
        (sdnp.full, ["shape", "fill_value"]),
        (sdnp.arange, ["start", "stop", "step"]),
        (sdnp.linspace, ["start", "stop", "num", "endpoint"]),
        (sdnp.logspace, ["start", "stop", "num", "endpoint", "base"]),
        (sdnp.geomspace, ["start", "stop", "num", "endpoint"]),
        (sdnp.eye, ["n", "dtype"]),
        (sdnp.eye_with, ["n", "m", "k", "dtype"]),
        (sdnp.tri, ["n", "dtype"]),
        (sdnp.tri_with, ["n", "m", "k", "dtype"]),
        (sdnp.tril, ["array", "k"]),
        (sdnp.triu, ["array", "k"]),
        (sdnp.diag, ["array", "k"]),
        (sdnp.meshgrid, ["arrays", "indexing"]),
    ],
)
def test_creation_signatures_match_the_binding(function, parameters):
    assert list(inspect.signature(function).parameters) == parameters


def test_keyword_only_creation_parameters_are_enforced():
    with pytest.raises(TypeError):
        sdnp.zeros(3, int)
    with pytest.raises(TypeError):
        sdnp.array([1], int)
    with pytest.raises(TypeError):
        sdnp.eye_with(2, 3, 1)
    with pytest.raises(TypeError):
        sdnp.linspace(0, 1, 3, False)


def test_array_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        sdnp.Array()


@pytest.mark.parametrize("scalar", [True, 1, 1.5, 1 + 2j])
def test_python_scalar_input_does_not_expose_zero_dimensional_arrays(scalar):
    with pytest.raises(ValueError, match="0-dimensional"):
        sdnp.array(scalar)


def test_empty_shape_is_rejected_at_the_python_boundary():
    with pytest.raises(ValueError, match="0-dimensional"):
        sdnp.zeros(())
    with pytest.raises(ValueError, match="0-dimensional"):
        sdnp.array(1, shape=[])


@pytest.mark.parametrize(
    ("array", "expected_type"),
    [
        (lambda: sdnp.array([True]), bool),
        (lambda: sdnp.array([7]), int),
        (lambda: sdnp.array([1.25]), float),
        (lambda: sdnp.array([1 + 2j]), complex),
    ],
)
def test_zero_dimensional_results_are_unwrapped_to_builtin_scalars(
    array, expected_type
):
    value = array()[0]
    assert type(value) is expected_type
    assert not isinstance(value, sdnp.Array)


@pytest.mark.parametrize("dtype", [str, bytes, object, "float64"])
def test_unsupported_dtype_objects_are_rejected(dtype):
    with pytest.raises(TypeError, match="unsupported dtype"):
        sdnp.zeros(2, dtype=dtype)
