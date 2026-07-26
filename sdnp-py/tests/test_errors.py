"""Public Python-boundary exception classes and stable message fragments."""

import re

import pytest
import sdnp


def assert_raises(error_type, fragment, call):
    with pytest.raises(error_type, match=re.escape(fragment)):
        call()


@pytest.mark.parametrize(
    ("fragment", "call"),
    [
        ("dtype", lambda: sdnp.zeros((2,), dtype=str)),
        ("expected nested list/tuple", lambda: sdnp.array(object())),
        (
            "where condition must be a bool array",
            lambda: sdnp.where(sdnp.array([1, 0]), 1, 0),
        ),
        (
            "index must be int, slice, ellipsis, None, or array",
            lambda: sdnp.array([1, 2])[1.5],
        ),
        (
            "fancy index must be an integer or boolean array",
            lambda: sdnp.array([1, 2])[sdnp.array([0.0])],
        ),
        (
            "axis must be an integer",
            lambda: sdnp.argmax(sdnp.ones((2,)), axis=1.5),
        ),
        (
            "concatenate argument must be a sequence of arrays",
            lambda: sdnp.concatenate(1),
        ),
        ("nditer must be an sdnp.Array", lambda: sdnp.nditer((1,))),
        (
            "modular array power is not supported",
            lambda: pow(sdnp.array([2]), 3, 5),
        ),
    ],
)
def test_type_errors_use_type_error(fragment, call):
    assert_raises(TypeError, fragment, call)


@pytest.mark.parametrize("value", ["123", b"123", bytearray(b"123")])
def test_string_like_sequences_are_rejected_without_recursion(value):
    with pytest.raises(TypeError, match="expected nested list/tuple"):
        sdnp.array(value)
    with pytest.raises(TypeError, match="unsupported element type"):
        sdnp.array([value])


def test_excessive_sequence_nesting_is_rejected_safely():
    value = 1
    for _ in range(66):
        value = [value]
    with pytest.raises(ValueError, match="maximum depth"):
        sdnp.array(value)


@pytest.mark.parametrize(
    ("fragment", "call"),
    [
        ("0-dimensional arrays cannot be created", lambda: sdnp.array(1)),
        ("shape dimensions must be non-negative", lambda: sdnp.zeros((-1, 2))),
        (
            "shape size overflows usize",
            lambda: sdnp.zeros((2**62, 8)),
        ),
        (
            "inhomogeneous nested sequence",
            lambda: sdnp.array([[1], [2, 3]]),
        ),
        ("arange step must not be zero", lambda: sdnp.arange(0, 3, 0)),
        (
            "linspace bounds must be finite",
            lambda: sdnp.linspace(float("nan"), 1.0, 3),
        ),
        (
            "logspace base must be finite and greater than zero",
            lambda: sdnp.logspace(0.0, 1.0, 3, base=0.0),
        ),
        (
            "geomspace bounds must have the same sign",
            lambda: sdnp.geomspace(-1.0, 1.0, 3),
        ),
        (
            "only one reshape dimension may be -1",
            lambda: sdnp.arange(4).reshape((-1, -1)),
        ),
        (
            "cannot reshape array of size 4",
            lambda: sdnp.arange(4).reshape((3,)),
        ),
        (
            "cannot squeeze axis 0 with length 2",
            lambda: sdnp.ones((2, 1)).squeeze(0),
        ),
        (
            "axes must be a permutation",
            lambda: sdnp.ones((2, 2)).permute_axes((0, 0)),
        ),
        (
            "cannot specify both axis and axes",
            lambda: sdnp.sum(sdnp.ones((2,)), axis=0, axes=(0,)),
        ),
        (
            "axes must not contain duplicates",
            lambda: sdnp.sum(sdnp.ones((2, 2)), axes=(0, -2)),
        ),
        (
            "min of empty array / empty axis",
            lambda: sdnp.min(sdnp.zeros((0,))),
        ),
        (
            "argmax of all-NaN slice",
            lambda: sdnp.argmax(
                sdnp.array([float("nan")]), nan_policy="ignore"
            ),
        ),
        (
            "nan_policy must be 'propagate' or 'ignore'",
            lambda: sdnp.sum(sdnp.ones((2,)), nan_policy="bad"),
        ),
        (
            "concatenate requires at least one array",
            lambda: sdnp.concatenate(()),
        ),
        (
            "all arrays must have the same dtype in concatenate",
            lambda: sdnp.concatenate((sdnp.array([1]), sdnp.array([1.0]))),
        ),
        (
            "all arrays must have the same shape",
            lambda: sdnp.stack((sdnp.ones((2,)), sdnp.ones((1, 2)))),
        ),
        (
            "operands could not be broadcast together",
            lambda: sdnp.where(
                sdnp.array([True, False]),
                sdnp.ones((3,)),
                sdnp.zeros((3,)),
            ),
        ),
        (
            "clip bounds must be scalar values",
            lambda: sdnp.clip(sdnp.arange(3), sdnp.array([0]), 2),
        ),
        (
            "sort is not supported for complex arrays",
            lambda: sdnp.sort(sdnp.array([1 + 1j])),
        ),
        (
            "ordering comparisons not supported for complex",
            lambda: sdnp.less(sdnp.array([1 + 1j]), 2 + 0j),
        ),
        (
            "nditer supports 1-2 operands",
            lambda: sdnp.nditer(()),
        ),
        (
            "nditer requires operands with the same dtype",
            lambda: sdnp.nditer((sdnp.array([1]), sdnp.array([1.0]))),
        ),
        (
            "operands could not be broadcast together",
            lambda: sdnp.nditer((sdnp.ones((2,)), sdnp.ones((3,)))),
        ),
        (
            "matmul inner dimensions differ",
            lambda: sdnp.matmul(sdnp.ones((1, 2)), sdnp.ones((1, 2))),
        ),
        (
            "dot supports only 1-D or 2-D operands",
            lambda: sdnp.dot(sdnp.ones((1, 1, 1)), sdnp.ones((1,))),
        ),
        (
            "vdot requires equal flattened sizes",
            lambda: sdnp.vdot(sdnp.ones((2,)), sdnp.ones((3,))),
        ),
        (
            "axis1 and axis2 must be different",
            lambda: sdnp.diagonal(sdnp.ones((2, 2)), axis1=0, axis2=0),
        ),
        (
            "diag requires a 1-D or 2-D array",
            lambda: sdnp.diag(sdnp.ones((1, 1, 1))),
        ),
        (
            "meshgrid inputs must be 1-D arrays",
            lambda: sdnp.meshgrid(sdnp.ones((2, 2))),
        ),
    ],
)
def test_value_errors_use_value_error(fragment, call):
    assert_raises(ValueError, fragment, call)


@pytest.mark.parametrize(
    ("fragment", "call"),
    [
        (
            "axis 2 is out of bounds",
            lambda: sdnp.sum(sdnp.ones((2, 2)), axis=2),
        ),
        (
            "axis 2 is out of bounds",
            lambda: sdnp.sort(sdnp.ones((2, 2)), axis=2),
        ),
        (
            "out of bounds",
            lambda: sdnp.array([1, 2])[5],
        ),
        (
            "single ellipsis",
            lambda: sdnp.ones((2, 2))[..., ...],
        ),
        (
            "too many indices",
            lambda: sdnp.ones((2, 2))[0, 0, 0],
        ),
        (
            "boolean index shape",
            lambda: sdnp.ones((2, 2))[sdnp.array([True, False, True])],
        ),
    ],
)
def test_index_errors_use_index_error(fragment, call):
    assert_raises(IndexError, fragment, call)


@pytest.mark.parametrize(
    ("fragment", "call"),
    [
        ("division by zero", lambda: sdnp.divide(1, 0)),
        (
            "division by zero",
            lambda: sdnp.trunc_divide(sdnp.array([1]), 0),
        ),
        (
            "division by zero",
            lambda: sdnp.remainder(sdnp.array([1]), sdnp.array([0])),
        ),
    ],
)
def test_integer_zero_division_uses_zero_division_error(fragment, call):
    assert_raises(ZeroDivisionError, fragment, call)
