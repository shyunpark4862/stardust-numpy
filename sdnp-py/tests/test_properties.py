import numpy as np
import pytest
import sdnp
from conftest import NP_DTYPES, assert_matches
from hypothesis import given
from strategies import broadcast_pair, nonempty_shaped_values, shaped_values


@pytest.mark.property
@given(nonempty_shaped_values())
def test_array_round_trip_matches_numpy(case):
    dtype, shape, values = case
    actual = sdnp.array(values, dtype=dtype)
    expected = np.asarray(values, dtype=NP_DTYPES[dtype]).reshape(shape)
    assert_matches(actual, expected)


@pytest.mark.property
@given(shaped_values(dtypes=(bool, int, float)))
def test_factory_shapes_and_values_match_numpy(case):
    dtype, shape, _values = case
    assert_matches(
        sdnp.zeros(shape, dtype=dtype),
        np.zeros(shape, dtype=NP_DTYPES[dtype]),
    )
    assert_matches(
        sdnp.ones(shape, dtype=dtype),
        np.ones(shape, dtype=NP_DTYPES[dtype]),
    )


@pytest.mark.property
@given(broadcast_pair())
def test_broadcast_add_matches_numpy(case):
    dtype, left, right = case
    actual = sdnp.add(
        sdnp.array(left, dtype=dtype),
        sdnp.array(right, dtype=dtype),
    )
    expected = np.add(
        np.asarray(left, dtype=NP_DTYPES[dtype]),
        np.asarray(right, dtype=NP_DTYPES[dtype]),
    )
    assert_matches(actual, expected)


@pytest.mark.property
@given(nonempty_shaped_values(dtypes=(int, float)))
def test_unary_and_transpose_properties(case):
    dtype, _shape, values = case
    actual = sdnp.array(values, dtype=dtype)
    expected = np.asarray(values, dtype=NP_DTYPES[dtype])

    assert_matches(-actual, -expected)
    assert_matches(abs(actual), np.abs(expected))
    assert_matches(actual.T, expected.T)
    assert_matches(actual.T.T, expected)


@pytest.mark.property
@given(
    nonempty_shaped_values(
        dtypes=(int, float), min_dims=2, max_dims=3, max_side=3
    )
)
def test_axis_reductions_match_numpy(case):
    dtype, _shape, values = case
    actual = sdnp.array(values, dtype=dtype)
    expected = np.asarray(values, dtype=NP_DTYPES[dtype])

    assert_matches(sdnp.sum(actual, axis=-1), np.sum(expected, axis=-1))
    assert_matches(sdnp.prod(actual, axis=0), np.prod(expected, axis=0))
    assert_matches(sdnp.mean(actual, axis=-1), np.mean(expected, axis=-1))


@pytest.mark.property
@given(
    nonempty_shaped_values(
        dtypes=(int, float), min_dims=1, max_dims=3, max_side=4
    )
)
def test_reverse_and_step_slices_match_numpy(case):
    dtype, _shape, values = case
    actual = sdnp.array(values, dtype=dtype)
    expected = np.asarray(values, dtype=NP_DTYPES[dtype])

    assert_matches(actual[::-1], expected[::-1])
    assert_matches(actual[::2], expected[::2])


@pytest.mark.property
@given(
    nonempty_shaped_values(
        dtypes=(int, float), min_dims=2, max_dims=2, max_side=3
    )
)
def test_matrix_times_transpose_matches_numpy(case):
    dtype, _shape, values = case
    actual = sdnp.array(values, dtype=dtype)
    expected = np.asarray(values, dtype=NP_DTYPES[dtype])
    assert_matches(sdnp.matmul(actual, actual.T), expected @ expected.T)
