import os
import warnings

import numpy as np
import pytest
import sdnp
from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


DTYPES = (bool, int, float, complex)
NUMERIC_DTYPES = (int, float, complex)
REAL_DTYPES = (bool, int, float)

NP_DTYPES = {
    bool: np.bool_,
    int: np.int64,
    float: np.float64,
    complex: np.complex128,
}


@pytest.fixture(params=DTYPES, ids=lambda dtype: dtype.__name__)
def dtype(request):
    return request.param


@pytest.fixture(params=NUMERIC_DTYPES, ids=lambda dtype: dtype.__name__)
def numeric_dtype(request):
    return request.param


def as_numpy(value):
    if not isinstance(value, sdnp.Array):
        return value
    result = np.asarray(value.to_list(), dtype=NP_DTYPES[value.dtype])
    return result.reshape(tuple(value.shape))


def assert_matches(actual, expected, *, check_dtype=True):
    if isinstance(actual, sdnp.Array):
        expected_array = np.asarray(expected)
        actual_array = as_numpy(actual)
        assert tuple(actual.shape) == expected_array.shape
        assert actual.size == expected_array.size
        if check_dtype:
            assert actual_array.dtype == expected_array.dtype
        if actual_array.dtype.kind in "fc" or expected_array.dtype.kind in "fc":
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(actual_array, expected_array)
        return

    assert np.asarray(expected).ndim == 0
    expected_scalar = np.asarray(expected).item()
    if isinstance(actual, (float, complex)) or isinstance(
        expected_scalar, (float, complex)
    ):
        assert actual == pytest.approx(expected_scalar, nan_ok=True)
    else:
        assert actual == expected_scalar


def numpy_call(callable_, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return callable_(*args, **kwargs)
