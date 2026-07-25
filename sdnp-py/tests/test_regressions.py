"""Regression coverage for previously fragile Python/Rust boundaries."""

import pytest
import sdnp

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


def assert_python_scalar(value):
    assert not isinstance(value, sdnp.Array)
    assert isinstance(value, (bool, int, float, complex))


def test_zero_dimensional_results_never_cross_python_boundary():
    vector = sdnp.array([2, 3])
    matrix = sdnp.array([[1, 2], [3, 4]])
    scalar_results = [
        vector[0],
        sdnp.add(1, 2),
        sdnp.negative(1),
        sdnp.equal(1, 1),
        sdnp.sum(vector),
        sdnp.sum(matrix, axes=(0, 1)),
        sdnp.argmax(vector),
        sdnp.dot(vector, vector),
        sdnp.matmul(vector, vector),
        sdnp.vdot(vector, vector),
        sdnp.trace(matrix),
        sdnp.ones((1,)).squeeze(),
    ]

    for result in scalar_results:
        assert_python_scalar(result)


def test_one_dimensional_iteration_and_full_indexing_do_not_leak_zero_dim():
    array = sdnp.array([10, 20])

    for value in [*array, array[0], array[-1]]:
        assert_python_scalar(value)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda: sdnp.add(I64_MAX, 1), I64_MIN),
        (lambda: sdnp.subtract(I64_MIN, 1), I64_MAX),
        (lambda: sdnp.multiply(I64_MAX, 2), -2),
        (lambda: sdnp.negative(I64_MIN), I64_MIN),
        (lambda: sdnp.absolute(I64_MIN), I64_MIN),
    ],
)
def test_scalar_i64_arithmetic_wraps_instead_of_panicking(call, expected):
    assert call() == expected


def test_array_i64_arithmetic_and_reductions_wrap_consistently():
    maximum = sdnp.array([I64_MAX])
    minimum = sdnp.array([I64_MIN])

    assert (maximum + 1).to_list() == [I64_MIN]
    assert (minimum - 1).to_list() == [I64_MAX]
    assert (maximum * 2).to_list() == [-2]
    assert (-minimum).to_list() == [I64_MIN]
    assert abs(minimum).to_list() == [I64_MIN]
    assert sdnp.sum(sdnp.array([I64_MAX, 1])) == I64_MIN
    assert sdnp.prod(sdnp.array([I64_MAX, 2])) == -2


def test_i64_linalg_accumulators_wrap_consistently():
    left = sdnp.array([I64_MAX, 1])
    right = sdnp.array([1, 1])

    assert sdnp.dot(left, right) == I64_MIN
    assert sdnp.vdot(left, right) == I64_MIN
    assert sdnp.matmul(left, right) == I64_MIN


def test_arange_near_i64_limits_does_not_overflow():
    assert sdnp.arange(I64_MAX - 1, I64_MAX, 2).to_list() == [I64_MAX - 1]
    assert sdnp.arange(I64_MIN + 1, I64_MIN, -2).to_list() == [I64_MIN + 1]


def test_huge_in_range_slice_steps_are_normalized_without_overflow():
    array = sdnp.arange(5)

    assert array[::I64_MAX].to_list() == [0]
    assert array[::-I64_MAX].to_list() == [4]
    assert array[::I64_MIN].to_list() == [4]
    assert array[::-1].to_list() == [4, 3, 2, 1, 0]


def test_duplicate_ellipsis_is_rejected_for_get_and_set():
    array = sdnp.ones((2, 2))

    with pytest.raises(IndexError, match="single ellipsis"):
        _ = array[..., ...]
    with pytest.raises(IndexError, match="single ellipsis"):
        array[..., ...] = 0


def test_explicit_dtype_is_applied_to_arrays_and_scalar_fill():
    source = sdnp.array([1, 2, 3])
    converted = sdnp.array(source, dtype=float)
    filled = sdnp.array(2, shape=(2,), dtype=float)

    assert converted.dtype is float
    assert converted.to_list() == [1.0, 2.0, 3.0]
    assert filled.dtype is float
    assert filled.to_list() == [2.0, 2.0]


def test_astype_narrowing_and_scalar_predicates_remain_supported():
    array = sdnp.array([0.0, 2.5])

    assert array.astype(bool).to_list() == [False, True]
    assert array.astype(int).to_list() == [0, 2]
    assert sdnp.equal(1, 1) is True
    assert sdnp.less(1.0, 2.0) is True
    assert sdnp.logical_and(1, 0) is False


def test_negative_and_positive_duplicate_axes_are_detected():
    array = sdnp.ones((2, 2))

    with pytest.raises(ValueError, match="duplicates"):
        sdnp.sum(array, axes=(0, -2))
    with pytest.raises(ValueError, match="permutation"):
        array.permute_axes((0, -2))


def test_empty_reductions_validate_only_axes_that_are_reduced():
    array = sdnp.zeros((0, 2))

    result = sdnp.min(array, axis=1)
    assert result.shape == [0]
    assert result.to_list() == []
    with pytest.raises(ValueError, match="empty"):
        sdnp.min(array, axis=0)


def test_all_nan_arg_reduction_checks_every_output_slice():
    array = sdnp.array([[1.0, float("nan")], [2.0, float("nan")]])

    with pytest.raises(ValueError, match="all-NaN"):
        sdnp.argmax(array, axis=0, nan_policy="ignore")


def test_join_list_inputs_and_promoted_shape_validation():
    array = sdnp.array([1, 2])

    assert sdnp.concatenate([array, array]).to_list() == [1, 2, 1, 2]
    with pytest.raises(ValueError, match="dimensions must match"):
        sdnp.vstack([sdnp.ones((1, 2)), sdnp.ones((1, 3))])


def test_clip_none_bounds_and_read_only_broadcast_assignment():
    array = sdnp.array([-2, 1, 5])

    assert sdnp.clip(array, None, 3).to_list() == [-2, 1, 3]
    assert sdnp.clip(array, 0, None).to_list() == [0, 1, 5]

    broadcast, _ = sdnp.meshgrid(
        sdnp.array([1.0, 2.0]), sdnp.array([10.0, 20.0])
    )
    with pytest.raises(ValueError, match="read-only|writable"):
        broadcast[0, 0] = 99


def test_integer_power_rejects_negative_exponents_for_scalars_and_arrays():
    with pytest.raises(ValueError, match="exponent"):
        sdnp.power(2, -1)
    with pytest.raises(ValueError, match="exponent"):
        sdnp.power(sdnp.array([2]), sdnp.array([-1]))
