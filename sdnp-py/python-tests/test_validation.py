"""Python-boundary validation (TypeError / ValueError / IndexError)."""

import pytest

import sdnp as np


def test_arange_zero_step():
    with pytest.raises(ValueError, match="step must not be zero"):
        np.arange(0, 10, 0)


def test_linspace_non_finite():
    with pytest.raises(ValueError, match="finite"):
        np.linspace(float("nan"), 1.0, 5)


def test_logspace_invalid_base():
    with pytest.raises(ValueError, match="base"):
        np.logspace(0, 1, 5, base=0)


def test_geomspace_sign_mismatch():
    with pytest.raises(ValueError, match="same sign"):
        np.geomspace(-1.0, 1.0, 5)


def test_sum_axis_out_of_bounds():
    a = np.array([[1, 2], [3, 4]])
    with pytest.raises(IndexError, match="out of bounds"):
        np.sum(a, axis=5)


def test_sum_duplicate_axes():
    a = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="duplicate"):
        np.sum(a, axes=[0, 0])


def test_sum_axis_and_axes_exclusive():
    a = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="both axis and axes"):
        np.sum(a, axis=0, axes=[0])


def test_concatenate_empty():
    with pytest.raises(ValueError, match="at least one"):
        np.concatenate(())


def test_stack_shape_mismatch():
    a = np.array([1, 2])
    b = np.array([[3, 4]])
    with pytest.raises(ValueError, match="same shape"):
        np.stack((a, b))


def test_slice_zero_step():
    a = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="step cannot be zero"):
        a[0:3:0]


def test_reshape_infer_minus_one():
    a = np.array([1, 2, 3, 4, 5, 6])
    assert a.reshape((2, -1)).shape == [2, 3]


def test_reshape_two_minus_one():
    a = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="only one"):
        a.reshape((-1, -1))


def test_where_requires_bool_condition():
    with pytest.raises(TypeError, match="bool"):
        np.where(np.array([1, 2]), 0, 1)


def test_meshgrid_bad_indexing():
    a = np.array([1, 2])
    with pytest.raises(ValueError, match="indexing"):
        np.meshgrid(a, indexing="bad")


def test_nditer_type_error():
    with pytest.raises(TypeError, match="nditer"):
        np.nditer((1, 2))


def test_explicit_dtype_is_not_silently_ignored():
    source = np.array([1, 2, 3])
    converted = np.array(source, dtype=float)
    assert converted.dtype is float
    assert converted.to_list() == [1.0, 2.0, 3.0]

    filled = np.array(2, shape=(2,), dtype=float)
    assert filled.dtype is float
    assert filled.to_list() == [2.0, 2.0]


def test_astype_supports_narrowing_conversions():
    a = np.array([0.0, 2.5])
    assert a.astype(bool).to_list() == [False, True]
    assert a.astype(int).to_list() == [0, 2]


def test_linalg_validates_rank_and_contraction_shapes():
    with pytest.raises(ValueError, match="0-D"):
        np.matmul(1, np.array([1]))
    with pytest.raises(ValueError, match="inner dimensions"):
        np.matmul(np.array([[1, 2]]), np.array([[1, 2]]))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        np.dot(np.ones((1, 1, 1)), np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="equal flattened sizes"):
        np.vdot(np.array([1, 2]), np.array([1]))


def test_diagonal_family_validates_rank_and_axes():
    cube = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        np.diag(cube)
    with pytest.raises(ValueError, match="different"):
        np.diagonal(cube, axis1=1, axis2=1)
    with pytest.raises(ValueError, match="at least two"):
        np.trace(np.array([1, 2]))


def test_meshgrid_requires_one_dimensional_inputs():
    with pytest.raises(ValueError, match="1-D"):
        np.meshgrid(np.ones((2, 2)))


def test_index_validation_precedes_core():
    a = np.array([[1, 2], [3, 4]])
    with pytest.raises(IndexError, match="single ellipsis"):
        _ = a[..., ...]
    with pytest.raises(IndexError, match="too many indices"):
        _ = a[0, 0, 0]
    with pytest.raises(TypeError, match="integer or boolean"):
        _ = a[np.array([0.0])]
    with pytest.raises(IndexError, match="boolean index shape"):
        _ = a[np.array([True, False, True])]


def test_empty_and_all_nan_reductions_are_checked_at_boundary():
    empty = np.zeros((0,))
    for reduction in (np.min, np.max, np.mean, np.var, np.std):
        with pytest.raises(ValueError, match="empty"):
            reduction(empty)
    with pytest.raises(ValueError, match="empty"):
        np.argmin(empty)
    with pytest.raises(ValueError, match="all-NaN"):
        np.argmax(np.array([float("nan")]), nan_policy="ignore")


def test_join_accepts_lists_and_validates_promoted_shapes():
    a = np.array([1, 2])
    assert np.concatenate([a, a]).to_list() == [1, 2, 1, 2]
    with pytest.raises(ValueError, match="dimensions must match"):
        np.vstack([np.ones((1, 2)), np.ones((1, 3))])


def test_arange_handles_i64_boundary_without_overflow():
    top = 2**63 - 1
    assert np.arange(top - 1, top, 2).to_list() == [top - 1]


@pytest.mark.parametrize("operation", [np.divide, np.trunc_divide, np.remainder])
def test_scalar_integer_zero_division_is_a_python_exception(operation):
    with pytest.raises(ZeroDivisionError):
        operation(1, 0)


def test_integer_power_validates_exponent_for_scalars_and_arrays():
    with pytest.raises(ValueError, match="exponent"):
        np.power(2, -1)
    with pytest.raises(ValueError, match="exponent"):
        np.power(np.array([2]), np.array([-1]))


def test_scalar_predicates_return_bool():
    assert np.equal(1, 1) is True
    assert np.less(1.0, 2.0) is True
    assert np.logical_and(1, 0) is False


def test_clip_accepts_one_sided_none_bounds():
    a = np.array([-2, 1, 5])
    assert np.clip(a, None, 3).to_list() == [-2, 1, 3]
    assert np.clip(a, 0, None).to_list() == [0, 1, 5]


def test_invalid_dtype_raises_type_error():
    with pytest.raises(TypeError, match="dtype"):
        np.zeros((2,), dtype=str)
