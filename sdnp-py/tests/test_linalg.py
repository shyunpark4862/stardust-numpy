import numpy as np
import pytest
import sdnp
from conftest import assert_matches


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ([[1, 2, 3], [4, 5, 6]], [2, -1, 3]),
        ([2, -1], [[1, 2, 3], [4, 5, 6]]),
        ([[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]),
    ],
)
def test_dot_vector_and_matrix_combinations(left, right):
    assert_matches(sdnp.dot(left, right), np.dot(left, right))


def test_dot_complex_does_not_conjugate():
    left = np.array([1 + 2j, 3 - 1j])
    right = np.array([2 - 1j, -4 + 3j])
    assert_matches(
        sdnp.dot(
            sdnp.array(left.tolist(), dtype=complex),
            sdnp.array(right.tolist(), dtype=complex),
        ),
        np.dot(left, right),
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ([[1, 2, 3], [4, 5, 6]], [2, -1, 3]),
        ([2, -1], [[1, 2, 3], [4, 5, 6]]),
        ([[[1, 2], [3, 4]]], [[[5, 6], [7, 8]], [[1, 0], [0, 1]]]),
    ],
)
def test_matmul_vector_matrix_and_batched_broadcast(left, right):
    assert_matches(sdnp.matmul(left, right), np.matmul(left, right))


def test_matmul_broadcasts_multiple_batch_dimensions():
    left = np.arange(2 * 1 * 3 * 4).reshape(2, 1, 3, 4)
    right = np.arange(1 * 5 * 4 * 2).reshape(1, 5, 4, 2)
    assert_matches(
        sdnp.matmul(sdnp.array(left.tolist()), sdnp.array(right.tolist())),
        np.matmul(left, right),
    )


def test_array_matmul_operator_matches_function():
    left = sdnp.array([[1.0, 2.0], [3.0, 4.0]])
    right = sdnp.array([[2.0], [-1.0]])
    assert_matches(left @ right, np.array([[0.0], [2.0]]))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ],
)
def test_vdot_flattens_inputs(left, right):
    assert_matches(sdnp.vdot(left, right), np.vdot(left, right))


def test_vdot_conjugates_first_complex_operand():
    left = np.array([[1 + 2j, 3 - 4j], [2j, -1 + 1j]])
    right = np.array([2 - 1j, 5j, 3 + 2j, 4 - 1j])
    assert_matches(
        sdnp.vdot(
            sdnp.array(left.tolist(), dtype=complex),
            sdnp.array(right.tolist(), dtype=complex),
        ),
        np.vdot(left, right),
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 2], [3, 4, 5]),
        ([[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.0], [2.0, 3.0]]),
        ([1 + 2j, 3 - 1j], [2j, 4 + 0j]),
    ],
)
def test_outer_flattens_inputs(left, right):
    assert_matches(sdnp.outer(left, right), np.outer(left, right))


@pytest.mark.parametrize(
    ("operation", "left", "right", "numpy_operation"),
    [
        (sdnp.dot, [True, False], [2, 3], np.dot),
        (sdnp.matmul, [[1, 2]], [[True], [False]], np.matmul),
        (sdnp.vdot, [1, 2], [True, False], np.vdot),
        (sdnp.outer, [True, False], [1, 2], np.outer),
        (sdnp.dot, [1, 2], [0.5, 1.5], np.dot),
        (sdnp.matmul, [[1.0, 2.0]], [[1j], [2j]], np.matmul),
        (sdnp.vdot, [1, 2], [1 + 1j, 2 - 1j], np.vdot),
        (sdnp.outer, [1.5, 2.5], [1 + 1j], np.outer),
    ],
)
def test_linalg_dtype_promotion(operation, left, right, numpy_operation):
    assert_matches(operation(left, right), numpy_operation(left, right))


@pytest.mark.parametrize("dtype", (bool, int, float, complex))
@pytest.mark.parametrize(
    ("offset", "axis1", "axis2"),
    [(0, 0, 1), (1, 0, 1), (-1, -2, -1), (0, 2, 0)],
)
def test_diagonal_and_trace_match_numpy(dtype, offset, axis1, axis2):
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    if dtype is bool:
        values = values % 3 == 0
    elif dtype is float:
        values = values.astype(float) / 2
    elif dtype is complex:
        values = values.astype(complex) + 1j * values[::-1]
    array = sdnp.array(values.tolist(), dtype=dtype)

    assert_matches(
        sdnp.diagonal(array, offset=offset, axis1=axis1, axis2=axis2),
        np.diagonal(values, offset=offset, axis1=axis1, axis2=axis2),
    )
    assert_matches(
        sdnp.trace(array, offset=offset, axis1=axis1, axis2=axis2),
        np.trace(values, offset=offset, axis1=axis1, axis2=axis2),
    )


def test_strided_linalg_inputs_match_numpy():
    np_left_base = np.arange(48).reshape(6, 8)
    np_right_base = np.arange(48).reshape(8, 6)
    sd_left_base = sdnp.arange(48).reshape((6, 8))
    sd_right_base = sdnp.arange(48).reshape((8, 6))
    np_left = np_left_base[::-2, 1::2]
    np_right = np_right_base[1::2, ::-2]
    sd_left = sd_left_base[::-2, 1::2]
    sd_right = sd_right_base[1::2, ::-2]

    assert_matches(sdnp.matmul(sd_left, sd_right), np.matmul(np_left, np_right))
    assert_matches(sdnp.dot(sd_left, sd_right), np.dot(np_left, np_right))
    assert_matches(
        sdnp.vdot(sd_left, sd_left[::-1]), np.vdot(np_left, np_left[::-1])
    )
    assert_matches(sdnp.outer(sd_left, sd_right), np.outer(np_left, np_right))
    assert_matches(
        sdnp.diagonal(sd_left, offset=-1),
        np.diagonal(np_left, offset=-1),
    )
    assert_matches(sdnp.trace(sd_left, offset=1), np.trace(np_left, offset=1))


def test_empty_contractions_return_correct_shapes_and_zero_values():
    left = sdnp.zeros((2, 0), dtype=float)
    right = sdnp.zeros((0, 3), dtype=float)
    assert_matches(sdnp.dot(left, right), np.zeros((2, 3)))
    assert_matches(sdnp.matmul(left, right), np.zeros((2, 3)))
    assert_matches(sdnp.vdot(sdnp.zeros((0,), dtype=complex), []), 0j)
    assert_matches(
        sdnp.outer(sdnp.zeros((0,), dtype=int), [1, 2]),
        np.empty((0, 2), dtype=np.int64),
    )


@pytest.mark.parametrize("dtype", (bool, int, float, complex))
def test_empty_diagonal_and_trace(dtype):
    array = sdnp.zeros((2, 0), dtype=dtype)
    expected = np.zeros(
        (2, 0),
        dtype={
            bool: np.bool_,
            int: np.int64,
            float: np.float64,
            complex: np.complex128,
        }[dtype],
    )
    assert_matches(sdnp.diagonal(array), np.diagonal(expected))
    assert_matches(sdnp.trace(array), np.trace(expected))


@pytest.mark.parametrize(
    "operation", (sdnp.dot, sdnp.matmul, sdnp.vdot, sdnp.outer)
)
def test_contractions_reject_bool_bool_operands(operation):
    with pytest.raises(ValueError, match="dtype mismatch"):
        operation(
            sdnp.array([True, False], dtype=bool),
            sdnp.array([False, True], dtype=bool),
        )


def test_dot_rejects_invalid_rank_and_inner_shape():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        sdnp.dot(sdnp.zeros((1, 2, 3)), sdnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="inner dimensions"):
        sdnp.dot(sdnp.zeros((2, 3)), sdnp.zeros((4, 2)))


def test_matmul_rejects_scalar_shape_and_batch_errors():
    with pytest.raises(ValueError, match="0-D"):
        sdnp.matmul(2, [1, 2])
    with pytest.raises(ValueError, match="inner dimensions"):
        sdnp.matmul(sdnp.zeros((2, 3)), sdnp.zeros((4, 2)))
    with pytest.raises(ValueError, match="batch dimensions"):
        sdnp.matmul(sdnp.zeros((2, 3, 4)), sdnp.zeros((5, 4, 2)))


def test_vdot_rejects_unequal_flattened_sizes():
    with pytest.raises(ValueError, match="equal flattened sizes"):
        sdnp.vdot(sdnp.zeros((2, 3)), sdnp.zeros((5,)))


@pytest.mark.parametrize("operation", (sdnp.diagonal, sdnp.trace))
def test_diagonal_operations_validate_rank_and_axes(operation):
    with pytest.raises(ValueError, match="at least two dimensions"):
        operation(sdnp.array([1, 2, 3]))
    with pytest.raises(ValueError, match="different"):
        operation(sdnp.zeros((2, 3)), axis1=0, axis2=-2)
    with pytest.raises(IndexError, match="axis"):
        operation(sdnp.zeros((2, 3)), axis1=0, axis2=2)
    with pytest.raises(TypeError):
        operation(sdnp.zeros((2, 3)), axis1=0.5)
