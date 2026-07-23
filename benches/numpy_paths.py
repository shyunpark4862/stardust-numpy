"""NumPy timing counterparts for ``benches/paths.rs``."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

SIDE = 1024
ELEMENTS = SIDE * SIDE


def matrix(seed: int) -> np.ndarray:
    values = np.arange(ELEMENTS, dtype=np.float64)
    return (((values + seed) % 4096) * 0.25).reshape(SIDE, SIDE)


def add_c_order(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Add into one freshly allocated C-order output, matching ``Array::from_vec``."""
    out = np.empty(left.shape, dtype=np.float64, order="C")
    np.add(left, right, out=out)
    return out


def measure(operation: Callable[[], Any]) -> float:
    """Return median seconds per call with automatically calibrated batches."""
    for _ in range(3):
        operation()

    iterations = 1
    while True:
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        elapsed = time.perf_counter() - started
        if elapsed >= 0.25:
            break
        iterations *= 2

    samples = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(7):
            started = time.perf_counter()
            for _ in range(iterations):
                operation()
            samples.append((time.perf_counter() - started) / iterations)
    finally:
        if gc_enabled:
            gc.enable()
    return statistics.median(samples)


def measure_batched(
    setup: Callable[[], Any],
    operation: Callable[[Any], None],
) -> float:
    """Time only the operation while preparing fresh mutable inputs outside it."""
    samples = []
    for _ in range(200):
        state = setup()
        started = time.perf_counter_ns()
        operation(state)
        samples.append((time.perf_counter_ns() - started) / 1_000_000_000)
    return statistics.median(samples)


def main() -> None:
    left = matrix(0)
    right = matrix(17)
    left_t = left.T
    right_t = right.T
    a = left
    a_nonzero = matrix(1)

    cube_values = np.arange(ELEMENTS, dtype=np.float64)
    cube = ((cube_values % 2048) * 0.5).reshape(64, 128, 128)
    permuted = cube.transpose(1, 0, 2)

    denominator_values = np.arange(ELEMENTS, dtype=np.float64)
    denominator = (
        ((denominator_values + 17) % 4096 + 1) * 0.25
    ).reshape(SIDE, SIDE)
    transposed = a.T
    row = np.arange(1, SIDE + 1, dtype=np.float64).reshape(1, SIDE)
    reshape_shape = (512, 2048)

    product_values = np.arange(ELEMENTS, dtype=np.int64)
    product_input = (
        1.0 + (product_values % 7).astype(np.float64) * 0.000_001
    ).reshape(SIDE, SIDE)

    mask = (np.arange(ELEMENTS) % 4 == 0).reshape(SIDE, SIDE)
    condition = (np.arange(ELEMENTS) % 2 == 0).reshape(SIDE, SIDE)
    grid_x = np.arange(SIDE, dtype=np.float64)
    grid_y = np.arange(SIDE, dtype=np.float64) * 2.0
    fancy_len = ELEMENTS // 4
    positions = np.arange(fancy_len, dtype=np.int64)
    rows = positions % SIDE
    cols = (positions * 17) % SIDE

    target = np.s_[: SIDE // 2, :]
    source = np.s_[SIDE // 2 :, :]
    unshared_values = np.full((SIDE // 2, SIDE), 3.5, dtype=np.float64)

    timings = {
        "full_1024x1024_f64": measure(
            lambda: np.full((SIDE, SIDE), 3.5, dtype=np.float64)
        ),
        "zeros_1024x1024_f64": measure(
            lambda: np.zeros((SIDE, SIDE), dtype=np.float64)
        ),
        "ones_1024x1024_f64": measure(
            lambda: np.ones((SIDE, SIDE), dtype=np.float64)
        ),
        "eye_1024_f64": measure(lambda: np.eye(SIDE, dtype=np.float64)),
        "transpose_view_f64": measure(lambda: a.T),
        "reshape_contiguous_view_f64": measure(
            lambda: a.reshape(reshape_shape)
        ),
        "broadcast_to_view_f64": measure(
            lambda: np.broadcast_to(row, (SIDE, SIDE))
        ),
        "broadcast_arrays_f64": measure(lambda: np.broadcast_arrays(a, row)),
        "copy_contiguous_f64": measure(lambda: a.copy(order="C")),
        "copy_transposed_f64": measure(lambda: transposed.copy(order="C")),
        "to_vec_transposed_f64": measure(
            lambda: np.array(transposed, order="C", copy=True)
        ),
        "sum_axis_last_contiguous_f64": measure(lambda: left.sum(axis=1)),
        "sum_axis_first_fixed_stride_f64": measure(lambda: left.sum(axis=0)),
        "sum_multi_axis_general_f64": measure(
            lambda: permuted.sum(axis=(0, 2))
        ),
        "var_axis_last_contiguous_f64": measure(lambda: left.var(axis=1)),
        "var_axis_first_fixed_stride_f64": measure(lambda: left.var(axis=0)),
        "min_axis_first_fixed_stride_f64": measure(
            lambda: a_nonzero.min(axis=0)
        ),
        "prod_axis_last_f64": measure(lambda: product_input.prod(axis=1)),
        "min_axis_last_f64": measure(lambda: a_nonzero.min(axis=1)),
        "max_axis_last_f64": measure(lambda: a_nonzero.max(axis=1)),
        "mean_axis_last_f64": measure(lambda: a_nonzero.mean(axis=1)),
        "std_axis_last_f64": measure(lambda: a_nonzero.std(axis=1)),
        "argmax_axis_last_f64": measure(lambda: a_nonzero.argmax(axis=1)),
        "any_axis_last_f64": measure(lambda: a_nonzero.any(axis=1)),
        "cumprod_axis_last_f64": measure(
            lambda: product_input.cumprod(axis=1)
        ),
        "add_contiguous_contiguous_f64": measure(
            lambda: add_c_order(left, right)
        ),
        "subtract_contiguous_contiguous_f64": measure(
            lambda: np.subtract(left, right)
        ),
        "add_strided_strided_f64": measure(
            lambda: add_c_order(left_t, right_t)
        ),
        "subtract_strided_strided_f64": measure(
            lambda: np.subtract(left_t, right_t)
        ),
        "negative_contiguous_f64": measure(lambda: np.negative(a_nonzero)),
        "absolute_contiguous_f64": measure(lambda: np.absolute(a_nonzero)),
        "isnan_contiguous_f64": measure(lambda: np.isnan(a_nonzero)),
        "multiply_row_broadcast_f64": measure(
            lambda: np.multiply(a_nonzero, row)
        ),
        "divide_contiguous_f64": measure(
            lambda: np.divide(a_nonzero, denominator)
        ),
        "trunc_divide_contiguous_f64": measure(
            lambda: np.trunc(np.divide(a_nonzero, denominator))
        ),
        "remainder_contiguous_f64": measure(
            lambda: np.remainder(a_nonzero, denominator)
        ),
        "power_contiguous_f64": measure(
            lambda: np.power(a_nonzero, denominator)
        ),
        "greater_contiguous_f64": measure(
            lambda: np.greater(a_nonzero, denominator)
        ),
        "cumsum_axis_last_contiguous_f64": measure(
            lambda: left.cumsum(axis=1)
        ),
        "cumsum_axis_first_strided_f64": measure(lambda: left.cumsum(axis=0)),
        "concatenate_axis0_contiguous_f64": measure(
            lambda: np.concatenate((left, right), axis=0)
        ),
        "concatenate_axis0_strided_f64": measure(
            lambda: np.concatenate((left_t, right_t), axis=0)
        ),
        "where_contiguous_f64": measure(
            lambda: np.where(condition, left, right)
        ),
        "where_strided_f64": measure(
            lambda: np.where(condition.T, left_t, right_t)
        ),
        "sort_axis_last_contiguous_f64": measure(
            lambda: np.sort(left, axis=-1, kind="stable")
        ),
        "sort_axis_last_strided_f64": measure(
            lambda: np.sort(left_t, axis=-1, kind="stable")
        ),
        "meshgrid_1024x1024_view_f64": measure(
            lambda: np.meshgrid(grid_x, grid_y, indexing="xy", copy=False)
        ),
        "fancy_gather_multidim_f64": measure(lambda: left[rows, cols]),
        "gather_basic_half_view_f64": measure(lambda: a[: SIDE // 2, :]),
        "gather_reverse_view_f64": measure(lambda: a[::-1, :]),
        "gather_boolean_mask_f64": measure(lambda: a[mask]),
        "scatter_array_f64/unshared_rhs": measure_batched(
            lambda: (left.copy(), unshared_values),
            lambda state: state[0].__setitem__(target, state[1]),
        ),
        "scatter_array_f64/shared_rhs": measure_batched(
            lambda: (lambda destination: (destination, destination[source]))(
                left.copy()
            ),
            lambda state: state[0].__setitem__(target, state[1]),
        ),
        "scatter_scalar_strided_f64": measure_batched(
            lambda: a.copy(),
            lambda destination: destination[:, ::2].fill(3.5),
        ),
        "scatter_scalar_fancy_f64": measure_batched(
            lambda: a.copy(),
            lambda destination: destination.__setitem__((rows, cols), 3.5),
        ),
    }

    print(
        json.dumps(
            {
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "machine": platform.machine(),
                "timings_seconds": timings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
