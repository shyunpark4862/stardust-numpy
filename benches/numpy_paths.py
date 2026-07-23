#!/usr/bin/env python3
"""NumPy timing counterparts for ``benches/paths.rs``."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

SMALL_SIDE = 32
MEDIUM_SIDE = 256
SIDE = 1024
ELEMENTS = SIDE * SIDE

Benchmark = Callable[[], float]


def shaped_f64(shape: tuple[int, ...], seed: int) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float64)
    return (((values + seed) % 4096) * 0.25).reshape(shape)


def matrix(seed: int) -> np.ndarray:
    return shaped_f64((SIDE, SIDE), seed)


def binary_c_order(
    operation: np.ufunc, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Run a binary ufunc into one C-order output, matching ``Array::from_vec``."""
    shape = np.broadcast_shapes(left.shape, right.shape)
    dtype = np.result_type(left.dtype, right.dtype)
    out = np.empty(shape, dtype=dtype, order="C")
    operation(left, right, out=out)
    return out


def concatenate_c_order(
    arrays: Sequence[np.ndarray], axis: int
) -> np.ndarray:
    """Concatenate directly into one C-order output without an intermediate."""
    normalized_axis = axis % arrays[0].ndim
    shape = list(arrays[0].shape)
    shape[normalized_axis] = sum(a.shape[normalized_axis] for a in arrays)
    out = np.empty(shape, dtype=np.result_type(*arrays), order="C")

    start = 0
    for array in arrays:
        stop = start + array.shape[normalized_axis]
        destination = [slice(None)] * out.ndim
        destination[normalized_axis] = slice(start, stop)
        np.copyto(out[tuple(destination)], array)
        start = stop
    return out


def where_c_order(
    condition: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Materialize ``where`` in C order, copying only if NumPy chooses otherwise."""
    out = np.where(condition, left, right)
    return out if out.flags.c_contiguous else np.ascontiguousarray(out)


def sort_c_order(array: np.ndarray, axis: int | None) -> np.ndarray:
    """Copy to C order, then stable-sort in place like the Rust implementation."""
    out = np.array(array, order="C", copy=True)
    out.sort(axis=axis, kind="stable")
    return out


def divide_i64(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Match Rust's integer ``/`` for the non-negative benchmark fixture."""
    out = np.empty(left.shape, dtype=np.int64, order="C")
    np.floor_divide(left, right, out=out)
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


def build_benchmarks() -> dict[str, Benchmark]:
    left = matrix(0)
    right = matrix(17)
    left_t = left.T
    right_t = right.T
    a = left
    a_nonzero = matrix(1)

    small_left = shaped_f64((SMALL_SIDE, SMALL_SIDE), 0)
    small_right = shaped_f64((SMALL_SIDE, SMALL_SIDE), 17)
    medium_left = shaped_f64((MEDIUM_SIDE, MEDIUM_SIDE), 0)
    medium_right = shaped_f64((MEDIUM_SIDE, MEDIUM_SIDE), 17)
    one_d_left = shaped_f64((ELEMENTS,), 0)
    one_d_right = shaped_f64((ELEMENTS,), 17)
    three_d_left = shaped_f64((64, 128, 128), 0)
    three_d_right = shaped_f64((64, 128, 128), 17)

    cube_values = np.arange(ELEMENTS, dtype=np.float64)
    cube = ((cube_values % 2048) * 0.5).reshape(64, 128, 128)
    permuted = cube.transpose(1, 0, 2)

    denominator_values = np.arange(ELEMENTS, dtype=np.float64)
    denominator = (
        ((denominator_values + 17) % 4096 + 1) * 0.25
    ).reshape(SIDE, SIDE)
    row = np.arange(1, SIDE + 1, dtype=np.float64).reshape(1, SIDE)
    column = np.arange(1, SIDE + 1, dtype=np.float64).reshape(SIDE, 1)
    scalar = np.array([[2.0]], dtype=np.float64)
    reshape_shape = (512, 2048)

    integer_left = (
        np.arange(ELEMENTS, dtype=np.int64) % 4096
    ).reshape(SIDE, SIDE)
    integer_denominator = (
        np.arange(ELEMENTS, dtype=np.int64) % 31 + 1
    ).reshape(SIDE, SIDE)

    product_values = np.arange(ELEMENTS, dtype=np.int64)
    product_input = (
        1.0 + (product_values % 7).astype(np.float64) * 0.000_001
    ).reshape(SIDE, SIDE)

    mask = (np.arange(ELEMENTS) % 4 == 0).reshape(SIDE, SIDE)
    condition = (np.arange(ELEMENTS) % 2 == 0).reshape(SIDE, SIDE)
    selection_scalar = np.array([[3.5]], dtype=np.float64)
    grid_x = np.arange(SIDE, dtype=np.float64)
    grid_y = np.arange(SIDE, dtype=np.float64) * 2.0
    fancy_len = ELEMENTS // 4
    positions = np.arange(fancy_len, dtype=np.int64)
    rows = positions % SIDE
    cols = (positions * 17) % SIDE

    target = np.s_[: SIDE // 2, :]
    source = np.s_[SIDE // 2 :, :]
    unshared_values = np.full((SIDE // 2, SIDE), 3.5, dtype=np.float64)
    strided_values = np.full((SIDE, SIDE // 2), 3.5, dtype=np.float64)

    return {
        "full_1024x1024_f64": lambda: measure(
            lambda: np.full((SIDE, SIDE), 3.5, dtype=np.float64)
        ),
        "zeros_1024x1024_f64": lambda: measure(
            lambda: np.zeros((SIDE, SIDE), dtype=np.float64)
        ),
        "ones_1024x1024_f64": lambda: measure(
            lambda: np.ones((SIDE, SIDE), dtype=np.float64)
        ),
        "eye_1024_f64": lambda: measure(
            lambda: np.eye(SIDE, dtype=np.float64)
        ),
        "transpose_view_f64": lambda: measure(lambda: a.T),
        "reshape_contiguous_view_f64": lambda: measure(
            lambda: a.reshape(reshape_shape)
        ),
        "broadcast_to_view_f64": lambda: measure(
            lambda: np.broadcast_to(row, (SIDE, SIDE))
        ),
        "broadcast_arrays_f64": lambda: measure(
            lambda: np.broadcast_arrays(a, row)
        ),
        "copy_contiguous_f64": lambda: measure(lambda: a.copy(order="C")),
        "copy_transposed_f64": lambda: measure(
            lambda: left_t.copy(order="C")
        ),
        "to_vec_transposed_f64": lambda: measure(
            lambda: np.array(left_t, order="C", copy=True)
        ),
        "sum_axis_last_contiguous_f64": lambda: measure(
            lambda: left.sum(axis=1)
        ),
        "sum_axis_first_fixed_stride_f64": lambda: measure(
            lambda: left.sum(axis=0)
        ),
        "sum_total_contiguous_f64": lambda: measure(lambda: left.sum()),
        "sum_multi_axis_general_f64": lambda: measure(
            lambda: permuted.sum(axis=(0, 2))
        ),
        "var_axis_last_contiguous_f64": lambda: measure(
            lambda: left.var(axis=1)
        ),
        "var_axis_first_fixed_stride_f64": lambda: measure(
            lambda: left.var(axis=0)
        ),
        "min_axis_first_fixed_stride_f64": lambda: measure(
            lambda: a_nonzero.min(axis=0)
        ),
        "prod_axis_last_f64": lambda: measure(
            lambda: product_input.prod(axis=1)
        ),
        "min_axis_last_f64": lambda: measure(
            lambda: a_nonzero.min(axis=1)
        ),
        "max_axis_last_f64": lambda: measure(
            lambda: a_nonzero.max(axis=1)
        ),
        "max_axis_first_fixed_stride_f64": lambda: measure(
            lambda: a_nonzero.max(axis=0)
        ),
        "mean_axis_last_f64": lambda: measure(
            lambda: a_nonzero.mean(axis=1)
        ),
        "std_axis_last_f64": lambda: measure(
            lambda: a_nonzero.std(axis=1)
        ),
        "argmax_axis_last_f64": lambda: measure(
            lambda: a_nonzero.argmax(axis=1)
        ),
        "argmin_axis_last_f64": lambda: measure(
            lambda: a_nonzero.argmin(axis=1)
        ),
        "any_axis_last_f64": lambda: measure(
            lambda: a_nonzero.any(axis=1)
        ),
        "all_axis_last_f64": lambda: measure(
            lambda: a_nonzero.all(axis=1)
        ),
        "cumprod_axis_last_f64": lambda: measure(
            lambda: product_input.cumprod(axis=1)
        ),
        "add_contiguous_contiguous_32x32_f64": lambda: measure(
            lambda: binary_c_order(np.add, small_left, small_right)
        ),
        "add_contiguous_contiguous_256x256_f64": lambda: measure(
            lambda: binary_c_order(np.add, medium_left, medium_right)
        ),
        "add_contiguous_contiguous_f64": lambda: measure(
            lambda: binary_c_order(np.add, left, right)
        ),
        "add_contiguous_contiguous_1m_1d_f64": lambda: measure(
            lambda: binary_c_order(np.add, one_d_left, one_d_right)
        ),
        "add_contiguous_contiguous_64x128x128_f64": lambda: measure(
            lambda: binary_c_order(np.add, three_d_left, three_d_right)
        ),
        "subtract_contiguous_contiguous_f64": lambda: measure(
            lambda: binary_c_order(np.subtract, left, right)
        ),
        "add_strided_strided_f64": lambda: measure(
            lambda: binary_c_order(np.add, left_t, right_t)
        ),
        "add_contiguous_strided_f64": lambda: measure(
            lambda: binary_c_order(np.add, left, right_t)
        ),
        "subtract_strided_strided_f64": lambda: measure(
            lambda: binary_c_order(np.subtract, left_t, right_t)
        ),
        "negative_contiguous_f64": lambda: measure(
            lambda: np.negative(a_nonzero)
        ),
        "absolute_contiguous_f64": lambda: measure(
            lambda: np.absolute(a_nonzero)
        ),
        "isnan_contiguous_f64": lambda: measure(
            lambda: np.isnan(a_nonzero)
        ),
        "multiply_row_broadcast_f64": lambda: measure(
            lambda: binary_c_order(np.multiply, a_nonzero, row)
        ),
        "multiply_column_broadcast_f64": lambda: measure(
            lambda: binary_c_order(np.multiply, a_nonzero, column)
        ),
        "multiply_scalar_broadcast_f64": lambda: measure(
            lambda: binary_c_order(np.multiply, a_nonzero, scalar)
        ),
        "add_i64_f64_contiguous": lambda: measure(
            lambda: binary_c_order(np.add, integer_left, a_nonzero)
        ),
        "divide_i64_contiguous": lambda: measure(
            lambda: divide_i64(integer_left, integer_denominator)
        ),
        "divide_contiguous_f64": lambda: measure(
            lambda: binary_c_order(np.divide, a_nonzero, denominator)
        ),
        "trunc_divide_contiguous_f64": lambda: measure(
            lambda: np.trunc(binary_c_order(np.divide, a_nonzero, denominator))
        ),
        "remainder_contiguous_f64": lambda: measure(
            lambda: binary_c_order(np.remainder, a_nonzero, denominator)
        ),
        "power_contiguous_f64": lambda: measure(
            lambda: binary_c_order(np.power, a_nonzero, denominator)
        ),
        "greater_contiguous_f64": lambda: measure(
            lambda: binary_c_order(np.greater, a_nonzero, denominator)
        ),
        "cumsum_axis_last_contiguous_f64": lambda: measure(
            lambda: left.cumsum(axis=1)
        ),
        "cumsum_axis_first_strided_f64": lambda: measure(
            lambda: left.cumsum(axis=0)
        ),
        "cumprod_axis_first_strided_f64": lambda: measure(
            lambda: product_input.cumprod(axis=0)
        ),
        "concatenate_axis0_contiguous_f64": lambda: measure(
            lambda: concatenate_c_order((left, right), axis=0)
        ),
        "concatenate_axis0_strided_f64": lambda: measure(
            lambda: concatenate_c_order((left_t, right_t), axis=0)
        ),
        "concatenate_axis1_contiguous_f64": lambda: measure(
            lambda: concatenate_c_order((left, right), axis=1)
        ),
        "concatenate_axis1_strided_f64": lambda: measure(
            lambda: concatenate_c_order((left_t, right_t), axis=1)
        ),
        "stack_axis0_contiguous_f64": lambda: measure(
            lambda: np.stack((left, right), axis=0)
        ),
        "where_contiguous_f64": lambda: measure(
            lambda: where_c_order(condition, left, right)
        ),
        "where_strided_f64": lambda: measure(
            lambda: where_c_order(condition.T, left_t, right_t)
        ),
        "where_scalar_broadcast_f64": lambda: measure(
            lambda: where_c_order(condition, selection_scalar, right)
        ),
        "clip_contiguous_f64": lambda: measure(
            lambda: np.clip(left, 128.0, 896.0)
        ),
        "nonzero_bool_contiguous": lambda: measure(
            lambda: np.nonzero(condition)
        ),
        "sort_axis_last_contiguous_f64": lambda: measure(
            lambda: sort_c_order(left, axis=-1)
        ),
        "sort_axis_last_strided_f64": lambda: measure(
            lambda: sort_c_order(left_t, axis=-1)
        ),
        "argsort_axis_last_contiguous_f64": lambda: measure(
            lambda: np.argsort(left, axis=-1, kind="stable")
        ),
        "unique_flatten_f64": lambda: measure(lambda: np.unique(left)),
        "linspace_1m_f64": lambda: measure(
            lambda: np.linspace(0.0, 1.0, ELEMENTS, endpoint=True)
        ),
        "meshgrid_1024x1024_view_f64": lambda: measure(
            lambda: np.meshgrid(
                grid_x, grid_y, indexing="xy", copy=False
            )
        ),
        "fancy_gather_multidim_f64": lambda: measure(
            lambda: left[rows, cols]
        ),
        "gather_basic_half_view_f64": lambda: measure(
            lambda: a[: SIDE // 2, :]
        ),
        "gather_reverse_view_f64": lambda: measure(lambda: a[::-1, :]),
        "gather_boolean_mask_f64": lambda: measure(lambda: a[mask]),
        "scatter_array_f64/unshared_rhs": lambda: measure_batched(
            lambda: (left.copy(), unshared_values),
            lambda state: state[0].__setitem__(target, state[1]),
        ),
        "scatter_array_f64/shared_rhs": lambda: measure_batched(
            lambda: (lambda destination: (destination, destination[source]))(
                left.copy()
            ),
            lambda state: state[0].__setitem__(target, state[1]),
        ),
        "scatter_array_strided_basic_f64": lambda: measure_batched(
            lambda: left.copy(),
            lambda destination: destination.__setitem__(
                np.s_[:, ::2], strided_values
            ),
        ),
        "scatter_scalar_strided_f64": lambda: measure_batched(
            lambda: a.copy(),
            lambda destination: destination[:, ::2].fill(3.5),
        ),
        "scatter_scalar_fancy_f64": lambda: measure_batched(
            lambda: a.copy(),
            lambda destination: destination.__setitem__((rows, cols), 3.5),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy counterparts for SDNP Criterion paths."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="exact benchmark path(s) to run",
    )
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        metavar="TEXT",
        help="run paths containing TEXT; repeat for OR matching",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered benchmark paths and exit",
    )
    return parser.parse_args()


def select_benchmarks(
    benchmarks: dict[str, Benchmark],
    exact: Sequence[str],
    matches: Sequence[str],
) -> list[str]:
    unknown = sorted(set(exact) - benchmarks.keys())
    if unknown:
        joined = ", ".join(unknown)
        raise SystemExit(f"unknown benchmark path(s): {joined}; use --list")

    if not exact and not matches:
        return sorted(benchmarks)

    selected = set(exact)
    selected.update(
        name
        for name in benchmarks
        if any(fragment in name for fragment in matches)
    )
    if not selected:
        raise SystemExit("no benchmark paths matched; use --list")
    return sorted(selected)


def main() -> None:
    args = parse_args()
    benchmarks = build_benchmarks()

    if args.list:
        print("\n".join(sorted(benchmarks)))
        return

    selected = select_benchmarks(benchmarks, args.paths, args.match)
    timings = {name: benchmarks[name]() for name in selected}
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
