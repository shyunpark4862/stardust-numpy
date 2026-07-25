"""Low-level calibrated timing and distribution statistics."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MeasureConfig:
    warmups: int = 3
    samples: int = 11
    iterations: int | None = None
    target_sample_ms: float = 50.0

    def validate(self) -> None:
        if self.warmups < 0:
            raise ValueError("warmups must be non-negative")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if self.iterations is not None and self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.target_sample_ms <= 0:
            raise ValueError("target-sample-ms must be positive")


@dataclass(frozen=True)
class BackendTask:
    """Create one timed callable; fresh tasks are rebuilt for each invocation."""

    factory: Callable[[], Callable[[], Any]]
    fresh: bool = False


@dataclass(frozen=True)
class RawSample:
    sample_index: int
    iterations: int
    elapsed_ns: int
    ns_per_call: float


@dataclass(frozen=True)
class Distribution:
    p25_ns: float
    median_ns: float
    p75_ns: float
    mean_ns: float
    minimum_ns: float
    maximum_ns: float
    samples: int
    iterations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class Measurement:
    distribution: Distribution
    raw: tuple[RawSample, ...]


def percentile(values: list[float], percentile_value: float) -> float:
    """Linear-interpolated percentile with deterministic endpoint behavior."""
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _run_batch(operation: Callable[[], Any], iterations: int) -> int:
    started = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    return time.perf_counter_ns() - started


def _run_fresh_batch(task: BackendTask, iterations: int) -> int:
    operations = [task.factory() for _ in range(iterations)]
    started = time.perf_counter_ns()
    for operation in operations:
        operation()
    return time.perf_counter_ns() - started


def _calibrate(task: BackendTask, config: MeasureConfig) -> int:
    if config.iterations is not None:
        return config.iterations
    if task.fresh:
        return 1

    operation = task.factory()
    target_ns = int(config.target_sample_ms * 1_000_000)
    iterations = 1
    while iterations < 1 << 30:
        elapsed = _run_batch(operation, iterations)
        if elapsed >= target_ns:
            return iterations
        if elapsed == 0:
            iterations *= 10
        else:
            estimate = max(2, min(10, target_ns // elapsed))
            iterations *= estimate
    return iterations


def measure(task: BackendTask, config: MeasureConfig) -> Measurement:
    config.validate()
    iterations = _calibrate(task, config)

    persistent = None if task.fresh else task.factory()
    for _ in range(config.warmups):
        if task.fresh:
            _run_fresh_batch(task, iterations)
        else:
            if persistent is None:
                raise RuntimeError("non-fresh benchmark has no operation")
            _run_batch(persistent, iterations)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    raw: list[RawSample] = []
    try:
        for sample_index in range(config.samples):
            if task.fresh:
                elapsed = _run_fresh_batch(task, iterations)
            else:
                if persistent is None:
                    raise RuntimeError("non-fresh benchmark has no operation")
                elapsed = _run_batch(persistent, iterations)
            raw.append(
                RawSample(
                    sample_index=sample_index,
                    iterations=iterations,
                    elapsed_ns=elapsed,
                    ns_per_call=elapsed / iterations,
                )
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    values = [sample.ns_per_call for sample in raw]
    distribution = Distribution(
        p25_ns=percentile(values, 0.25),
        median_ns=statistics.median(values),
        p75_ns=percentile(values, 0.75),
        mean_ns=statistics.fmean(values),
        minimum_ns=min(values),
        maximum_ns=max(values),
        samples=len(values),
        iterations=iterations,
    )
    return Measurement(distribution, tuple(raw))
