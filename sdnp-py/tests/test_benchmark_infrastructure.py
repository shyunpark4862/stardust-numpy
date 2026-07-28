import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sdnp

from benches.cases import (
    build_cases,
    exported_function_gaps,
    registered_functions,
    select_cases,
)
from benches.matrix import SIZE_ELEMENTS, shape_for
from benches.measure import BackendTask, MeasureConfig, measure, percentile
from benches.report import MARKDOWN_PATH, render_canvas, render_markdown


def test_registry_covers_every_exported_function():
    assert exported_function_gaps(list(sdnp.__all__)) == set()


def test_every_registered_function_builds_runnable_small_cases():
    cases = [case for case in build_cases("full") if case.size == "small"]
    assert {case.function for case in cases} == set(registered_functions())
    for case in cases:
        sdnp_task, numpy_task = case.prepare()
        sdnp_task.factory()()
        numpy_task.factory()()


def test_shape_matrix_covers_all_sizes_and_high_dimensions():
    for size, target in SIZE_ELEMENTS.items():
        for ndim in (1, 2, 3, 6):
            shape = shape_for(size, ndim)
            assert len(shape) == ndim
            assert all(dimension > 0 for dimension in shape)
            elements = 1
            for dimension in shape:
                elements *= dimension
            assert elements <= target


def test_function_and_matrix_filters_compose_as_intersection():
    selected = select_cases(
        build_cases("full"),
        functions=["add"],
        dtypes=["float64"],
        sizes=["small"],
        ndims=[6],
    )
    assert selected
    assert {case.function for case in selected} == {"add"}
    assert {case.dtype for case in selected} == {"float64"}
    assert {case.size for case in selected} == {"small"}
    assert {case.ndim for case in selected} == {6}


def test_percentiles_use_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.25) == 17.5
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.75) == 32.5


def test_measurement_supports_calibrated_and_explicit_iterations():
    task = BackendTask(lambda: lambda: 1 + 1)
    calibrated = measure(
        task,
        MeasureConfig(
            warmups=0,
            samples=2,
            target_sample_ms=0.01,
        ),
    )
    explicit = measure(
        task,
        MeasureConfig(
            warmups=1,
            samples=2,
            iterations=3,
        ),
    )
    assert calibrated.distribution.iterations >= 1
    assert explicit.distribution.iterations == 3
    assert len(calibrated.raw) == 2
    assert len(explicit.raw) == 2


def test_fresh_tasks_ignore_explicit_iterations():
    copies: list[int] = []

    def factory():
        copies.append(1)
        return lambda: None

    measurement = measure(
        BackendTask(factory, fresh=True),
        MeasureConfig(
            warmups=0,
            samples=2,
            iterations=100,
        ),
    )
    assert measurement.distribution.iterations == 1
    assert len(copies) == 2


def test_report_renderers_are_data_only():
    distribution = {
        "p25_ns": 10.0,
        "median_ns": 12.0,
        "p75_ns": 14.0,
        "mean_ns": 12.5,
        "minimum_ns": 9.0,
        "maximum_ns": 15.0,
        "samples": 3,
        "iterations": 2,
    }
    payload = {
        "metadata": {
            "run_id": "test",
            "started_at": "2026-01-01T00:00:00+00:00",
            "platform": "test-platform",
            "python_version": "3.12",
            "numpy_version": "2.0",
            "sdnp_version": "0.1.0",
            "build_profile": "release",
            "git_commit": "abc",
        },
        "config": {
            "profile": "smoke",
            "warmups": 1,
            "samples": 3,
            "iterations": 2,
            "target_sample_ms": 1.0,
        },
        "filters": {},
        "raw_csv": {
            "path": "benches/results/benchmark.csv",
            "rows": 6,
            "sha256": "deadbeef",
        },
        "results": [
            {
                "case": {
                    "id": "ufunc.add.float64.small.2d.default",
                    "function": "add",
                    "category": "ufunc",
                    "dtype": "float64",
                    "size": "small",
                    "ndim": 2,
                    "shape": [8, 8],
                    "variant": "default",
                },
                "sdnp": distribution,
                "numpy": distribution,
                "ratio_median": 1.0,
            }
        ],
    }
    markdown = render_markdown(payload)
    canvas = render_canvas(payload)
    assert "| Category | Cases | Median ratio | Mean ratio |" in markdown
    assert "Complete Results" not in markdown
    assert 'from "cursor/canvas"' in canvas
    assert '"sdnp mean", "NumPy mean", "Ratio"' in canvas
    assert "3× target" in canvas
    assert "sdnp p25" not in canvas
    assert "NumPy p75" not in canvas
    assert "원인" not in markdown
    assert "최적화 우선순위" not in canvas
    assert MARKDOWN_PATH == ROOT / "BENCHMARK.md"


def test_expansion_ops_use_side_limited_inputs_for_large_1d():
    side = math.isqrt(SIZE_ELEMENTS["large"])
    budget = SIZE_ELEMENTS["large"]
    for function in ("tril", "triu", "diag", "outer", "meshgrid"):
        case = select_cases(
            build_cases("full"),
            functions=[function],
            sizes=["large"],
            ndims=[1],
        )[0]
        sdnp_task, numpy_task = case.prepare()
        sdnp_result = sdnp_task.factory()()
        numpy_result = numpy_task.factory()()

        if function == "meshgrid":
            assert len(sdnp_result) == len(numpy_result) == 2
            for grid in sdnp_result:
                assert math.prod(grid.shape) == budget
            for grid in numpy_result:
                assert math.prod(grid.shape) == budget
            continue

        assert math.prod(sdnp_result.shape) == budget
        assert math.prod(numpy_result.shape) == budget
        assert side * side == budget
