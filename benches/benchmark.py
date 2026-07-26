#!/usr/bin/env python3
"""Run, list, and render sdnp-py versus NumPy benchmarks."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sdnp
from tqdm import tqdm

from benches.cases import (
    build_cases,
    exported_function_gaps,
    registered_functions,
    select_cases,
)
from benches.matrix import DTYPES, NDIMS, PROFILES, SIZE_ELEMENTS
from benches.measure import MeasureConfig, Measurement, measure
from benches.merge import merge_results, prepare_merge_state
from benches.report import RESULTS_DIR, render_outputs

SUMMARY_PATH = RESULTS_DIR / "benchmark.json"
RAW_PATH = RESULTS_DIR / "benchmark.csv"
SEED = 1729
NATIVE_SDNP = importlib.import_module("sdnp.sdnp")

CSV_FIELDS = (
    "run_id",
    "case_id",
    "function",
    "category",
    "dtype",
    "size",
    "ndim",
    "shape",
    "variant",
    "backend",
    "backend_order",
    "sample_index",
    "iterations",
    "elapsed_ns",
    "ns_per_call",
    "seed",
    "measured_at",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_text(
    completed: int,
    total: int,
    case_id: str,
    sdnp_ns: float,
    numpy_ns: float,
    ratio: float,
) -> str:
    """Format one benchmark result as two lines of at most 80 columns."""
    prefix = f"  {completed:>{len(str(total))}}/{total}  "
    case_line = textwrap.fill(
        case_id,
        width=80,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=True,
        break_on_hyphens=False,
    )
    timing_line = (
        f"{' ' * len(prefix)}sdnp {sdnp_ns:.1f} ns"
        f"  ·  NumPy {numpy_ns:.1f} ns"
        f"  ·  ratio {ratio:.3f}x"
    )
    return f"{case_line}\n{timing_line}"


def _distribution(measurement: Measurement) -> dict[str, float | int]:
    return measurement.distribution.to_dict()


def _metadata(run_id: str, started_at: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sdnp_version": getattr(sdnp, "__version__", "0.1.0"),
        "build_profile": getattr(NATIVE_SDNP, "__build_profile__", "unknown"),
        "optimized": bool(getattr(NATIVE_SDNP, "__optimized__", False)),
        "git_commit": _git_commit(),
        "seed": SEED,
    }


def _filters(args: argparse.Namespace) -> dict[str, Any]:
    filters = {
        "function": args.function,
        "case": args.case,
        "match": args.match,
        "category": args.category,
        "dtype": args.dtype,
        "size": args.size,
        "ndim": args.ndim,
    }
    if args.merge:
        filters["merge"] = True
    return filters


def _summary_payload(
    *,
    metadata: dict[str, Any],
    config: MeasureConfig,
    profile: str,
    filters: dict[str, Any],
    results: list[dict[str, Any]],
    raw_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": metadata,
        "config": {
            "profile": profile,
            "warmups": config.warmups,
            "samples": config.samples,
            "iterations": config.iterations,
            "target_sample_ms": config.target_sample_ms,
        },
        "filters": filters,
        "raw_csv": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "rows": raw_rows,
            "sha256": _sha256(RAW_PATH),
        },
        "results": results,
    }


def _write_raw(
    writer: csv.DictWriter,
    handle: Any,
    *,
    run_id: str,
    case: Any,
    backend: str,
    backend_order: int,
    measurement: Measurement,
) -> int:
    measured_at = datetime.now(timezone.utc).isoformat()
    for sample in measurement.raw:
        writer.writerow(
            {
                "run_id": run_id,
                "case_id": case.id,
                "function": case.function,
                "category": case.category,
                "dtype": case.dtype,
                "size": case.size,
                "ndim": case.ndim,
                "shape": "x".join(map(str, case.shape)),
                "variant": case.variant,
                "backend": backend,
                "backend_order": backend_order,
                "sample_index": sample.sample_index,
                "iterations": sample.iterations,
                "elapsed_ns": sample.elapsed_ns,
                "ns_per_call": f"{sample.ns_per_call:.9f}",
                "seed": SEED,
                "measured_at": measured_at,
            }
        )
    handle.flush()
    os.fsync(handle.fileno())
    return len(measurement.raw)


def run(args: argparse.Namespace) -> int:
    if args.list_functions:
        print("\n".join(registered_functions()))
        return 0
    if not getattr(NATIVE_SDNP, "__optimized__", False):
        raise SystemExit(
            "benchmarking requires an optimized extension; "
            "run `maturin develop --release` in sdnp-py"
        )

    gaps = exported_function_gaps(list(sdnp.__all__))
    if gaps:
        raise SystemExit(
            "benchmark registry misses exported functions: "
            + ", ".join(sorted(gaps))
        )

    all_cases = build_cases(args.profile)
    try:
        cases = select_cases(
            all_cases,
            functions=args.function,
            case_ids=args.case,
            matches=args.match,
            categories=args.category,
            dtypes=args.dtype,
            sizes=args.size,
            ndims=args.ndim,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.list:
        for case in cases:
            print(case.id)
        return 0

    config = MeasureConfig(
        warmups=args.warmups,
        samples=args.samples,
        iterations=args.iterations,
        target_sample_ms=args.target_sample_ms,
    )
    config.validate()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    metadata = _metadata(run_id, started_at)
    replaced_case_ids = {case.id for case in cases}
    kept_results: list[dict[str, Any]] = []
    preserved_rows: list[dict[str, str]] = []
    if args.merge:
        kept_results, preserved_rows, had_base = prepare_merge_state(
            summary_path=SUMMARY_PATH,
            raw_path=RAW_PATH,
            cases=cases,
            profile=args.profile,
        )
        if not had_base:
            print(
                "warning: --merge requested but no existing summary was found; "
                "writing a fresh result set",
                file=sys.stderr,
            )
        elif preserved_rows or kept_results:
            print(
                f"merge: keeping {len(kept_results)} existing case(s) and "
                f"replacing {len(replaced_case_ids)} case(s)",
                file=sys.stderr,
            )
    results: list[dict[str, Any]] = []
    raw_rows = len(preserved_rows)

    with RAW_PATH.open("w", newline="", encoding="utf-8") as raw_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        if preserved_rows:
            writer.writerows(preserved_rows)
        raw_handle.flush()
        os.fsync(raw_handle.fileno())

        total = len(cases)
        disable_progress = None if args.progress is None else not args.progress
        progress = tqdm(
            total=total,
            desc="benchmark",
            unit="case",
            ncols=80,
            dynamic_ncols=False,
            file=sys.stdout,
            disable=disable_progress,
            leave=True,
        )
        for index, case in enumerate(cases, start=1):
            progress.set_postfix_str(case.function, refresh=True)
            sd_task, np_task = case.prepare()
            if index % 2:
                order = (("sdnp", sd_task), ("numpy", np_task))
            else:
                order = (("numpy", np_task), ("sdnp", sd_task))

            measured: dict[str, Measurement] = {}
            for backend_order, (backend, task) in enumerate(order, start=1):
                measured[backend] = measure(task, config)
                raw_rows += _write_raw(
                    writer,
                    raw_handle,
                    run_id=run_id,
                    case=case,
                    backend=backend,
                    backend_order=backend_order,
                    measurement=measured[backend],
                )

            sd_distribution = measured["sdnp"].distribution
            np_distribution = measured["numpy"].distribution
            ratio = sd_distribution.median_ns / np_distribution.median_ns
            results.append(
                {
                    "case": case.metadata(),
                    "sdnp": _distribution(measured["sdnp"]),
                    "numpy": _distribution(measured["numpy"]),
                    "ratio_median": ratio,
                }
            )
            merged_results = (
                merge_results(
                    kept_results,
                    results,
                    profile=args.profile,
                    replaced_case_ids=replaced_case_ids,
                )
                if args.merge
                else results
            )
            payload = _summary_payload(
                metadata=metadata,
                config=config,
                profile=args.profile,
                filters=_filters(args),
                results=merged_results,
                raw_rows=raw_rows,
            )
            _atomic_json(SUMMARY_PATH, payload)
            progress.write(
                _result_text(
                    index,
                    total,
                    case.id,
                    sd_distribution.median_ns,
                    np_distribution.median_ns,
                    ratio,
                ),
                file=sys.stdout,
            )
            progress.update(1)
            del sd_task, np_task, measured
            gc.collect()
        progress.close()

    markdown_path, canvas_path = render_outputs(SUMMARY_PATH)
    print(f"raw:     {RAW_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"markdown:{markdown_path}")
    print(f"canvas:  {canvas_path}")
    return 0


def render(args: argparse.Namespace) -> int:
    markdown_path, canvas_path = render_outputs(args.summary)
    print(f"markdown:{markdown_path}")
    print(f"canvas:  {canvas_path}")
    return 0


def list_command(args: argparse.Namespace) -> int:
    if args.functions:
        print("\n".join(registered_functions()))
        return 0
    for case in build_cases(args.profile):
        print(case.id)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list functions or cases")
    list_parser.add_argument("--functions", action="store_true")
    list_parser.add_argument("--profile", choices=PROFILES, default="standard")
    list_parser.set_defaults(handler=list_command)

    run_parser = subparsers.add_parser("run", help="run selected benchmarks")
    run_parser.add_argument("--profile", choices=PROFILES, default="standard")
    run_parser.add_argument("--function", action="append", default=[])
    run_parser.add_argument("--case", action="append", default=[])
    run_parser.add_argument("--match", action="append", default=[])
    run_parser.add_argument("--category", action="append", default=[])
    run_parser.add_argument(
        "--dtype", action="append", choices=DTYPES, default=[]
    )
    run_parser.add_argument(
        "--size",
        action="append",
        choices=tuple(SIZE_ELEMENTS),
        default=[],
    )
    run_parser.add_argument(
        "--ndim",
        action="append",
        type=int,
        choices=NDIMS,
        default=[],
    )
    run_parser.add_argument("--warmups", type=int, default=3)
    run_parser.add_argument("--samples", type=int, default=11)
    run_parser.add_argument("--iterations", type=int)
    run_parser.add_argument("--target-sample-ms", type=float, default=50.0)
    run_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force-enable or disable the live progress bar",
    )
    run_parser.add_argument("--list-functions", action="store_true")
    run_parser.add_argument("--list", action="store_true")
    run_parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "merge this run into benches/results/benchmark.json and "
            "benchmark.csv instead of replacing the full result set"
        ),
    )
    run_parser.set_defaults(handler=run)

    render_parser = subparsers.add_parser(
        "render",
        help="regenerate Markdown and Canvas from summary JSON",
    )
    render_parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=SUMMARY_PATH,
    )
    render_parser.set_defaults(handler=render)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
