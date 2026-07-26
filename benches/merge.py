"""Merge partial benchmark runs into an existing summary and raw CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benches.cases import BenchmarkCase, build_cases


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def case_order(profile: str) -> dict[str, int]:
    return {case.id: index for index, case in enumerate(build_cases(profile))}


def order_results(
    results: list[dict[str, Any]], profile: str
) -> list[dict[str, Any]]:
    order = case_order(profile)
    return sorted(
        results,
        key=lambda result: order.get(result["case"]["id"], len(order)),
    )


def merge_results(
    existing: list[dict[str, Any]],
    updated: list[dict[str, Any]],
    *,
    profile: str,
    replaced_case_ids: set[str],
) -> list[dict[str, Any]]:
    kept = [
        result
        for result in existing
        if result["case"]["id"] not in replaced_case_ids
    ]
    return order_results(kept + updated, profile)


def preserved_csv_rows(
    path: Path, replaced_case_ids: set[str]
) -> list[dict[str, str]]:
    if not path.exists() or not replaced_case_ids:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            row
            for row in reader
            if row["case_id"] not in replaced_case_ids
        ]


def prepare_merge_state(
    *,
    summary_path: Path,
    raw_path: Path,
    cases: list[BenchmarkCase],
    profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], bool]:
    """Return kept results, preserved CSV rows, and whether a base summary existed."""
    replaced_case_ids = {case.id for case in cases}
    payload = load_summary(summary_path)
    if payload is None:
        return [], preserved_csv_rows(raw_path, replaced_case_ids), False

    existing_results = payload.get("results", [])
    if not isinstance(existing_results, list):
        existing_results = []

    kept_results = [
        result
        for result in existing_results
        if result["case"]["id"] not in replaced_case_ids
    ]
    preserved_rows = preserved_csv_rows(raw_path, replaced_case_ids)
    return kept_results, preserved_rows, True
