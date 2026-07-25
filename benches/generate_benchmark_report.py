#!/usr/bin/env python3
"""Generate benches/BENCHMARK_REPORT.md from Criterion + NumPy benchmark outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benches" / "BENCHMARK_REPORT.md"


def load_render_module():
    spec = importlib.util.spec_from_file_location(
        "render_benchmark_canvas",
        ROOT / "benches" / "render_benchmark_canvas.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmt_us(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def ratio_cell(ratio: float) -> str:
    if ratio <= 1.0:
        return f"**{ratio:.2f}×** ✓"
    return f"{ratio:.2f}×"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def collect_rows(rbc, rust: dict[str, float], np_secs: dict[str, float]):
    rows = []
    for key, (group, name) in rbc.METADATA.items():
        if key not in rust or key not in np_secs:
            continue
        sdnp = rust[key]
        numpy = np_secs[key] * 1e6
        rows.append(
            {
                "key": key,
                "group": group,
                "name": name,
                "sdnp": sdnp,
                "numpy": numpy,
                "ratio": sdnp / numpy,
            }
        )
    return rows


OPTIMIZATION_TEMPLATE = ROOT / "benches" / "optimization_techniques.md"


def load_optimization_section() -> str:
    if not OPTIMIZATION_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Missing optimization template: {OPTIMIZATION_TEMPLATE}"
        )
    return OPTIMIZATION_TEMPLATE.read_text(encoding="utf-8").strip()


def build_report(
    rows: list[dict],
    numpy_meta: dict,
    rust_log: Path,
    numpy_json: Path,
    optimization_section: str,
) -> str:
    wins = sum(1 for r in rows if r["ratio"] <= 1.0)
    slower = sorted(rows, key=lambda r: -r["ratio"])[:10]
    faster = sorted(rows, key=lambda r: r["ratio"])[:10]

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    group_rows = []
    for group in sorted(by_group):
        items = by_group[group]
        ratios = [r["ratio"] for r in items]
        group_rows.append(
            [
                group,
                str(len(items)),
                str(sum(1 for r in ratios if r <= 1.0)),
                f"{statistics.mean(ratios):.2f}×",
                f"{statistics.median(ratios):.2f}×",
            ]
        )

    slower_rows = [
        [
            r["group"],
            r["name"],
            fmt_us(r["sdnp"]),
            fmt_us(r["numpy"]),
            f"{r['ratio']:.2f}×",
        ]
        for r in slower
    ]

    faster_rows = [
        [
            r["group"],
            r["name"],
            fmt_us(r["sdnp"]),
            fmt_us(r["numpy"]),
            ratio_cell(r["ratio"]),
        ]
        for r in faster
    ]

    full_rows = []
    for r in sorted(rows, key=lambda x: (x["group"], x["name"])):
        full_rows.append(
            [
                r["group"],
                r["name"],
                fmt_us(r["sdnp"]),
                fmt_us(r["numpy"]),
                ratio_cell(r["ratio"]),
            ]
        )

    today = date.today().isoformat()
    machine = numpy_meta.get("machine", "unknown")
    numpy_ver = numpy_meta.get("numpy_version", "unknown")
    python_ver = numpy_meta.get("python_version", "unknown")

    return f"""# SDNP Paths 벤치마크 리포트

> 자동 생성: `{rust_log.relative_to(ROOT)}` + `{numpy_json.relative_to(ROOT)}`  
> 생성일: {today}  
> 재생성: `python benches/generate_benchmark_report.py --skip-run`

## 측정 환경

| 항목 | 값 |
|------|-----|
| 플랫폼 | {machine} (Apple Silicon) |
| NumPy | {numpy_ver} |
| Python | {python_ver} |
| Rust 벤치 | `cargo bench --bench paths` (Criterion) |
| NumPy 벤치 | `python benches/numpy_paths.py` |
| 비교 경로 수 | {len(rows)} (양쪽 모두 존재하는 키만) |

### 측정 방법론

- Rust와 NumPy 벤치는 **순차 실행**한다. 동시 실행 시 CPU 간섭으로 결과가 왜곡된다.
- 표의 시간은 Criterion **median** / NumPy **단일 측정값**(초 → µs 변환)이다.
- **비율** = SDNP ÷ NumPy. **1.00× 이하**이면 SDNP가 같거나 빠름(✓).
- View 계열은 allocator·복사 비용이 없어 µs 미만으로 매우 작게 나올 수 있다.

## Executive Summary

| 지표 | 값 |
|------|-----|
| 비교 경로 | {len(rows)} |
| SDNP 우세·동률 (≤ 1.00×) | **{wins}** / {len(rows)} ({100 * wins / len(rows):.0f}%) |
| SDNP 열세 (> 1.00×) | {len(rows) - wins} / {len(rows)} |

### 주요 결과

- **Reduction first axis:** prefix `TraversalSchedule` 덕분에 `sum/mean/prod · first axis`, `any/all · first axis`, `var/std · first axis`가 NumPy 대비 **0.28–0.97×**.
- **Reduction last axis:** `sum · last axis` **0.66×**, `var/std · last axis` **~0.40×** — suffix 8-lane + two-pass var가 효과적.
- **min/max:** f64/i64 last axis는 여전히 NumPy **~1.8×** 열세. first axis prefix는 f64 **~1.9×**, bool은 **0.77×**.
- **최대 격차:** `sum · multi-axis general` **5.02×**, `scatter array · shared RHS` **4.12×**, `sort · last axis contiguous` **2.90×**.
- **View·Spaces:** `meshgrid · 1024×1024 view` **0.06×**, broadcast/view 경로 전반 우세.

## 카테고리별 요약

{md_table(["카테고리", "경로 수", "SDNP ≤ NumPy", "평균 비율", "중앙값 비율"], group_rows)}

## SDNP 우세 Top 10 (비율 낮을수록 SDNP가 빠름)

{md_table(["카테고리", "경로", "SDNP (µs)", "NumPy (µs)", "비율"], faster_rows)}

## 개선 우선순위 Top 10 (SDNP가 느린 순)

{md_table(["카테고리", "경로", "SDNP (µs)", "NumPy (µs)", "비율"], slower_rows)}

## 전체 결과

{md_table(["카테고리", "경로", "SDNP (µs)", "NumPy (µs)", "비율"], full_rows)}

{optimization_section}

## 재현 방법

```bash
# 1) Rust (약 7분)
cargo bench --bench paths 2>&1 | tee benches/.bench-rust.log

# 2) NumPy (약 5분) — Rust 완료 후 실행
.venv/bin/python benches/numpy_paths.py
# → benches/.bench-numpy.json

# 3) 리포트 + 캔버스
python benches/generate_benchmark_report.py --skip-run
python benches/render_benchmark_canvas.py --skip-run \\
  --rust-log benches/.bench-rust.log \\
  --numpy-json benches/.bench-numpy.json
```

---

*이 문서는 `benches/generate_benchmark_report.py`로 생성된다. 벤치 데이터 갱신 후 `--skip-run`으로 표만 다시 만들 수 있다. 최적화 기법 본문은 `benches/optimization_techniques.md`에서 편집한다.*
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Use existing log/json instead of re-running benchmarks",
    )
    parser.add_argument(
        "--rust-log",
        type=Path,
        default=ROOT / "benches" / ".bench-rust.log",
    )
    parser.add_argument(
        "--numpy-json",
        type=Path,
        default=ROOT / "benches" / ".bench-numpy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT,
    )
    args = parser.parse_args()

    rbc = load_render_module()

    if not args.skip_run:
        print("Running Rust benchmarks...", file=sys.stderr)
        rust_text = rbc.run_rust_bench()
        args.rust_log.write_text(rust_text)
        print("Running NumPy benchmarks...", file=sys.stderr)
        numpy_data = rbc.run_numpy_bench()
        args.numpy_json.write_text(json.dumps(numpy_data, indent=2))
    else:
        rust_text = args.rust_log.read_text()
        numpy_data = json.loads(args.numpy_json.read_text())

    rust = rbc.parse_criterion_log(rust_text)
    rows = collect_rows(rbc, rust, numpy_data["timings_seconds"])
    if not rows:
        print("No matched benchmark paths.", file=sys.stderr)
        return 1

    report = build_report(
        rows,
        numpy_data,
        args.rust_log,
        args.numpy_json,
        load_optimization_section(),
    )
    args.output.write_text(report, encoding="utf-8")
    wins = sum(1 for r in rows if r["ratio"] <= 1.0)
    print(
        f"Wrote {args.output} ({len(rows)} paths, SDNP ≤ NumPy: {wins}/{len(rows)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
