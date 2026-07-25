#!/usr/bin/env python3
"""Merge Criterion + NumPy benchmark outputs into paths-benchmark-results.canvas.tsx."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = (
    Path.home()
    / ".cursor/projects/Users-sanghyunpark-Desktop-stardust-stardust-numpy/canvases"
    / "paths-benchmark-results.canvas.tsx"
)

from benchmark_manifest import DIAGNOSTIC_KEYS, METADATA

# SDNP µs before Phase 6 linalg optimization (2026-07-25 full run).
PHASE6_PRIOR_US: dict[str, float] = {
    "dot_1d_contiguous_1m_f64": 663.8,
    "matmul_contiguous_32x32_f64": 12.21,
    "matmul_contiguous_128x128_f64": 823.8,
    "matmul_contiguous_256x256_f64": 6213.5,
    "matmul_strided_strided_128x128_f64": 1037.7,
    "matmul_batched_8x64x64_f64": 758.3,
    "tril_1024x1024_contiguous_f64": 1016.5,
    "tril_1024x1024_strided_f64": 2346.5,
    "triu_1024x1024_contiguous_f64": 1001.3,
}

PROPAGATE_PRIOR_US: dict[str, float] = {
    "sum_axis_last_contiguous_f64": 80.167,
    "sum_axis_first_fixed_stride_f64": 128.15,
    "sum_multi_axis_general_f64": 575.92,
    "prod_axis_last_f64": 762.63,
    "prod_axis_first_fixed_stride_f64": 127.81,
    "var_axis_last_contiguous_f64": 242.31,
    "min_axis_last_f64": 155.19,
    "argmin_axis_last_f64": 938.38,
    "cumsum_axis_last_contiguous_f64": 1206.1,
    "cumsum_axis_first_strided_f64": 6994.9,
}

TIME_RE = re.compile(
    r"time:\s+\[[\d.]+\s+\S+\s+([\d.]+)\s+(\S+)"
)
NAME_RE = re.compile(r"^[\w/.-]+$")
SKIP_PREFIXES = (
    "Found ",
    "Success",
    "Testing ",
    "Gnuplot",
    "Warming ",
    "Collecting ",
    "Analyzing",
    "change:",
    "No change",
    "Performance has",
)


def to_microseconds(value: float, unit: str) -> float:
    unit = unit.replace("µ", "u")
    if unit == "ns":
        return value / 1000.0
    if unit == "ms":
        return value * 1000.0
    return value


def parse_criterion_log(text: str) -> dict[str, float]:
    results: dict[str, float] = {}
    pending_name: str | None = None
    group: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("Benchmarking "):
            group = stripped.removeprefix("Benchmarking ").split(":")[0].strip()
            continue

        if any(stripped.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue

        if "time:" in stripped:
            time_match = TIME_RE.search(stripped)
            if time_match:
                mid, unit = time_match.groups()
                name_part = stripped.split("time:", 1)[0].strip()
                key = name_part or pending_name or group
                if key:
                    if key in {"unshared_rhs", "shared_rhs"}:
                        key = f"scatter_array_f64/{key}"
                    results[key] = to_microseconds(float(mid), unit)
                pending_name = None
                group = None
            continue

        if NAME_RE.match(stripped):
            pending_name = stripped

    return results


def run_rust_bench() -> str:
    proc = subprocess.run(
        ["cargo", "bench", "--bench", "paths"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout + proc.stderr


def run_numpy_bench() -> dict:
    python = ROOT / ".venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    proc = subprocess.run(
        [str(python), "benches/numpy_paths.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def fmt_us(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def render_canvas(
    sdnp: dict[str, float],
    numpy: dict[str, float],
    numpy_meta: dict,
) -> str:
    rows: list[tuple[str, str, str, float, float]] = []
    for key, (group, name) in METADATA.items():
        if key not in sdnp or key not in numpy:
            continue
        rows.append((group, name, key, sdnp[key], numpy[key] * 1_000_000))

    def ratio(sdnp_us: float, numpy_us: float) -> float:
        return sdnp_us / numpy_us

    sdnp_wins = sum(1 for _, _, _, s, n in rows if ratio(s, n) <= 1)
    gaps = sorted(
        (
            {
                "group": g,
                "name": n,
                "key": k,
                "sdnpUs": s,
                "numpyUs": nump,
                "r": ratio(s, nump),
            }
            for g, n, k, s, nump in rows
        ),
        key=lambda item: item["r"],
        reverse=True,
    )
    compute_gaps = [g for g in gaps if "view" not in g["name"].lower() or "transposed" in g["name"]]
    worst = gaps[0] if gaps else {"name": "n/a", "r": 1.0, "sdnpUs": 0, "numpyUs": 0}
    top10 = compute_gaps[:10]

    results_lines = [
        f'  {{ group: "{g}", name: "{n}", sdnpUs: {fmt_us(s)}, numpyUs: {fmt_us(nump)} }},'
        for g, n, _, s, nump in rows
    ]

    chart_categories = json.dumps([r["name"] for r in top10], ensure_ascii=False)
    chart_data = json.dumps([round(r["r"], 2) for r in top10])

    priority_lines = []
    for i, item in enumerate(top10[:5], start=1):
        priority_lines.append(
            f'          <Text>\n'
            f'            <Text weight="semibold">{i}. {item["name"]}</Text>\n'
            f'            {{" — "}}\n'
            f'            {item["r"]:.1f}×\n'
            f'          </Text>'
        )

    fast_items = [
        item
        for item in rows
        if ratio(item[3], item[4]) <= 1 and "view" not in item[1].lower()
    ][:8]

    fast_lines = []
    for group, name, _, s, n in fast_items:
        r = ratio(s, n)
        if r <= 1:
            label = f"{1/r:.2f}× 빠름" if r < 1 else "동률"
            fast_lines.append(
                f'            <Text><Text weight="semibold">{name}</Text> — {label}</Text>'
            )

    numpy_version = numpy_meta.get("numpy_version", "?")
    python_version = numpy_meta.get("python_version", "?")
    machine = numpy_meta.get("machine", "?")

    phase6_rows = []
    for key, prior_us in PHASE6_PRIOR_US.items():
        if key not in sdnp:
            continue
        _, name = METADATA[key]
        after_us = sdnp[key]
        phase6_rows.append(
            {
                "name": name,
                "beforeUs": prior_us,
                "afterUs": after_us,
                "speedup": prior_us / after_us if after_us else 1.0,
            }
        )
    phase6_rows.sort(key=lambda item: item["speedup"], reverse=True)
    phase6_categories = json.dumps(
        [r["name"] for r in phase6_rows], ensure_ascii=False
    )
    phase6_series = ",\n    ".join(
        [
            (
                f'{{ name: {json.dumps("최적화 전 SDNP (µs)", ensure_ascii=False)}, '
                f'data: {json.dumps([round(r["beforeUs"], 1) for r in phase6_rows])}, '
                f'tone: "neutral" as const }}'
            ),
            (
                f'{{ name: {json.dumps("최적화 후 SDNP (µs)", ensure_ascii=False)}, '
                f'data: {json.dumps([round(r["afterUs"], 1) for r in phase6_rows])}, '
                f'tone: "success" as const }}'
            ),
        ]
    )
    phase6_lines = []
    for item in phase6_rows:
        phase6_lines.append(
            f'            <Text><Text weight="semibold">{item["name"]}</Text>'
            f' — {item["speedup"]:.1f}× 빠름'
            f' ({fmt_us(item["beforeUs"])} → {fmt_us(item["afterUs"])} µs)</Text>'
        )

    iteration_rows = [
        {
            "name": name,
            "sdnpUs": sdnp_us,
            "numpyUs": numpy_us,
            "ratio": ratio(sdnp_us, numpy_us),
        }
        for group, name, _, sdnp_us, numpy_us in rows
        if group == "Iteration"
    ]
    iteration_rows.sort(key=lambda item: item["ratio"])
    iteration_categories = json.dumps(
        [item["name"] for item in iteration_rows], ensure_ascii=False
    )
    iteration_data = json.dumps(
        [round(item["ratio"], 3) for item in iteration_rows]
    )
    iteration_lines = [
        (
            f'            <Text><Text weight="semibold">{item["name"]}</Text>'
            f' — SDNP {item["ratio"]:.3f}×'
            f' ({fmt_us(item["sdnpUs"])} µs vs {fmt_us(item["numpyUs"])} µs)</Text>'
        )
        for item in iteration_rows
    ]
    iteration_section = ""
    if iteration_rows:
        iteration_section = f'''
      <Stack gap={{12}}>
        <H2>Phase 7 iteration · SDNP / NumPy</H2>
        <Text tone="tertiary" size="small">
          Y축: iterator 경로 · X축: SDNP 시간 ÷ NumPy 시간 (배, 낮을수록 빠름)
        </Text>
        <BarChart
          categories={{{iteration_categories}}}
          series={{[{{ name: "SDNP / NumPy 배율", data: {iteration_data}, tone: "info" as const }}]}}
          horizontal
          height={{280}}
          beginAtZero
          referenceLines={{[{{ value: 1, label: "동률", tone: "neutral" }}]}}
          showValues
        />
        <Text tone="tertiary" size="small">
          Source: Criterion median 및 NumPy median · 전체 logical 원소 checksum 소비 ·
          {machine} · NumPy {numpy_version} / Python {python_version} · 2026-07-26
        </Text>
        <Stack gap={{6}}>
{chr(10).join(iteration_lines)}
        </Stack>
      </Stack>
'''

    focused_sections = []
    for group, title, caption in (
        (
            "신규 기능",
            "신규 기능 · SDNP / NumPy",
            "squeeze metadata view 및 astype C-order materialization",
        ),
        (
            "NaN policy",
            "NanPolicy::Ignore · SDNP / NumPy",
            "NaN skip 전용 suffix/prefix/general/cumulative kernels",
        ),
    ):
        selected = [
            {
                "name": name,
                "sdnpUs": sdnp_us,
                "numpyUs": numpy_us,
                "ratio": ratio(sdnp_us, numpy_us),
            }
            for row_group, name, _, sdnp_us, numpy_us in rows
            if row_group == group
        ]
        if not selected:
            continue
        selected.sort(key=lambda item: item["ratio"])
        categories = json.dumps(
            [item["name"] for item in selected], ensure_ascii=False
        )
        data = json.dumps([round(item["ratio"], 3) for item in selected])
        focused_sections.append(
            f'''
      <Stack gap={{12}}>
        <H2>{title}</H2>
        <Text tone="tertiary" size="small">
          Y축: 연산 경로 · X축: SDNP 시간 ÷ NumPy 시간 (배, 낮을수록 빠름)
        </Text>
        <BarChart
          categories={{{categories}}}
          series={{[{{ name: "SDNP / NumPy 배율", data: {data}, tone: "info" as const }}]}}
          horizontal
          height={{{max(240, len(selected) * 34)}}}
          beginAtZero
          referenceLines={{[{{ value: 1, label: "동률", tone: "neutral" }}]}}
          showValues
        />
        <Text tone="tertiary" size="small">
          Source: Criterion median 및 NumPy 7회 median · {caption} ·
          {machine} · NumPy {numpy_version} / Python {python_version} · 2026-07-26
        </Text>
      </Stack>
'''
        )
    focused_sections_text = "\n".join(focused_sections)

    propagate_rows = []
    for key, before_us in PROPAGATE_PRIOR_US.items():
        if key not in sdnp:
            continue
        _, name = METADATA[key]
        after_us = sdnp[key]
        propagate_rows.append(
            {
                "name": name,
                "beforeUs": before_us,
                "afterUs": after_us,
                "ratio": after_us / before_us,
            }
        )
    propagate_categories = json.dumps(
        [item["name"] for item in propagate_rows], ensure_ascii=False
    )
    propagate_data = json.dumps(
        [round(item["ratio"], 3) for item in propagate_rows]
    )
    propagate_lines = [
        (
            f'            <Text><Text weight="semibold">{item["name"]}</Text>'
            f' — {(item["ratio"] - 1) * 100:+.1f}%'
            f' ({fmt_us(item["beforeUs"])} → {fmt_us(item["afterUs"])} µs)</Text>'
        )
        for item in propagate_rows
    ]
    propagate_section = ""
    if propagate_rows:
        propagate_section = f'''
      <Stack gap={{12}}>
        <H2>NanPolicy::Propagate · 정책 추가 전 대비</H2>
        <Text tone="tertiary" size="small">
          Y축: 동일 reduction 경로 · X축: 현재 시간 ÷ 정책 추가 전 시간 (배)
        </Text>
        <BarChart
          categories={{{propagate_categories}}}
          series={{[{{ name: "현재 / 이전", data: {propagate_data}, tone: "success" as const }}]}}
          horizontal
          height={{{max(300, len(propagate_rows) * 34)}}}
          beginAtZero
          referenceLines={{[{{ value: 1, label: "회귀 없음 기준", tone: "neutral" }}]}}
          showValues
        />
        <Text tone="tertiary" size="small">
          Source: 동일 Criterion 경로의 정책 추가 전 마지막 median과 현재 선택 재측정 ·
          1.0 미만은 개선 · 2026-07-26
        </Text>
        <Stack gap={{6}}>
{chr(10).join(propagate_lines)}
        </Stack>
      </Stack>
'''

    return f'''import {{
  BarChart,
  Callout,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
}} from "cursor/canvas";

type Result = {{
  group: string;
  name: string;
  sdnpUs: number;
  numpyUs: number;
}};

const results: Result[] = [
{chr(10).join(results_lines)}
];

const groups = ["전체", ...Array.from(new Set(results.map((r) => r.group)))];

function ratio(result: Result) {{
  return result.sdnpUs / result.numpyUs;
}}

function formatTime(us: number) {{
  if (us < 1) return `${{(us * 1000).toFixed(1)}} ns`;
  if (us >= 1000) return `${{(us / 1000).toFixed(2)}} ms`;
  return `${{us.toFixed(us < 100 ? 2 : 1)}} µs`;
}}

function comparison(result: Result) {{
  const r = ratio(result);
  if (r <= 1) {{
    return {{
      text: `SDNP ${{(1 / r).toFixed(2)}}× 빠름`,
      tone: "success" as const,
    }};
  }}
  return {{
    text: `NumPy ${{r.toFixed(2)}}× 빠름`,
    tone: r >= 5 ? ("danger" as const) : r >= 2 ? ("warning" as const) : ("neutral" as const),
  }};
}}

function PathsBenchmarkResults() {{
  const theme = useHostTheme();
  const [selectedGroup, setSelectedGroup] = useCanvasState("paths-group", "전체");
  const visible =
    selectedGroup === "전체"
      ? results
      : results.filter((r) => r.group === selectedGroup);

  const sdnpWins = results.filter((r) => ratio(r) <= 1).length;
  const gaps = results
    .map((r) => ({{ ...r, r: ratio(r) }}))
    .filter((r) => r.r > 1)
    .sort((a, b) => b.r - a.r);
  const worst = gaps[0] ?? {{ name: "n/a", r: 1 }};
  const computeGaps = gaps.filter((r) => !r.name.includes("view"));

  const chartRows = computeGaps.slice(0, 10);
  const chartCategories = {chart_categories};
  const chartSeries = [
    {{
      name: "SDNP / NumPy 배율",
      data: {chart_data},
      tone: "danger" as const,
    }},
  ];
  const phase6Categories = {phase6_categories};
  const phase6Series = [
    {phase6_series}
  ];

  return (
    <Stack gap={{24}} style={{{{ padding: 24 }}}}>
      <Stack gap={{8}}>
        <Text size="small" weight="semibold" style={{{{ color: theme.accent.primary }}}}>
          SDNP · benches/paths
        </Text>
        <H1>통합 경로 벤치 · SDNP vs NumPy</H1>
        <Text tone="secondary">
          단일 스레드 실행 시간(median). 낮을수록 좋습니다.
          SDNP는 Criterion median, NumPy는 7회 측정 median입니다.
        </Text>
      </Stack>

      <Grid columns={{4}} gap={{16}}>
        <Stat value={{String(results.length)}} label="측정 경로" />
        <Stat
          value={{`${{sdnpWins}} / ${{results.length}}`}}
          label="SDNP 우세·동률"
          tone="success"
        />
        <Stat
          value={{`${{worst.r.toFixed(1)}}×`}}
          label={{`최대 격차 · ${{worst.name}}`}}
          tone="danger"
        />
        <Stat value="≈1.0×" label="생성·contiguous copy" tone="success" />
      </Grid>

      <Callout tone="info" title="요약">
        기존 전체 결과를 보존하고 squeeze·astype·NanPolicy 및 이번에 수정한
        커널 경로만 선택 재측정해 병합했습니다. SDNP 우세·동률 경로는
        {sdnp_wins}/{len(rows)}개입니다.
      </Callout>

{focused_sections_text}
{propagate_section}
      <Stack gap={{12}}>
        <H2>Phase 6 linalg · 최적화 전후 (SDNP 자체 시간)</H2>
        <Text tone="tertiary" size="small">
          Y축: 연산 경로 · X축: SDNP 실행 시간 (µs, median) · 2026-07-25 대비
          2026-07-26 영향 경로만 재측정
        </Text>
        <BarChart
          categories={{phase6Categories}}
          series={{phase6Series}}
          horizontal
          height={{360}}
          beginAtZero
        />
        <Stack gap={{6}}>
{chr(10).join(phase6_lines)}
        </Stack>
      </Stack>

{iteration_section}
      <Stack gap={{12}}>
        <H2>NumPy 대비 느린 compute 경로 (상위 10)</H2>
        <Text tone="tertiary" size="small">
          Y축: 연산 경로 · X축: SDNP 시간 ÷ NumPy 시간 (배) · view 메타데이터 경로 제외
        </Text>
        <BarChart
          categories={{chartCategories}}
          series={{chartSeries}}
          horizontal
          height={{320}}
          beginAtZero
          referenceLines={{[{{ value: 1, label: "동률", tone: "neutral" }}]}}
        />
        <Text tone="tertiary" size="small">
          Source: Criterion median 및 NumPy median · {machine} · NumPy {numpy_version}
          / Python {python_version} · Phase 6 영향 경로 재측정 2026-07-26
        </Text>
      </Stack>

      <Stack gap={{12}}>
        <H2>전체 측정 결과</H2>
        <Row gap={{8}} wrap>
          {{groups.map((group) => (
            <span key={{group}}>
              <Pill
                active={{selectedGroup === group}}
                onClick={{() => setSelectedGroup(group)}}
              >
                {{group}}
              </Pill>
            </span>
          ))}}
        </Row>
        <Table
          headers={{["분류", "연산", "SDNP", "NumPy", "상대 성능"]}}
          rows={{visible.map((r) => [
            r.group,
            r.name,
            formatTime(r.sdnpUs),
            formatTime(r.numpyUs),
            comparison(r).text,
          ])}}
          columnAlign={{["left", "left", "right", "right", "right"]}}
          rowTone={{visible.map((r) => comparison(r).tone)}}
          striped
          stickyHeader
        />
        <Text tone="tertiary" size="small">
          Source: cargo bench --bench paths (Criterion 0.5.1) · NumPy {numpy_version}
          / Python {python_version} · {machine} · 2026-07-26 · Rust: opt-level 3,
          fat LTO, codegen-units 1
        </Text>
      </Stack>

      <Grid columns={{2}} gap={{20}}>
        <Stack gap={{10}}>
          <H2>우선 최적화 후보</H2>
{chr(10).join(priority_lines)}
          <Text tone="tertiary" size="small">
            matmul은 NumPy(Accelerate GEMM) 대비 여전히 한 자릿수~두 자릿수 배
            느립니다. 남은 후보는 BLAS급 GEMM, general multi-axis sum, sort/arg
            계열입니다.
          </Text>
        </Stack>
        <Stack gap={{10}}>
          <H2>SDNP 우세 경로 (일부)</H2>
{chr(10).join(fast_lines)}
        </Stack>
      </Grid>

      <Callout tone="warning" title="해석 시 주의">
        transpose·reshape·broadcast 같은 view 연산은 수십 ns라 NumPy의 Python
        호출 비용이 대부분입니다. zeros/eye는 allocator zero-page 재사용 영향이
        커서 메모리 쓰기 처리량으로 읽으면 안 됩니다. strided materialization은
        두 구현 모두 C-order 결과를 생성하도록 맞췄습니다.
      </Callout>
    </Stack>
  );
}}

export default PathsBenchmarkResults;
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-log", type=Path)
    parser.add_argument("--numpy-json", type=Path)
    parser.add_argument("--output", type=Path, default=CANVAS)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if args.skip_run:
        if not args.rust_log or not args.numpy_json:
            raise SystemExit("--skip-run requires --rust-log and --numpy-json")
        rust_text = args.rust_log.read_text()
        numpy_payload = json.loads(args.numpy_json.read_text())
    else:
        print("Running cargo bench --bench paths …", file=sys.stderr)
        rust_text = run_rust_bench()
        print("Running NumPy benchmarks …", file=sys.stderr)
        numpy_payload = run_numpy_bench()

    sdnp = parse_criterion_log(rust_text)
    numpy_seconds = numpy_payload["timings_seconds"]
    canvas = render_canvas(sdnp, numpy_seconds, numpy_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canvas)
    matched = sum(
        1 for key in METADATA if key in sdnp and key in numpy_seconds
    )
    diagnostics = sum(1 for key in DIAGNOSTIC_KEYS if key in sdnp)
    print(
        f"Wrote {args.output} ({len(METADATA)} comparable keys, "
        f"matched {matched} paths, {diagnostics} Rust-only diagnostics)"
    )


if __name__ == "__main__":
    main()
