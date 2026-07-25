"""Deterministic Markdown and Cursor Canvas rendering from summary JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benches" / "results"
MARKDOWN_PATH = ROOT / "BENCHMARK.md"


def default_canvas_path() -> Path:
    workspace_slug = str(ROOT).strip("/").replace("/", "-")
    return (
        Path.home()
        / ".cursor"
        / "projects"
        / workspace_slug
        / "canvases"
        / "sdnp-numpy-benchmark.canvas.tsx"
    )


def _time(value_ns: float) -> str:
    if value_ns < 1_000:
        return f"{value_ns:.1f} ns"
    if value_ns < 1_000_000:
        return f"{value_ns / 1_000:.2f} µs"
    if value_ns < 1_000_000_000:
        return f"{value_ns / 1_000_000:.2f} ms"
    return f"{value_ns / 1_000_000_000:.2f} s"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    metadata = payload["metadata"]
    config = payload["config"]
    faster = sum(result["ratio_median"] <= 1.0 for result in results)

    by_category: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_category[result["case"]["category"]].append(result["ratio_median"])

    category_lines = [
        "| Category | Cases | Median ratio | Mean ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, ratios in sorted(by_category.items()):
        ordered = sorted(ratios)
        median = ordered[len(ordered) // 2]
        mean = sum(ratios) / len(ratios)
        category_lines.append(
            f"| {_escape(category)} | {len(ratios)} | {median:.3f}× | {mean:.3f}× |"
        )

    result_lines = [
        "| Function | dtype | size | ndim | shape | "
        "sdnp p25 | sdnp p50 | sdnp p75 | sdnp mean | "
        "NumPy p25 | NumPy p50 | NumPy p75 | NumPy mean | ratio |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        case = result["case"]
        sd = result["sdnp"]
        np_result = result["numpy"]
        result_lines.append(
            "| "
            + " | ".join(
                [
                    _escape(case["function"]),
                    case["dtype"],
                    case["size"],
                    str(case["ndim"]),
                    "×".join(map(str, case["shape"])),
                    _time(sd["p25_ns"]),
                    _time(sd["median_ns"]),
                    _time(sd["p75_ns"]),
                    _time(sd["mean_ns"]),
                    _time(np_result["p25_ns"]),
                    _time(np_result["median_ns"]),
                    _time(np_result["p75_ns"]),
                    _time(np_result["mean_ns"]),
                    f"{result['ratio_median']:.3f}×",
                ]
            )
            + " |"
        )

    filters = payload.get("filters", {})
    return "\n".join(
        [
            "# sdnp-py vs NumPy Benchmark Results",
            "",
            f"- Run: `{metadata['run_id']}`",
            f"- Timestamp: `{metadata['started_at']}`",
            f"- Platform: `{metadata['platform']}`",
            f"- Python: `{metadata['python_version']}`",
            f"- sdnp: `{metadata['sdnp_version']}` (`{metadata['build_profile']}`)",
            f"- NumPy: `{metadata['numpy_version']}`",
            f"- Git commit: `{metadata['git_commit']}`",
            f"- Profile: `{config['profile']}`",
            f"- Warmups / samples: `{config['warmups']}` / `{config['samples']}`",
            f"- Iterations: `{config['iterations'] or 'auto'}`",
            f"- Target sample: `{config['target_sample_ms']} ms`",
            f"- Filters: `{json.dumps(filters, sort_keys=True)}`",
            f"- Raw CSV: `{payload['raw_csv']['path']}` "
            f"({payload['raw_csv']['rows']} rows, sha256 `{payload['raw_csv']['sha256']}`)",
            "",
            "## Summary",
            "",
            f"- Cases: **{len(results)}**",
            f"- sdnp median ≤ NumPy median: **{faster} / {len(results)}**",
            "",
            *category_lines,
            "",
            "## Complete Results",
            "",
            *result_lines,
            "",
            "_Generated directly from benchmark summary JSON._",
            "",
        ]
    )


def render_canvas(payload: dict[str, Any]) -> str:
    compact_results = [
        {
            **result["case"],
            "sdnp": result["sdnp"],
            "numpy": result["numpy"],
            "ratio": result["ratio_median"],
        }
        for result in payload["results"]
    ]
    data = json.dumps(
        compact_results, ensure_ascii=False, separators=(",", ":")
    )
    source = json.dumps(
        {
            "run": payload["metadata"]["run_id"],
            "timestamp": payload["metadata"]["started_at"],
            "python": payload["metadata"]["python_version"],
            "numpy": payload["metadata"]["numpy_version"],
            "sdnp": payload["metadata"]["sdnp_version"],
            "platform": payload["metadata"]["platform"],
        },
        ensure_ascii=False,
    )
    return f"""// Generated from benches/results/benchmark.json.
import {{
  BarChart,
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
}} from "cursor/canvas";

type Distribution = {{
  p25_ns: number;
  median_ns: number;
  p75_ns: number;
  mean_ns: number;
  minimum_ns: number;
  maximum_ns: number;
  samples: number;
  iterations: number;
}};

type Result = {{
  id: string;
  function: string;
  category: string;
  dtype: string;
  size: string;
  ndim: number;
  shape: number[];
  variant: string;
  sdnp: Distribution;
  numpy: Distribution;
  ratio: number;
}};

const results: Result[] = {data};
const source = {source};

function formatTime(ns: number) {{
  if (ns < 1_000) return `${{ns.toFixed(1)}} ns`;
  if (ns < 1_000_000) return `${{(ns / 1_000).toFixed(2)}} µs`;
  if (ns < 1_000_000_000) return `${{(ns / 1_000_000).toFixed(2)}} ms`;
  return `${{(ns / 1_000_000_000).toFixed(2)}} s`;
}}

function choices(key: "category" | "dtype" | "size" | "ndim") {{
  return ["all", ...Array.from(new Set(results.map((result) => String(result[key]))))];
}}

export default function SdnpNumpyBenchmark() {{
  const [category, setCategory] = useCanvasState("benchmark-category", "all");
  const [dtype, setDtype] = useCanvasState("benchmark-dtype", "all");
  const [size, setSize] = useCanvasState("benchmark-size", "all");
  const [ndim, setNdim] = useCanvasState("benchmark-ndim", "all");
  const visible = results.filter((result) =>
    (category === "all" || result.category === category) &&
    (dtype === "all" || result.dtype === dtype) &&
    (size === "all" || result.size === size) &&
    (ndim === "all" || String(result.ndim) === ndim)
  );
  const chartRows = [...visible].sort((a, b) => b.ratio - a.ratio).slice(0, 30);
  const wins = visible.filter((result) => result.ratio <= 1).length;
  const medianRatio = visible.length
    ? [...visible].map((result) => result.ratio).sort((a, b) => a - b)[Math.floor(visible.length / 2)]
    : 0;

  return (
    <Stack gap={{24}} style={{{{ padding: 24 }}}}>
      <Stack gap={{6}}>
        <H1>sdnp-py vs NumPy Benchmark Results</H1>
        <Text tone="secondary">
          Median execution time comparison. Lower values are faster.
        </Text>
      </Stack>

      <Grid columns={{3}} gap={{16}}>
        <Stat value={{String(visible.length)}} label="Visible cases" />
        <Stat value={{`${{wins}} / ${{visible.length}}`}} label="sdnp ≤ NumPy" />
        <Stat value={{`${{medianRatio.toFixed(3)}}×`}} label="Median sdnp / NumPy" />
      </Grid>

      <Stack gap={{10}}>
        <H2>Filters</H2>
        {{([
          ["Category", choices("category"), category, setCategory],
          ["dtype", choices("dtype"), dtype, setDtype],
          ["Size", choices("size"), size, setSize],
          ["Dimensions", choices("ndim"), ndim, setNdim],
        ] as const).map(([label, values, selected, setter]) => (
          <Row gap={{8}} wrap>
            <Text weight="semibold">{{label}}</Text>
            {{values.map((value) => (
              <Pill active={{selected === value}} onClick={{() => setter(value)}}>
                {{value}}
              </Pill>
            ))}}
          </Row>
        ))}}
      </Stack>

      {{chartRows.length > 0 && (
        <Stack gap={{10}}>
          <H2>Median execution-time ratio by case</H2>
          <Text tone="tertiary" size="small">
            X-axis: sdnp median / NumPy median (ratio) · Y-axis: benchmark case ·
            1.0 is equal performance
          </Text>
          <BarChart
            categories={{chartRows.map((result) => result.id)}}
            series={{[{{ name: "sdnp / NumPy median", data: chartRows.map((result) => result.ratio), tone: "info" }}]}}
            horizontal
            height={{Math.max(280, chartRows.length * 28)}}
            beginAtZero
            referenceLines={{[{{ value: 1, label: "equal", tone: "neutral" }}]}}
            showValues
          />
          <Text tone="tertiary" size="small">
            Source: raw benchmark samples summarized as medians · Run {{source.run}} ·
            {{source.timestamp}} · Python {{source.python}} · NumPy {{source.numpy}} ·
            sdnp {{source.sdnp}} · {{source.platform}}
          </Text>
        </Stack>
      )}}

      <Stack gap={{10}}>
        <H2>Complete benchmark statistics</H2>
        <Table
          headers={{[
            "Function", "dtype", "Size", "Shape",
            "sdnp p25", "sdnp p50", "sdnp p75", "sdnp mean",
            "NumPy p25", "NumPy p50", "NumPy p75", "NumPy mean", "Ratio"
          ]}}
          rows={{visible.map((result) => [
            result.function,
            result.dtype,
            result.size,
            result.shape.join("×"),
            formatTime(result.sdnp.p25_ns),
            formatTime(result.sdnp.median_ns),
            formatTime(result.sdnp.p75_ns),
            formatTime(result.sdnp.mean_ns),
            formatTime(result.numpy.p25_ns),
            formatTime(result.numpy.median_ns),
            formatTime(result.numpy.p75_ns),
            formatTime(result.numpy.mean_ns),
            `${{result.ratio.toFixed(3)}}×`,
          ])}}
          columnAlign={{["left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}}
          striped
          stickyHeader
        />
        <Text tone="tertiary" size="small">
          Source: summary JSON and linked raw CSV · Run {{source.run}} · {{source.timestamp}}
        </Text>
      </Stack>
    </Stack>
  );
}}
"""


def render_outputs(summary_path: Path) -> tuple[Path, Path]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = render_markdown(payload)
    canvas = render_canvas(payload)

    MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    canvas_path = default_canvas_path()
    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_path.write_text(canvas, encoding="utf-8")
    return MARKDOWN_PATH, canvas_path
