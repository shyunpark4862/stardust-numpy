# sdnp-py vs NumPy Benchmark Results

- Run: `20260725T225517.805754Z`
- Timestamp: `2026-07-25T22:55:17.805693+00:00`
- Platform: `macOS-26.5.2-arm64-arm-64bit`
- Python: `3.12.11`
- sdnp: `0.1.0` (`release`)
- NumPy: `2.5.1`
- Git commit: `8bd4131c6188acafb8bf08e08674f2a6e5003f74`
- Profile: `smoke`
- Warmups / samples: `1` / `3`
- Iterations: `2`
- Target sample: `50.0 ms`
- Filters: `{"case": [], "category": [], "dtype": [], "function": ["add"], "match": [], "ndim": [], "size": []}`
- Raw CSV: `benches/results/benchmark.csv` (6 rows, sha256 `35853ae8057ba27b0a75db54b44b238b934ec50e58295dea0ad81c9121b2dc26`)

## Summary

- Cases: **1**
- sdnp median ≤ NumPy median: **0 / 1**

| Category | Cases | Median ratio | Mean ratio |
| --- | ---: | ---: | ---: |
| ufunc | 1 | 28.253× | 28.253× |

## Complete Results

| Function | dtype | size | ndim | shape | sdnp p25 | sdnp p50 | sdnp p75 | sdnp mean | NumPy p25 | NumPy p50 | NumPy p75 | NumPy mean | ratio |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add | float64 | small | 2 | 8×8 | 8.12 µs | 8.25 µs | 8.58 µs | 8.39 µs | 281.5 ns | 292.0 ns | 323.0 ns | 305.7 ns | 28.253× |

_Generated directly from benchmark summary JSON._
