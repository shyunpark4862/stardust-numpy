# sdnp-py vs NumPy Benchmark Results

- Run: `20260726T165718.346038Z`
- Timestamp: `2026-07-26T16:57:18.345911+00:00`
- Platform: `macOS-26.5.2-arm64-arm-64bit`
- Python: `3.12.11`
- sdnp: `0.1.0` (`release`)
- NumPy: `2.5.1`
- Git commit: `e19d165351a61c4bc73d30ffb3db18cfe8c810c7`
- Profile: `full`
- Warmups / samples: `1` / `11`
- Iterations: `auto`
- Target sample: `25.0 ms`
- Filters: `{"case": [], "category": [], "dtype": [], "function": ["Array.permute_axes"], "match": [], "merge": true, "ndim": [], "size": []}`
- Raw CSV: `benches/results/benchmark.csv` (83952 rows, sha256 `4473f4da2bb0f8851fdba5e19e6c67dd17ce126e065bbe35e642b85fa458457c`)

## Summary

- Cases: **3816**
- sdnp median ≤ NumPy median: **1934 / 3816**

| Category | Cases | Median ratio | Mean ratio |
| --- | ---: | ---: | ---: |
| array | 912 | 1.315× | 1.452× |
| creation | 339 | 0.706× | 0.874× |
| dunder | 528 | 1.110× | 1.542× |
| iteration | 108 | 1.220× | 1.534× |
| linalg | 189 | 0.760× | 1.568× |
| manipulation | 192 | 0.869× | 0.876× |
| reduction | 552 | 0.315× | 0.493× |
| selection | 132 | 0.951× | 0.964× |
| sorting | 120 | 0.622× | 0.784× |
| ufunc | 744 | 1.060× | 1.133× |

_Generated directly from benchmark summary JSON._
