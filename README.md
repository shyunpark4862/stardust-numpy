# stardust-numpy (`sdnp`)

Educational NumPy-style array library in Rust, designed so a **PyO3 Python
binding** can be added later (see `plan.md` Phase 8).

- `Array<T>` with compile-time generics
- Auto-promotion: `bool < i64 < f64 < Complex<f64>`
- 0-D arrays allowed (`shape == []`, size 1); unlike the Python reference
- No `str` / object / mixed-dtype arrays / runtime dtype API in the Rust core
- Views share via `Arc`; writes use copy-on-write (not NumPy write-through)

Spec / behavioral reference: sibling Python project `../numpy` (with intentional differences).

## Setup

```bash
# Install Rust if needed: https://rustup.rs
cargo test
```

## Benchmarks

```bash
# Rust: all paths or one Criterion substring
cargo bench --bench paths
cargo bench --bench paths -- concatenate_axis0_strided

# NumPy: project-local Python 3 environment
uv venv --python python3 .venv
uv pip install --python .venv/bin/python numpy
.venv/bin/python benches/numpy_paths.py

# NumPy: list, exact path, or substring selection
.venv/bin/python benches/numpy_paths.py --list
.venv/bin/python benches/numpy_paths.py concatenate_axis0_strided_f64
.venv/bin/python benches/numpy_paths.py --match concatenate --match where

# Rebuild the Markdown report and Cursor canvas from existing measurements
.venv/bin/python benches/generate_benchmark_report.py --skip-run
.venv/bin/python benches/render_benchmark_canvas.py --skip-run \
  --rust-log benches/.bench-rust.log \
  --numpy-json benches/.bench-numpy.json
```

Materializing comparison paths produce C-contiguous outputs on both sides.
With no Python path filter, the script keeps its JSON output format and runs
the full suite.

Rust and NumPy measurements must run sequentially to avoid CPU interference.
The generated report is `benches/BENCHMARK_REPORT.md`; its optimization
reference is maintained in `benches/optimization_techniques.md`. Raw
`.bench-rust.log` and `.bench-numpy.json` files are local generated artifacts
and are ignored; the Markdown report is the reviewable checked-in result.

## Layout

```
src/
  dtype.rs       Scalar + Promote + CastTo + AsBool
  axis.rs        shared negative-axis normalization
  shape.rs       size / strides / contiguity / offset_at
  error.rs
  array/         Array<T>, element get/set (CoW), views (transpose/reshape)
  index/         IndexSpec, bounds helpers, gather / scatter
  create.rs      zeros/ones/full/arange/eye
  broadcast.rs   broadcast_shape / broadcast_to / …
  ufunc/         kernels + ops + traits (internals pub(crate))
  layout.rs      joint operand layout coalescing (pub(crate))
  run.rs         prepared fixed-stride runs (pub(crate))
  stride_iter.rs StrideIter (pub(crate); public API in Phase 7)
  reduce/        reduction geometry, traits, ops, split kernel families
  join.rs        concatenate / stack / vstack / hstack
  select.rs      where_ / nonzero / clip
  sort.rs        sort / argsort / unique
  diagonal.rs    shared diagonal geometry (pub(crate))
  linalg/        Phase 6 geometry / kernels / ops (pub(crate) skeleton)
  format.rs      Phase 7 (pub(crate) stub)
```

## What already works (Phase 0–5)

- `Array::from_vec` / `from_slice` / `get` / `set` (CoW) / `item` / `to_vec` / `as_c_contiguous_slice`
- `transpose` / `t` / `permute_axes` / `reshape` / `copy` / `broadcast_to` (broadcast views read-only)
- indexing: `IndexSpec` + `gather` / `scatter` / `scatter_array` (basic → view; fancy/bool → copy; 음수·step·newaxis·ellipsis)
- `zeros` / `ones` / `full` / `arange` (`i64`) / `eye` (+ broadcasting)
- ufuncs: `add`/`subtract`/`multiply`/`divide`/`trunc_divide`/`remainder`/`power`/`negative`/`absolute`
- comparisons + `logical_and`/`or`/`not` + `isnan`/`isinf`/`isfinite`
- complex: `conj` / `real` / `imag`
- reductions: `sum`/`prod`/`min`/`max`/`mean`/`var`/`std`/`any`/`all`
  (`axes`, `keepdims`); `argmin`/`argmax`/`cumsum`/`cumprod` (`axis`)
- joining: `concatenate` / `stack` / `vstack` / `hstack`
- selection: `where_` / `nonzero` / `clip`
- sorting: `sort` / `sort_in_place` / `argsort` / `unique`
- spaces: `linspace` / `logspace` / `geomspace` / `meshgrid`
- `dtype::{Scalar, Promote, CastTo, AsBool}`

Intentional differences vs NumPy: no operator overloading; `divide` follows Rust `/`; CoW on write; no `out`/`where`.

Fill in modules phase-by-phase; Python bindings are Phase 8.

## Internal terminology

- **C-contiguous**: an entire logical array has C-order strides; a non-zero
  backing-buffer offset is allowed.
- **Unit-stride run**: one coalesced run advances an operand by one element.
  The whole array need not be C-contiguous.
- **Run grid / run**: `RunPlan` traverses a grid of fixed-stride linear runs.
- **Kept / reduced axes**: reduction output axes versus axes folded into each
  output. `output_len` and `reduction_len` are their element counts.
- **Prefix / suffix reduction**: reduced axes form the leading or trailing
  logical axis block; this is unrelated to join's leading dimensions.
