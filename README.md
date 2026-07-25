# stardust-numpy (`sdnp`)

Educational NumPy-style array library in Rust with an optional **PyO3 Python
binding** (`sdnp` package in `sdnp-py/`).

- `Array<T>` with compile-time generics
- Auto-promotion: `bool < i64 < f64 < Complex<f64>`
- 0-D arrays allowed (`shape == []`, size 1); unlike the Python reference
- No `str` / object / mixed-dtype arrays / runtime dtype API in the Rust core
- Views share via `Arc`; writes use copy-on-write (not NumPy write-through)

Spec / behavioral reference: sibling Python project `../numpy` (with intentional differences).

## Setup

```bash
# Rust core
cargo test

# Python binding (requires Python >= 3.12, maturin)
pip install maturin pytest
cd sdnp-py
maturin develop          # or: pip install -e .
pytest python-tests
```

On Python 3.14+, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` when building if
PyO3 has not yet published 3.14 wheels for your platform.

Quick smoke check:

```python
import sdnp
a = sdnp.zeros((2, 3))
assert (a + sdnp.ones((2, 3))).shape == [2, 3]
assert not isinstance(sdnp.sum(a), sdnp.Array)  # 0-D unwrap at boundary
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
  dtype/         Scalar + promotion + explicit ArrayCast
  axis.rs        shared negative-axis normalization
  shape.rs       size / strides / contiguity / offset_at
  error.rs
  array/         Array<T>, element get/set (CoW), transpose/reshape/squeeze
  index/         IndexSpec, bounds helpers, gather / scatter
  creation/      factories, ranges/spaces, grids, triangular arrays
  broadcast.rs   broadcast_shape / broadcast_to / …
  ufunc/         kernels + ops + traits (internals pub(crate))
  traversal/     coalesced layouts, RunPlan, stride cursors
  iteration/     ndindex / ndenumerate / nditer / flat / axis-0 iteration
  reduction/     plans, traits, ops, split kernel families
  manipulation/  concatenate / stack / vstack / hstack
  selection/     where_ / nonzero / clip
  sorting/       sort / argsort / unique
  linalg/        contraction/diagonal geometry, traits, kernels, public ops
```

## What already works (Phase 0–7.5)

- `Array::from_vec` / `from_slice` / `get` / `set` (CoW) / `item` / `to_vec` / `as_c_contiguous_slice`
- `transpose` / `t` / `permute_axes` / `reshape` / `squeeze` / `copy` / `astype`
- internal `broadcast_to` support (broadcast views remain read-only)
- indexing: `IndexSpec` + `gather` / `scatter` / `scatter_array` (basic → view; fancy/bool → copy; 음수·step·newaxis·ellipsis)
- `zeros` / `ones` / `full` / `arange` (`i64`) / `eye` (+ broadcasting)
- ufuncs: `add`/`subtract`/`multiply`/`divide`/`trunc_divide`/`remainder`/`power`/`negative`/`absolute`
- comparisons + `logical_and`/`or`/`not` + `isnan`/`isinf`/`isfinite`
- complex: `conj` / `real` / `imag`
- reductions: `sum`/`prod`/`min`/`max`/`mean`/`var`/`std`/`any`/`all`
  (`axes`, `keepdims`); `argmin`/`argmax`/`cumsum`/`cumprod` (`axis`);
  numeric reductions accept `NanPolicy::{Propagate, Ignore}`
- joining: `concatenate` / `stack` / `vstack` / `hstack`
- selection: `where_` / `nonzero` / `clip`
- sorting: `sort` / `argsort` / `unique`
- spaces: `linspace` / `logspace` / `geomspace` / `meshgrid`
- linear algebra: `dot` / `matmul` / `vdot` / `outer` / `diagonal` / `trace`
- triangular arrays: `tri` / `tri_with` / `tril` / `triu` / `diag`
- iteration: `ndindex` / `ndenumerate` / `nditer` / `Array::flat` /
  `Array::iter_axis0` / `Array::axis0_len`
- `dtype::{Scalar, Promote, CastTo, ArrayCast, AsBool}`

Intentional differences vs NumPy: no operator overloading; scalar-valued
linear algebra results are 0-D `Array`s; `dot` supports only 1-D/2-D inputs;
`divide` follows Rust `/`; CoW on write; no `out`/`where`.

Fill in modules phase-by-phase; Python bindings ship in `sdnp-py/` (Phase 8).

## Python API (Phase 8)

Import as `sdnp`. Public surface matches the Rust subset documented above,
with these binding policies:

- **0-D unwrap**: full reductions, integer basic indexing, 1-D `__iter__`, and
  scalar linear-algebra results return Python scalars — never user-visible 0-D
  `Array` instances.
- **Creation**: `sdnp.array(3)` and `shape=()` are rejected with `ValueError`.
- **Excluded from `__all__`**: `item`, buffer/broadcast helpers, `gather` /
  `scatter`, internal layout utilities (see `plan.md`).
- **Intentional gaps**: no transcendental ufuncs (`sin`, `exp`, …); `/` follows
  Rust true division semantics; `//` maps to `trunc_divide`.

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
