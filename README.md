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

## Layout

```
src/
  dtype.rs       Scalar + Promote + CastTo + AsBool
  shape.rs       size / strides / contiguity / buffer_index
  error.rs
  array/         Array<T>, element get/set (CoW), views (transpose/reshape)
  index/         IndexSpec, bounds helpers, gather / scatter
  create.rs      zeros/ones/full/arange/eye
  broadcast.rs   broadcast_shape / broadcast_to / …
  ufunc/         kernels + ops + traits (internals pub(crate))
  stride_iter.rs StrideIter (pub(crate); public API in Phase 7)
  reduce/        sum/prod/min/max/mean/arg/cum/var/std/any/all
  join.rs        concatenate / stack / vstack / hstack
  select.rs      where_ / nonzero / clip
  sort.rs        sort / argsort / unique
  linalg.rs      Phase 6 (pub(crate) stub)
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
- reductions: `sum`/`prod`/`min`/`max`/`mean`/`argmin`/`argmax`/`cumsum`/`cumprod`/`var`/`std`/`any`/`all` (`axis`, `keepdims`)
- joining: `concatenate` / `stack` / `vstack` / `hstack`
- selection: `where_` / `nonzero` / `clip`
- sorting: `sort` / `sort_in_place` / `argsort` / `unique`
- spaces: `linspace` / `logspace` / `geomspace` / `meshgrid`
- `dtype::{Scalar, Promote, CastTo, AsBool}`

Intentional differences vs NumPy: no operator overloading; `divide` follows Rust `/`; CoW on write; no `out`/`where`.

Fill in modules phase-by-phase; Python bindings are Phase 8.
