## Rust 코드에 적용된 최적화 기법

SDNP(stardust-numpy) 전역에서 사용하는 성능 전략을 서브시스템별로 정리한다. 공통 설계 원칙은 다음과 같다.

- **Contiguous fast path 우선:** logical C-order slice가 있으면 flat zip/map/`chunks_exact`로 처리하고, strided machinery는 건너뛴다.
- **Strided 경로 통합:** `CoalescedLayout` → `RunPlan` → outer traversal + fixed-stride inner run. reduction·ufunc·indexing·join·selection이 같은 인프라를 공유한다.
- **컴파일러 벡터화 유도:** platform SIMD intrinsics·`unsafe`·`#[target_feature]`는 사용하지 않는다. 8-lane partial accumulator, `chunks_exact`, tight loop 등으로 LLVM auto-vectorization / ILP를 유도한다.
- **Dispatch는 경계에서 한 번:** fast path 판정(`as_c_contiguous_slice`)과 layout coalescing은 kernel 진입 시 수행; hot loop 안에서 재귀 dispatch 없음.

---

### 1. 공통 strided 인프라

#### 1.1 `CoalescedLayout` — 축 병합과 inner run 추출

**파일:** `src/layout.rs`

N차원 strided layout을 operand stride 배열들과 함께 정규화한다.

1. **Singleton 축 제거** — size 1 축은 주소 진행에 영향 없으므로 drop (병합 blocker 제거).
2. **인접 축 병합** — 모든 operand가 `outer_stride == inner_stride × len`이면 한 run으로 merge.
3. **마지막 축 = inner run** — `inner_len`, `inner_stride` per operand. 그 위는 outer traversal.

`RunPlan`, reduction `ReducedAxisRuns`, ufunc collect, indexing scatter 등이 이 결과를 소비한다.

#### 1.2 `RunPlan` + `RunKind` — prepared run dispatch

**파일:** `src/run.rs`

`CoalescedLayout` 위에 reusable traversal plan을 구축한다. inner stride를 `RunKind`로 분류한다.

| `RunKind` | stride | inner loop 동작 |
|-----------|--------|-----------------|
| `Contiguous` | 1 | slice zip / `copy_from_slice` / `fill` |
| `Repeated` | 0 | broadcast scalar hoist, `fill(value)` |
| `Strided` | 기타 | `pos += stride` 고정 stride loop |

**API:** `RunPlan::new`, `for_each`, `for_each_element`, `try_for_each`.

**Collect helpers:** `collect_unary`, `collect_binary`, `collect_ternary`, `extend_unary`, `try_collect_binary` — operand `RunKind` 조합별 specialized match 후 generic fallback.

#### 1.3 `StrideCursor` / `StrideIter` — incremental buffer index

**파일:** `src/stride_iter.rs`

- **`StrideCursor<N>`:** multi-index + lane별 buffer offset. `advance()` carry, `reset()` 재사용. cumsum 등 input/output offset 동시 진행(`N>1`).
- **`StrideIter`:** `StrideCursor<1>` 래퍼, `ExactSizeIterator`. indexing prepare, fancy scatter, `nonzero` 등 단순 1-operand walk.

Reduction general path는 **`RunPlan` outer + `ReducedAxisRuns` inner**로 migration 중이며, `StrideCursor`는 reduced-axis cursor 재사용에 쓰인다.

#### 1.4 C-contiguity 판정 (singleton 축 무시)

**파일:** `src/shape.rs`, `src/array/mod.rs`

Broadcast로 삽입된 size-1 축의 dummy stride가 contiguous 판정을 깨뜨리지 않도록, **size 1 축은 stride 검사 생략**. non-zero offset 허용.

**API:** `is_c_contiguous`, `Array::is_c_contiguous`, `Array::as_c_contiguous_slice` — ufunc/reduction/join/indexing dispatch의 공통 gate.

#### 1.5 `#[inline]` hot-path marking

trait method·buffer index·sort compare·run helper 등 hot loop 직전 helper에 `#[inline]` 밀집 (`shape.rs`, `array/mod.rs`, `reduce/traits.rs`, `ufunc/traits.rs`, `run.rs`, `sort.rs` 등).

---

### 2. Array · View · 메모리

#### 2.1 `Arc` shared buffer + copy-on-write

**파일:** `src/array/mod.rs`, `src/array/element.rs`

View clone은 `Arc` refcount만 증가. write 시 `ensure_unique_storage_for_write`:

- strong count == 1 → in-place mutation
- 전체 buffer cover → `Arc::make_mut`
- partial view → logical C-order만 materialize 후 write

scatter, `Array::set`, in-place sort 교체 등에서 buffer alias 안전성과 copy 최소화.

#### 2.2 Zero-copy view 연산

**파일:** `src/array/view.rs`, `src/broadcast.rs`, `src/index/ops.rs`, `src/join.rs`, `src/create.rs`

`Arc::clone` + shape/stride/offset만 변경. 데이터 복사 없음.

| 연산 | 메커니즘 |
|------|----------|
| `transpose`, `permute_axes` | stride 재배열 |
| `reshape` (contiguous) | shape/strides 재해석 |
| `broadcast_to` | stretch 축 stride 0 |
| basic gather (`IndexSpec`) | offset/shape 계산 |
| `insert_axis_view` (stack) | size-1 축 + stride 0 |
| `meshgrid` | 1-D reshape view + broadcast |

#### 2.3 Contiguous materialization

**파일:** `src/array/mod.rs`

- **`copy` / `to_vec_c_order`:** contiguous → `slice.to_vec()`. else → `RunPlan` + `collect_unary`.
- **`reshape` (non-contiguous):** view 불가 시 `to_vec_c_order` 후 새 allocation.

---

### 3. Broadcast

#### 3.1 Stride-0 broadcast view

**파일:** `src/broadcast.rs`

size-1 축을 target shape에 맞춰 **stride 0 view** 생성. `broadcast_arrays`는 각 operand에 `broadcast_to`만 적용 — 데이터 복제 없음. `RunKind::Repeated`와 연동.

#### 3.2 Lazy binary alignment

**파일:** `src/ufunc/kernels.rs`, `src/select.rs`

`align_binary`: shape이 이미 같으면 `None`(원본 그대로). 다를 때만 필요한 쪽 `broadcast_to`. `where_`는 condition/x/y 3-way broadcast.

#### 3.3 Joint operand coalescing

**파일:** `src/layout.rs`

multi-operand layout merge 시 **모든 operand stride가 동시에 linear**해야 병합 (broadcast stride 0 포함). operand별 개별 merge보다 긴 inner run 확보.

---

### 4. Ufunc (element-wise)

#### 4.1 Contiguous flat zip fast path

**파일:** `src/ufunc/kernels.rs`

`map_unary` / `map_binary` / `try_map_binary`:

```text
as_c_contiguous_slice() 성공 → iterator zip/map/collect
else                         → RunPlan + collect_*
```

#### 4.2 RunPlan strided fallback + RunKind specialization

non-contiguous operand는 coalesced run walk. broadcast×contiguous, contiguous×contiguous 등 조합별 inner loop (`collect_binary` match arms).

#### 4.3 dtype별 sealed trait dispatch

**파일:** `src/ufunc/traits.rs`, `src/ufunc/ops.rs`

`ElemAdd`, `FallibleElemDiv`, `FloatClassify` 등 dtype semantics를 compile-time trait으로 분리. `impl_arith_via_ops!` macro. infallible(f64/complex) vs fallible(i64/bool) 경로 분리.

#### 4.4 Infallible vs fallible kernel split

**파일:** `src/ufunc/kernels.rs`

`map_binary`(infallible, `Result` 없음) vs `try_map_binary` + `try_collect_binary`(에러 가능 — 항상 generic stride loop). hot loop에서 `Result` branch 제거.

---

### 5. Reduction

#### 5.1 `ReducePlan` — reduction geometry 사전 계산

**파일:** `src/reduce/axis.rs`

axis normalize, `output_len`, `reduction_len`, `kept_shape`, `reduced_shape`, `output_shape`를 한 번 계산. `mean` 등에서 plan 재사용.

#### 5.2 `TraversalSchedule` — layout-aware 물리 순회 선택

**파일:** `src/reduce/axis.rs`, `src/reduce/kernels/`

C-contiguous + reduced axis block 위치에 따라:

| 스케줄 | 조건 | 순회 |
|--------|------|------|
| `SuffixContiguous` | trailing reduced block | `chunks_exact(reduction_len)` per output slot |
| `PrefixContiguous` | leading reduced block | `chunks_exact(output_len)` row scan, all slots 동시 갱신 |
| `GeneralStrided` | 그 외 / non-contiguous | `ReducedAxisRuns` + `RunPlan` |

`reduction_path`가 연산 semantics와 무관하게 schedule만으로 suffix/prefix/general kernel 선택.

#### 5.3 Suffix / prefix contiguous kernel

**Suffix (`fold_contiguous_chunks`):** last-axis reduction on C-order — slot마다 contiguous chunk fold.

**Prefix (`fold_prefix_contiguous`):** first-axis reduction — memory-order row가 output-major:

```rust
for row in slice.chunks_exact(plan.output_len) {
    for (acc, &value) in out.iter_mut().zip(row) {
        *acc = accumulate(*acc, value);
    }
}
```

적용: `sum`, `prod`, `mean`(합산), `any`, `all`, `var/std`(prefix two-pass), typed min/max prefix.

#### 5.4 8-lane partial accumulator (ILP)

**파일:** `src/reduce/kernels/`

loop-carried dependency 제거:

```rust
let mut partials = [initial; 8];
for block in chunk.chunks_exact(8) {
    for lane in 0..8 {
        partials[lane] = accumulate(partials[lane], block[lane]);
    }
}
// partial tree merge → remainder scalar
```

**사용처:** `reduce_sum_with_plan`, f64 min/max suffix, `converted_sum_chunk`, `squared_deviation_sum_chunk`, `merge_eight_f64`.

#### 5.5 `ReducedAxisRuns` — general strided reduction coalescing

reduced-axis shape/strides → `RunPlan<1>` coalesce → `(run_count, run_len, operand_stride)`. run grid는 `RunPlan::for_each_element`, linear run은 `pos += operand_stride`. `StrideCursor`는 reduced-axis run-grid cursor로 재사용.

**API:** `fold_strided_general`, `var_strided_general`, `extremum_strided_general`.

#### 5.6 extremum 단일 coalesced run fast path

`extremum_strided_general`에서 `reduced.run_count == 1`이면 run-counting wrapper 없이 flat inner loop — NaN branch 많은 min/max에서 overhead 제거.

#### 5.7 dtype별 min/max 전용 kernel

**파일:** `src/reduce/kernels/`, `src/reduce/traits.rs`

| dtype | kernel | 핵심 |
|-------|--------|------|
| `bool` | `reduce_bool_extremum` | min = all (`&=`), max = any (OR-equals). comparison dispatch 없음 |
| `i64` | `reduce_i64_min/max` | plain `Ord` compare, NaN 분기 없음 |
| `f64` | `reduce_f64_extremum` | 8-lane + NaN mask (아래) |

`ExtremumReduce` trait이 dtype별 dedicated kernel로 dispatch.

#### 5.8 f64 NaN semantics + 성능 (min/max / sort)

**Reduction min/max:**

- **Suffix contiguous:** chunk[0] NaN → 즉시 `NaN` push; NaN-free chunk → 8-lane partial + lane별 `nan_masks`; 결과 NaN 시 canonical `f64::NAN`.
- **Prefix / general:** logical C-order first-NaN payload 보존 (`extremum_prefix_contiguous` + `f64::is_nan`).

**Sort (`src/sort.rs`):** `SortElement for f64` — non-NaN < NaN, NaN끼리 equal (NaN-last stable ordering).

#### 5.9 `var` / `std` two-pass + schedule별 경로

- **Suffix contiguous:** chunk별 mean pass → squared deviation pass (`var_contiguous_chunks`, 8-lane sum helper).
- **Prefix contiguous:** row scan mean 누적 → 동일 순서 variance pass (`var_prefix_contiguous`).
- **General strided:** outer/reduced run walk two-pass (`var_strided_general`).

`std`는 `var` 후 `transform_owned_c_order`로 `sqrt`.

#### 5.10 `transform_owned_c_order` — in-place post-process

fresh C-contiguous reduction output에 `Arc::make_mut` 후 in-place map. `mean`(count로 나누기), `std`(sqrt)에서 재할당 방지.

#### 5.11 `AxisTraversalPlan` — 단일 축 연산 geometry

**파일:** `src/reduce/axis.rs`, `src/reduce/kernels/`

`argmin`/`argmax`/`cumsum`/`cumprod`용 kept shape/strides 사전 계산. last axis + contiguous → row chunk scan.

#### 5.12 Cumulative scan dual-path

- **Last axis contiguous:** `cumulate_flat_contiguous` — row별 prefix write.
- **Strided / first axis:** `RunPlan<2>` input+output offset lockstep (`cumulate_flat_strided`, `cumulate_axis_strided`).

순서 의존 연산이므로 `TraversalSchedule` prefix/suffix fold 미적용.

#### 5.13 Argmin / argmax

- contiguous: linear scan with index tracking.
- strided: `RunPlan` + linear counter, NaN early exit (`try_for_each`).

---

### 6. Indexing (gather / scatter)

#### 6.1 Basic gather — zero-copy view

**파일:** `src/index/ops.rs`

slice / newaxis / integer indexing → `basic_view_meta`로 offset·shape·strides 계산 → `from_shared_parts`. 복사 없음.

#### 6.2 Fancy gather — preallocated copy loop

fancy indexing은 비연속 access → `FancyOffsetIter` + `Vec::with_capacity` + push copy.

#### 6.3 `prepare_index` — layout cache

**파일:** `src/index/prepare.rs`

ellipsis / bool mask / fancy array layout을 `PreparedEntry` + `FancyLayout`으로 한 번 계산. gather/scatter 공유.

#### 6.4 Boolean mask → integer fancy

mask contiguous → linear scan + multi-index advance. else `StrideIter::for_each`.

#### 6.5 Scatter stride-1 specialization

**파일:** `src/index/ops.rs`

| 조건 | 동작 |
|------|------|
| dest C-contiguous | `slice.fill` / `copy_from_slice` |
| `RunKind::UnitStride` inner run | subslice bulk `fill` / `copy_from_slice` |
| broadcast source | single-value `fill` |

scalar scatter (`scatter_basic_scalar`)와 array scatter (`scatter_basic_array`) 모두 적용.

#### 6.6 Scatter buffer alias safety

destination과 values가 같은 buffer → `shares_buffer_with` 검사 후 `values.copy()` (`prepare_scatter_source`).

#### 6.7 Fancy scatter source zip

values contiguous → offset iter zip assign. else `StrideIter` zip.

---

### 7. Join (concatenate / stack)

#### 7.1 Concatenate outer-slab + bulk extend

**파일:** `src/join.rs`

concat axis 기준 outer index로 slab base 계산.

- source contiguous → `Vec::extend_from_slice`
- strided → slab `RunPlan` + `extend_unary`

`Vec::with_capacity`로 allocation 1회.

#### 7.2 Stack via zero-copy axis insert

`insert_axis_view`(size-1 축, stride 0) → `concatenate`. `vstack`/`hstack`는 필요 시 view promotion 후 stack.

---

### 8. Selection

#### 8.1 `where_` triple contiguous fast path

**파일:** `src/select.rs`

condition/x/y 모두 contiguous → 3-way nested zip. else `collect_ternary` RunKind specialization.

#### 8.2 `clip` — ufunc path reuse

`map_unary` 위임 → contiguous / RunPlan 경로 자동 상속.

#### 8.3 `nonzero`

C-order `StrideIter::for_each` + true 원소 coordinate push.

---

### 9. Sorting

#### 9.1 Materialize-then-sort

**파일:** `src/sort.rs`

strided view in-place sort 불가 → `to_vec()` materialize 후 flat 또는 axis-local sort.

#### 9.2 Axis-local sort + scratch reuse

`outer × inner` loop, `Vec::with_capacity(axis_len)` scratch 재사용, flat buffer 내 gather-scatter.

**API:** `sort_values_along_axis`, `argsort_along_axis`.

#### 9.3 Stable NaN-last ordering

`SortElement::sort_cmp` per dtype. f64 NaN은 항상 last.

#### 9.4 `unique` — sort + linear merge

index sort by `unique_cmp` → single-pass group merge. NaN/complex-NaN equivalence.

#### 9.5 In-place sort buffer replacement

sorted C-contiguous array로 `Array` buffer 교체 → COW shared buffer 분리 (`sort_in_place`).

---

### 10. Creation / Spaces

#### 10.1 `meshgrid` zero-copy broadcast

**파일:** `src/create.rs`

1-D input reshape(view) + `broadcast_to`. contiguous input은 buffer 공유.

#### 10.2 `linspace` overflow-safe interpolation

opposite-sign large bounds에서 `start*(1-f)+stop*f` form으로 overflow 방지 (`linear_values`).

---

### 벤치 카테고리와 기법 매핑 (참고)

| 벤치 그룹 | 주로 exercise하는 기법 |
|-----------|------------------------|
| View·복사 | zero-copy view, broadcast stride 0, contiguous copy |
| Ufunc | flat zip vs RunPlan + RunKind |
| Reduction | TraversalSchedule, 8-lane, typed min/max, ReducedAxisRuns, two-pass var |
| Cumulative | row contiguous vs strided RunPlan<2> |
| Join | `extend_from_slice` vs `extend_unary` |
| Selection | ternary collect, ufunc inheritance |
| Indexing | basic view (µs), fancy/boolean copy loops |
| Sorting | materialize + axis scratch sort |
| Spaces | meshgrid broadcast view |
