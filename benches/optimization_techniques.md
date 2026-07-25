## Rust 코드에 적용된 최적화 기법

SDNP(stardust-numpy) 전역에서 사용하는 성능 전략을 서브시스템별로 정리한다. 공통 설계 원칙은 다음과 같다.

- **Contiguous fast path 우선:** logical C-order slice가 있으면 flat zip/map/`chunks_exact`로 처리하고, strided machinery는 건너뛴다.
- **Strided 경로 통합:** `CoalescedLayout` → `RunPlan` → outer traversal + fixed-stride inner run. reduction·ufunc·indexing·join·selection이 같은 인프라를 공유한다.
- **컴파일러 벡터화 유도:** platform SIMD intrinsics·`unsafe`·`#[target_feature]`는 사용하지 않는다. 8-lane partial accumulator, `chunks_exact`, tight loop 등으로 LLVM auto-vectorization / ILP를 유도한다.
- **Dispatch는 경계에서 한 번:** fast path 판정(`as_c_contiguous_slice`)과 layout coalescing은 kernel 진입 시 수행; hot loop 안에서 재귀 dispatch 없음.

---

### 1. 공통 strided 인프라

#### 1.1 `CoalescedLayout` — 축 병합과 inner run 추출

**파일:** `src/traversal/layout.rs`

N차원 strided layout을 operand stride 배열들과 함께 정규화한다.

1. **Singleton 축 제거** — size 1 축은 주소 진행에 영향 없으므로 drop (병합 blocker 제거).
2. **인접 축 병합** — 모든 operand가 `outer_stride == inner_stride × len`이면 한 run으로 merge.
3. **마지막 축 = inner run** — `inner_len`, `inner_stride` per operand. 그 위는 outer traversal.

`RunPlan`, reduction `ReducedAxisRuns`, ufunc collect, indexing scatter 등이 이 결과를 소비한다.

#### 1.2 `RunPlan` + `RunKind` — prepared run dispatch

**파일:** `src/traversal/run.rs`

`CoalescedLayout` 위에 reusable traversal plan을 구축한다. inner stride를 `RunKind`로 분류한다.

| `RunKind` | stride | inner loop 동작 |
|-----------|--------|-----------------|
| `Contiguous` | 1 | slice zip / `copy_from_slice` / `fill` |
| `Repeated` | 0 | broadcast scalar hoist, `fill(value)` |
| `Strided` | 기타 | `pos += stride` 고정 stride loop |

**API:** `RunPlan::new`, `for_each`, `for_each_element`, `try_for_each`.

**Collect helpers:** `collect_unary`, `collect_binary`, `collect_ternary`, `extend_unary`, `try_collect_binary` — operand `RunKind` 조합별 specialized match 후 generic fallback.

#### 1.3 `StrideCursor` / `StrideIter` — incremental buffer index

**파일:** `src/traversal/stride_iter.rs`

- **`StrideCursor<N>`:** multi-index + lane별 buffer offset. `advance()` carry, `reset()` 재사용. cumsum 등 input/output offset 동시 진행(`N>1`).
- **`StrideIter`:** `StrideCursor<1>` 래퍼, `ExactSizeIterator`. indexing prepare, fancy scatter, `nonzero` 등 단순 1-operand walk.

Reduction general path는 **`RunPlan` outer + `ReducedAxisRuns` inner**로 migration 중이며, `StrideCursor`는 reduced-axis cursor 재사용에 쓰인다.

#### 1.4 C-contiguity 판정 (singleton 축 무시)

**파일:** `src/shape.rs`, `src/array/mod.rs`

Broadcast로 삽입된 size-1 축의 dummy stride가 contiguous 판정을 깨뜨리지 않도록, **size 1 축은 stride 검사 생략**. non-zero offset 허용.

**API:** `is_c_contiguous`, `Array::is_c_contiguous`, `Array::as_c_contiguous_slice` — ufunc/reduction/join/indexing dispatch의 공통 gate.

#### 1.5 `#[inline]` hot-path marking

trait method·buffer index·sort compare·run helper 등 hot loop 직전 helper에 `#[inline]` 밀집 (`shape.rs`, `array/mod.rs`, `reduction/traits.rs`, `ufunc/traits.rs`, `traversal/run.rs`, `sorting/` 등).

---

### 2. Array · View · 메모리

#### 2.1 `Arc` shared buffer + copy-on-write

**파일:** `src/array/mod.rs`, `src/array/element.rs`

View clone은 `Arc` refcount만 증가. write 시 `ensure_unique_storage_for_write`:

- strong count == 1 → in-place mutation
- 전체 buffer cover → `Arc::make_mut`
- partial view → logical C-order만 materialize 후 write

scatter와 `Array::set`에서 buffer alias 안전성과 copy 최소화.

#### 2.2 Zero-copy view 연산

**파일:** `src/array/view.rs`, `src/broadcast.rs`, `src/index/ops.rs`, `src/manipulation/`, `src/creation/`

`Arc::clone` + shape/stride/offset만 변경. 데이터 복사 없음.

| 연산 | 메커니즘 |
|------|----------|
| `transpose`, `permute_axes` | stride 재배열 |
| `reshape` (contiguous) | shape/strides 재해석 |
| `squeeze` | size-1 shape/stride metadata 제거 |
| `broadcast_to` | stretch 축 stride 0 |
| basic gather (`IndexSpec`) | offset/shape 계산 |
| `insert_axis_view` (stack) | size-1 축 + stride 0 |
| `meshgrid` | 1-D reshape view + broadcast |

#### 2.3 Contiguous materialization

**파일:** `src/array/mod.rs`

- **`copy` / `to_vec_c_order`:** contiguous → `slice.to_vec()`. else → `RunPlan` + `collect_unary`.
- **`reshape` (non-contiguous):** view 불가 시 `to_vec_c_order` 후 새 allocation.
- **`astype`:** contiguous flat map 또는 strided `RunPlan` map으로 4×4 dtype 변환 후 새 C-order output.

---

### 3. Broadcast

#### 3.1 Stride-0 broadcast view

**파일:** `src/broadcast.rs`

size-1 축을 target shape에 맞춰 **stride 0 view** 생성. `broadcast_arrays`는 각 operand에 `broadcast_to`만 적용 — 데이터 복제 없음. `RunKind::Repeated`와 연동.

#### 3.2 Lazy binary alignment

**파일:** `src/ufunc/kernels.rs`, `src/selection/ops.rs`

`align_binary`: shape이 이미 같으면 `None`(원본 그대로). 다를 때만 필요한 쪽 `broadcast_to`. `where_`는 condition/x/y 3-way broadcast.

#### 3.3 Joint operand coalescing

**파일:** `src/traversal/layout.rs`

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

`map_binary`(infallible, `Result` 없음) vs `try_map_binary` + `try_collect_binary`(에러 가능). 두 경로 모두 `UnitStride`/`Repeated` 조합을 먼저 특화하고 generic stride loop로 fallback한다.

---

### 5. Reduction

#### 5.1 `ReducePlan` — reduction geometry 사전 계산

**파일:** `src/reduction/plan.rs`

axis normalize, `output_len`, `reduction_len`, `kept_shape`, `reduced_shape`, `output_shape`를 한 번 계산. `mean` 등에서 plan 재사용.

#### 5.2 `TraversalSchedule` — layout-aware 물리 순회 선택

**파일:** `src/reduction/plan.rs`, `src/reduction/kernels/`

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

**파일:** `src/reduction/kernels/`

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

**사용처:** `reduce_associative_with_plan`(sum/prod), i64/f64 min/max suffix, `converted_sum_chunk`, `squared_deviation_sum_chunk`, `merge_eight_f64`.

#### 5.5 `ReducedAxisRuns` — general strided reduction coalescing

reduced-axis shape/strides → `RunPlan<1>` coalesce → `(run_count, run_len, operand_stride)`. run grid는 `RunPlan::for_each_element`, linear run은 `pos += operand_stride`. `StrideCursor`는 reduced-axis run-grid cursor로 재사용.

**API:** `fold_strided_general`, `var_strided_general`, `extremum_strided_general`.

#### 5.6 extremum 단일 coalesced run fast path

`extremum_strided_general`에서 `reduced.run_count == 1`이면 run-counting wrapper 없이 flat inner loop — NaN branch 많은 min/max에서 overhead 제거.

#### 5.7 dtype별 min/max 전용 kernel

**파일:** `src/reduction/kernels/`, `src/reduction/traits.rs`

| dtype | kernel | 핵심 |
|-------|--------|------|
| `bool` | `reduce_bool_extremum` | min = all (`&=`), max = any (OR-equals). comparison dispatch 없음 |
| `i64` | `reduce_i64_min/max` | 8-lane `Ord` compare, NaN 분기 없음 |
| `f64` | `reduce_f64_extremum` | 8-lane + NaN mask (아래) |

`ExtremumReduce` trait이 dtype별 dedicated kernel로 dispatch.

#### 5.8 f64 NaN semantics + 성능 (min/max / sort)

**Reduction min/max:**

- **Suffix contiguous:** chunk[0] NaN → 즉시 `NaN` push; NaN-free chunk → 8-lane partial + lane별 `nan_masks`; 결과 NaN 시 canonical `f64::NAN`.
- **Prefix / general:** logical C-order first-NaN payload 보존 (`extremum_prefix_contiguous` + `f64::is_nan`).

**Sort (`src/sorting/`):** `SortElement for f64` — non-NaN < NaN, NaN끼리 equal (NaN-last stable ordering).

#### 5.9 `var` / `std` two-pass + schedule별 경로

- **Suffix contiguous:** chunk별 mean pass → squared deviation pass (`var_contiguous_chunks`, 8-lane sum helper).
- **Prefix contiguous:** row scan mean 누적 → 동일 순서 variance pass (`var_prefix_contiguous`).
- **General strided:** outer/reduced run walk two-pass (`var_strided_general`).

`std`는 `var` 후 `transform_owned_c_order`로 `sqrt`.

#### 5.10 `transform_owned_c_order` — in-place post-process

fresh C-contiguous reduction output에 `Arc::make_mut` 후 in-place map. `mean`(count로 나누기), `std`(sqrt)에서 재할당 방지.

#### 5.11 `AxisTraversalPlan` — 단일 축 연산 geometry

**파일:** `src/reduction/plan.rs`, `src/reduction/kernels/`

`argmin`/`argmax`/`cumsum`/`cumprod`용 kept shape/strides 사전 계산. last axis + contiguous → row chunk scan.

#### 5.12 Cumulative scan dual-path

- **Last axis contiguous:** `cumulate_flat_contiguous` — row별 prefix write.
- **Strided / first axis:** `RunPlan<2>` input+output offset lockstep (`cumulate_flat_strided`, `cumulate_axis_strided`).

순서 의존 연산이므로 `TraversalSchedule` prefix/suffix fold 미적용.

#### 5.13 Argmin / argmax

- contiguous: linear scan with index tracking.
- strided: `RunPlan` + linear counter, NaN early exit (`try_for_each`).

#### 5.14 `NanPolicy` 경계 dispatch

`NanPolicy::Propagate`는 기존 dtype별 kernel body에 그대로 진입한다.
`NanPolicy::Ignore`는 suffix/prefix/general별 별도 kernel family를 사용해
NaN 검사와 valid-count 상태가 propagate hot loop에 섞이지 않는다.

- suffix: 8-lane accumulator + lane별 valid count
- prefix: output slot별 accumulator/count row scan
- general: 기존 `ReducedAxisRuns` 재사용
- var/std: valid 원소만 대상으로 기존 two-pass 유지
- cumulative: contiguous/strided별 별도 순차 scan

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

**파일:** `src/manipulation/`

concat axis 기준 outer index로 slab base 계산.

- source contiguous → `Vec::extend_from_slice`
- strided → slab `RunPlan` + `extend_unary`

`Vec::with_capacity`로 allocation 1회.

#### 7.2 Stack via zero-copy axis insert

`insert_axis_view`(size-1 축, stride 0) → `concatenate`. `vstack`/`hstack`는 필요 시 view promotion 후 stack.

---

### 8. Selection

#### 8.1 `where_` triple contiguous fast path

**파일:** `src/selection/ops.rs`

condition/x/y 모두 contiguous → 3-way nested zip. else `collect_ternary` RunKind specialization.

#### 8.2 `clip` — ufunc path reuse

`map_unary` 위임 → contiguous / RunPlan 경로 자동 상속.

#### 8.3 `nonzero`

C-order `StrideIter::for_each` + true 원소 coordinate push.

---

### 9. Sorting

#### 9.1 Single materialization then sort

**파일:** `src/sorting/`

strided view는 `to_vec()`으로 한 번 materialize한 뒤 flat 또는 axis-local
sort를 수행한다. 마지막 축은 output buffer의 contiguous chunk를 직접
정렬해 별도 axis scratch gather/scatter를 건너뛴다.

#### 9.2 Axis-local sort + scratch reuse

`outer × inner` loop, `Vec::with_capacity(axis_len)` scratch 재사용, flat buffer 내 gather-scatter.

**API:** `sort_values_along_axis`, `argsort_along_axis`.

#### 9.3 Stable NaN-last ordering

`SortElement::sort_cmp` per dtype. f64 NaN은 항상 last.

#### 9.4 `unique` — sort + linear merge

index sort by `unique_cmp` → single-pass group merge. NaN/complex-NaN equivalence.

### 10. Creation / Spaces

#### 10.1 `meshgrid` zero-copy broadcast

**파일:** `src/creation/grids.rs`, `src/creation/ranges.rs`

1-D input reshape(view) + `broadcast_to`. contiguous input은 buffer 공유.

#### 10.2 `linspace` overflow-safe interpolation

opposite-sign large bounds에서 `start*(1-f)+stop*f` form으로 overflow 방지 (`linear_values`).

---

### 11. Linear algebra / Diagonal

#### 11.1 Prepared contraction geometry

**파일:** `src/linalg/geometry.rs`, `src/linalg/kernels.rs`

`MatmulPlan`이 1-D operand의 virtual matrix 축, contraction 길이, output
shape, broadcast된 batch stride를 kernel 진입 전에 한 번 계산한다. Batch
축은 `RunPlan<2>`로 함께 순회하므로 size-1 broadcast 축은 stride 0으로
동일 matrix base를 재사용한다.

#### 11.2 Boundary dispatch and contiguous matrix rows

**파일:** `src/linalg/ops.rs`, `src/linalg/kernels.rs`

`dot`/`matmul` 진입점은 right operand의 마지막 축이 unit-stride인지 한 번
검사한다. 길이 1을 초과하는 non-unit-stride 축만 호출당 한 번 C-order로
materialize하고, 이미 적합한 배열은 `Arc` clone만 수행한다.

일반적인 C-order 경로는 `i-k-j` 순서로 순회한다. right row와 output row를
연속 slice로 잘라 8-lane chunk로 갱신하므로 hot loop의 주소
계산을 없애고 LLVM auto-vectorization을 유도한다. 진짜 strided 경로는
기존 fixed-stride scalar walk를 유지한다.

#### 11.3 Eight-lane vector contractions

matrix-vector/vector-vector의 unit-stride contraction과 `vdot`은
`chunks_exact(8)` 및 8개 독립 accumulator를 사용해 loop-carried
dependency를 줄인다. partial 결과는 `ContractElement::add`로 병합하므로
f64/i64/complex뿐 아니라 bool OR-of-AND semiring도 같은 kernel을 쓴다.

`vdot`은 `Array::to_c_order_cow`로 각 operand를 독립 판정한다. contiguous
입력은 slice를 그대로 빌리고 strided 입력만 `RunPlan`으로 logical
C-order materialize하며, promoted left value의 conjugation은 lane loop
안에서 적용한다.

`outer`도 동일한 `to_c_order_cow` 경계를 사용하므로 contiguous operand는
빌리고 strided operand만 C-order로 materialize한다.

#### 11.4 Shared diagonal geometry and fixed-stride walk

`DiagonalGeometry`가 `eye`, `diag`, `diagonal`, `trace`의 start/length
계산을 공유한다. N-D `diagonal`/`trace`는 kept-axis base를 `RunPlan`으로
순회하고 각 대각선은 `row_stride + column_stride` 고정 stride로 걷는다.
`trace`는 중간 diagonal 배열을 만들지 않고 바로 `SumReduce` accumulator에
fold한다.

#### 11.5 Triangle single-pass materialization

`tri`는 checked-size output을 한 번 할당해 row-major로 0/1을 생성한다.
`tril`/`triu`는 입력을 논리 C-order로 한 번 materialize한 뒤 matrix, row,
column counter로 마지막 두 축의 mask를 적용한다. hot loop에 linear-index
division/modulo가 없으며 별도 boolean mask 배열도 만들지 않는다.

---

### 12. NumPy-style iteration

#### 12.1 Lazy contiguous/strided dispatch

**파일:** `src/iteration/`, `src/traversal/stride_iter.rs`

`Array::flat`은 진입 시 `as_c_contiguous_slice`를 한 번 확인한다. contiguous
입력은 slice iterator로 직접 읽고, transpose·negative-stride 입력은 기존
`StrideIter`가 계산한 backing-buffer offset을 따라간다. 어떤 경로도 전체
배열을 materialize하지 않는다.

`ndenumerate`는 독립 순회 구현 대신 `NdIndex`와 `FlatIter`를 zip한다.
`NdIndex`는 indexing에서 이미 쓰는 `advance_multi_index` odometer를
재사용한다.

#### 12.2 Broadcast-once multi-operand iteration

`nditer`는 생성 시 operand를 공통 shape로 한 번 broadcast한다. 모든
operand가 contiguous면 공통 linear position으로 읽고, stride-0 또는
strided operand가 있으면 공통 multi-index에서 기존 `offset_at`으로 각
buffer 위치를 계산한다. ufunc/reduce의 `RunPlan` hot path는 변경하지 않는다.

#### 12.3 Axis-0 shared views

axis-0 iterator는 trailing shape/strides metadata를 한 번 준비하고, 각
step에서 axis-0 offset만 전진시켜 `Arc` backing buffer를 공유하는 view를
만든다. 쓰기는 기존 `Array` CoW 정책에 따라 원본과 분리된다.

---

### 벤치 카테고리와 기법 매핑 (참고)

| 벤치 그룹 | 주로 exercise하는 기법 |
|-----------|------------------------|
| View·복사 | zero-copy squeeze, broadcast stride 0, contiguous copy, astype |
| Ufunc | flat zip vs RunPlan + RunKind |
| Reduction | TraversalSchedule, 8-lane, typed min/max, NanPolicy split, two-pass var |
| Cumulative | row contiguous vs strided RunPlan<2> |
| Join | `extend_from_slice` vs `extend_unary` |
| Selection | ternary collect, ufunc inheritance |
| Indexing | basic view (µs), fancy/boolean copy loops |
| Sorting | materialize + axis scratch sort |
| Spaces | meshgrid broadcast view |
| Linalg | batch RunPlan, slice i-k-j, 8-lane dot/vdot, conditional materialization |
| Triangle·Diagonal | shared geometry, fixed-stride diagonal, counter-based mask |
| Iteration | slice/StrideIter dispatch, index-value zip, broadcast-once nditer, axis-0 views |
