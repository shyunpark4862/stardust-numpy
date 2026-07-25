# SDNP Paths 벤치마크 리포트

> 자동 생성: `benches/.bench-rust.log` + `benches/.bench-numpy.json`
> 생성일: 2026-07-26
> 재생성: `python benches/generate_benchmark_report.py --skip-run`

## 측정 환경

| 항목 | 값 |
|------|-----|
| 플랫폼 | arm64 (Apple Silicon) |
| NumPy | 2.5.1 |
| Python | 3.12.11 |
| Rust 벤치 | `cargo bench --bench paths` (Criterion) |
| NumPy 벤치 | `python benches/numpy_paths.py` |
| 비교 경로 수 | 92 (양쪽 모두 존재하는 키만) |

### 측정 방법론

- Rust와 NumPy 벤치는 **순차 실행**한다. 동시 실행 시 CPU 간섭으로 결과가 왜곡된다.
- 표의 시간은 Criterion **median** / NumPy **단일 측정값**(초 → µs 변환)이다.
- **비율** = SDNP ÷ NumPy. **1.00× 이하**이면 SDNP가 같거나 빠름(✓).
- View 계열은 allocator·복사 비용이 없어 µs 미만으로 매우 작게 나올 수 있다.

## Executive Summary

| 지표 | 값 |
|------|-----|
| 비교 경로 | 92 |
| SDNP 우세·동률 (≤ 1.00×) | **42** / 92 (46%) |
| SDNP 열세 (> 1.00×) | 50 / 92 |

### 주요 결과

- **Reduction first axis:** prefix `TraversalSchedule` 덕분에 `sum/mean/prod · first axis`, `any/all · first axis`, `var/std · first axis`가 NumPy 대비 **0.28–0.97×**.
- **Reduction last axis:** `sum · last axis` **0.66×**, `var/std · last axis` **~0.40×** — suffix 8-lane + two-pass var가 효과적.
- **min/max:** f64/i64 last axis는 여전히 NumPy **~1.8×** 열세. first axis prefix는 f64 **~1.9×**, bool은 **0.77×**.
- **최대 격차:** `sum · multi-axis general` **5.02×**, `scatter array · shared RHS` **4.12×**, `sort · last axis contiguous` **2.90×**.
- **View·Spaces:** `meshgrid · 1024×1024 view` **0.06×**, broadcast/view 경로 전반 우세.

## 카테고리별 요약

| 카테고리 | 경로 수 | SDNP ≤ NumPy | 평균 비율 | 중앙값 비율 |
| --- | --- | --- | --- | --- |
| Cumulative | 4 | 2 | 0.74× | 0.75× |
| Indexing | 9 | 0 | 1.92× | 1.63× |
| Join | 5 | 1 | 1.20× | 1.01× |
| Reduction | 30 | 17 | 1.20× | 0.96× |
| Selection | 5 | 3 | 1.00× | 0.80× |
| Sorting | 4 | 2 | 1.60× | 1.43× |
| Spaces | 2 | 1 | 0.80× | 0.80× |
| Ufunc | 22 | 11 | 0.96× | 1.00× |
| View·복사 | 7 | 4 | 0.87× | 0.67× |
| 생성 | 4 | 1 | 1.03× | 1.06× |

## SDNP 우세 Top 10 (비율 낮을수록 SDNP가 빠름)

| 카테고리 | 경로 | SDNP (µs) | NumPy (µs) | 비율 |
| --- | --- | --- | --- | --- |
| View·복사 | broadcast_to view | 0.03 | 0.99 | **0.04×** ✓ |
| Spaces | meshgrid · 1024×1024 view | 0.21 | 3.702 | **0.06×** ✓ |
| View·복사 | broadcast_arrays | 0.13 | 1.747 | **0.07×** ✓ |
| Ufunc | add · contiguous 32×32 | 0.21 | 1.245 | **0.17×** ✓ |
| Reduction | all · first axis fixed stride | 79.65 | 287.4 | **0.28×** ✓ |
| Reduction | any · first axis fixed stride | 80.28 | 287.9 | **0.28×** ✓ |
| Reduction | all · last axis | 78.19 | 278.9 | **0.28×** ✓ |
| Reduction | any · last axis | 77.15 | 271.3 | **0.28×** ✓ |
| Ufunc | greater contiguous | 146.2 | 505.2 | **0.29×** ✓ |
| Reduction | std · last axis | 257.1 | 654.3 | **0.39×** ✓ |

## 개선 우선순위 Top 10 (SDNP가 느린 순)

| 카테고리 | 경로 | SDNP (µs) | NumPy (µs) | 비율 |
| --- | --- | --- | --- | --- |
| Reduction | sum · multi-axis general | 611.7 | 121.9 | 5.02× |
| Indexing | scatter array · shared RHS | 215.3 | 52.31 | 4.12× |
| Sorting | sort · last axis contiguous | 1152.5 | 397.5 | 2.90× |
| Indexing | scalar scatter · fancy | 1403.2 | 557.0 | 2.52× |
| Indexing | fancy gather · multidim | 1504.6 | 655.1 | 2.30× |
| Sorting | argsort · last axis contiguous | 1248.7 | 591.0 | 2.11× |
| Indexing | boolean mask gather | 2462.0 | 1178.9 | 2.09× |
| Reduction | min · i64 first axis fixed stride | 267.6 | 135.1 | 1.98× |
| Join | concatenate · axis 1 strided → C | 3866.9 | 1978.3 | 1.95× |
| Reduction | max · i64 first axis fixed stride | 262.3 | 135.0 | 1.94× |

## 전체 결과

| 카테고리 | 경로 | SDNP (µs) | NumPy (µs) | 비율 |
| --- | --- | --- | --- | --- |
| Cumulative | cumprod · first axis strided | 6456.3 | 6245.1 | 1.03× |
| Cumulative | cumprod · last axis | 1353.2 | 2814.2 | **0.48×** ✓ |
| Cumulative | cumsum · first axis strided | 6510.5 | 6438.0 | 1.01× |
| Cumulative | cumsum · last axis contiguous | 1214.3 | 2813.8 | **0.43×** ✓ |
| Indexing | basic half slice view | 0.13 | 0.11 | 1.23× |
| Indexing | boolean mask gather | 2462.0 | 1178.9 | 2.09× |
| Indexing | fancy gather · multidim | 1504.6 | 655.1 | 2.30× |
| Indexing | reverse slice view | 0.13 | 0.10 | 1.30× |
| Indexing | scalar scatter · fancy | 1403.2 | 557.0 | 2.52× |
| Indexing | scalar scatter · strided | 129.6 | 122.2 | 1.06× |
| Indexing | scatter array · shared RHS | 215.3 | 52.31 | 4.12× |
| Indexing | scatter array · strided basic | 230.7 | 141.3 | 1.63× |
| Indexing | scatter array · unshared RHS | 54.14 | 51.42 | 1.05× |
| Join | concatenate · axis 0 contiguous | 217.7 | 217.3 | 1.00× |
| Join | concatenate · axis 0 strided → C | 2772.5 | 1868.6 | 1.48× |
| Join | concatenate · axis 1 contiguous | 283.0 | 516.7 | **0.55×** ✓ |
| Join | concatenate · axis 1 strided → C | 3866.9 | 1978.3 | 1.95× |
| Join | stack · axis 0 contiguous | 219.1 | 217.2 | 1.01× |
| Reduction | all · first axis fixed stride | 79.65 | 287.4 | **0.28×** ✓ |
| Reduction | all · last axis | 78.19 | 278.9 | **0.28×** ✓ |
| Reduction | any · first axis fixed stride | 80.28 | 287.9 | **0.28×** ✓ |
| Reduction | any · last axis | 77.15 | 271.3 | **0.28×** ✓ |
| Reduction | argmax · last axis | 962.7 | 526.2 | 1.83× |
| Reduction | argmin · last axis | 955.4 | 522.9 | 1.83× |
| Reduction | max · bool first axis fixed stride | 16.69 | 22.00 | **0.76×** ✓ |
| Reduction | max · bool last axis contiguous | 10.68 | 7.208 | 1.48× |
| Reduction | max · first axis fixed stride | 266.8 | 137.3 | 1.94× |
| Reduction | max · i64 first axis fixed stride | 262.3 | 135.0 | 1.94× |
| Reduction | max · i64 last axis contiguous | 121.4 | 87.74 | 1.38× |
| Reduction | max · last axis | 158.9 | 87.30 | 1.82× |
| Reduction | mean · first axis fixed stride | 130.7 | 137.1 | **0.95×** ✓ |
| Reduction | mean · last axis | 87.67 | 127.9 | **0.69×** ✓ |
| Reduction | min · bool first axis fixed stride | 16.94 | 21.90 | **0.77×** ✓ |
| Reduction | min · bool last axis contiguous | 10.85 | 7.678 | 1.41× |
| Reduction | min · first axis fixed stride | 264.4 | 137.0 | 1.93× |
| Reduction | min · i64 first axis fixed stride | 267.6 | 135.1 | 1.98× |
| Reduction | min · i64 last axis contiguous | 120.0 | 87.55 | 1.37× |
| Reduction | min · last axis | 160.0 | 87.24 | 1.83× |
| Reduction | prod · first axis fixed stride | 129.7 | 136.4 | **0.95×** ✓ |
| Reduction | prod · last axis | 751.0 | 760.8 | **0.99×** ✓ |
| Reduction | std · first axis fixed stride | 285.2 | 709.1 | **0.40×** ✓ |
| Reduction | std · last axis | 257.1 | 654.3 | **0.39×** ✓ |
| Reduction | sum · first axis fixed stride | 131.8 | 135.5 | **0.97×** ✓ |
| Reduction | sum · last axis contiguous | 83.49 | 125.9 | **0.66×** ✓ |
| Reduction | sum · multi-axis general | 611.7 | 121.9 | 5.02× |
| Reduction | sum · total contiguous | 92.45 | 119.7 | **0.77×** ✓ |
| Reduction | var · first axis fixed stride | 286.6 | 709.5 | **0.40×** ✓ |
| Reduction | var · last axis contiguous | 263.5 | 652.0 | **0.40×** ✓ |
| Selection | clip · contiguous | 140.6 | 268.1 | **0.52×** ✓ |
| Selection | nonzero · bool contiguous | 2056.5 | 1951.9 | 1.05× |
| Selection | where · contiguous | 238.8 | 299.3 | **0.80×** ✓ |
| Selection | where · scalar broadcast | 200.3 | 282.5 | **0.71×** ✓ |
| Selection | where · strided | 2427.6 | 1265.1 | 1.92× |
| Sorting | argsort · last axis contiguous | 1248.7 | 591.0 | 2.11× |
| Sorting | sort · last axis contiguous | 1152.5 | 397.5 | 2.90× |
| Sorting | sort · last axis strided → C | 4463.2 | 6897.6 | **0.65×** ✓ |
| Sorting | unique · flattened | 14011.0 | 18600.5 | **0.75×** ✓ |
| Spaces | linspace · 1M | 756.4 | 491.6 | 1.54× |
| Spaces | meshgrid · 1024×1024 view | 0.21 | 3.702 | **0.06×** ✓ |
| Ufunc | absolute contiguous | 143.9 | 109.9 | 1.31× |
| Ufunc | add · contiguous 1M 1-D | 201.5 | 207.3 | **0.97×** ✓ |
| Ufunc | add · contiguous 256×256 | 12.22 | 10.75 | 1.14× |
| Ufunc | add · contiguous 32×32 | 0.21 | 1.245 | **0.17×** ✓ |
| Ufunc | add · contiguous 64×128×128 | 205.9 | 207.9 | **0.99×** ✓ |
| Ufunc | add · contiguous × contiguous | 204.2 | 206.5 | **0.99×** ✓ |
| Ufunc | add · contiguous × strided | 1450.0 | 1069.6 | 1.36× |
| Ufunc | add · i64 + f64 | 214.5 | 504.1 | **0.43×** ✓ |
| Ufunc | add · strided × strided | 3163.4 | 2134.8 | 1.48× |
| Ufunc | divide contiguous | 216.0 | 214.2 | 1.01× |
| Ufunc | divide · i64 fallible | 1111.3 | 793.5 | 1.40× |
| Ufunc | greater contiguous | 146.2 | 505.2 | **0.29×** ✓ |
| Ufunc | isnan contiguous | 80.12 | 76.63 | 1.05× |
| Ufunc | multiply · column broadcast | 147.1 | 266.5 | **0.55×** ✓ |
| Ufunc | multiply · row broadcast | 206.5 | 305.3 | **0.68×** ✓ |
| Ufunc | multiply · scalar broadcast | 145.5 | 114.3 | 1.27× |
| Ufunc | negative contiguous | 142.2 | 111.1 | 1.28× |
| Ufunc | power contiguous | 4016.4 | 3798.6 | 1.06× |
| Ufunc | remainder contiguous | 1718.0 | 2603.5 | **0.66×** ✓ |
| Ufunc | subtract · contiguous | 204.6 | 206.6 | **0.99×** ✓ |
| Ufunc | subtract · strided × strided | 3243.8 | 2124.5 | 1.53× |
| Ufunc | trunc_divide contiguous | 215.8 | 355.7 | **0.61×** ✓ |
| View·복사 | broadcast_arrays | 0.13 | 1.747 | **0.07×** ✓ |
| View·복사 | broadcast_to view | 0.03 | 0.99 | **0.04×** ✓ |
| View·복사 | copy contiguous | 106.7 | 99.92 | 1.07× |
| View·복사 | copy transposed → C-order | 1594.4 | 852.0 | 1.87× |
| View·복사 | reshape contiguous view | 0.04 | 0.08 | **0.46×** ✓ |
| View·복사 | to_vec transposed | 1648.1 | 853.3 | 1.93× |
| View·복사 | transpose view | 0.04 | 0.05 | **0.67×** ✓ |
| 생성 | eye 1024 | 36.10 | 39.08 | **0.92×** ✓ |
| 생성 | full 1024×1024 | 67.02 | 62.85 | 1.07× |
| 생성 | ones 1024×1024 | 67.18 | 63.31 | 1.06× |
| 생성 | zeros 1024×1024 | 32.82 | 30.70 | 1.07× |

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

## 재현 방법

```bash
# 1) Rust (약 7분)
cargo bench --bench paths 2>&1 | tee benches/.bench-rust.log

# 2) NumPy (약 5분) — Rust 완료 후 실행
.venv/bin/python benches/numpy_paths.py
# → benches/.bench-numpy.json

# 3) 리포트 + 캔버스
python benches/generate_benchmark_report.py --skip-run
python benches/render_benchmark_canvas.py --skip-run \
  --rust-log benches/.bench-rust.log \
  --numpy-json benches/.bench-numpy.json
```

---

*이 문서는 `benches/generate_benchmark_report.py`로 생성된다. 벤치 데이터 갱신 후 `--skip-run`으로 표만 다시 만들 수 있다. 최적화 기법 본문은 `benches/optimization_techniques.md`에서 편집한다.*
