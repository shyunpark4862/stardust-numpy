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
| 비교 경로 수 | 130 (양쪽 모두 존재하는 키만) |

### 측정 방법론

- Rust와 NumPy 벤치는 **순차 실행**한다. 동시 실행 시 CPU 간섭으로 결과가 왜곡된다.
- 표의 시간은 Criterion **median** / NumPy **7회 측정 median**(초 → µs 변환)이다.
- **비율** = SDNP ÷ NumPy. **1.00× 이하**이면 SDNP가 같거나 빠름(✓).
- View 계열은 allocator·복사 비용이 없어 µs 미만으로 매우 작게 나올 수 있다.
- Iteration 계열은 iterator 생성뿐 아니라 전체 logical 원소를 checksum으로 소비한다.

## Executive Summary

| 지표 | 값 |
|------|-----|
| 비교 경로 | 130 |
| SDNP 우세·동률 (≤ 1.00×) | **67** / 130 (52%) |
| SDNP 열세 (> 1.00×) | 63 / 130 |

### 주요 결과

- **전체 최저 비율:** `flat · contiguous 1M` 0.02×.
- **전체 최고 비율:** `matmul · contiguous 256×256` 24.80×.
- **Phase 6 선형대수·대각선:** 14개 경로, 중앙값 1.78×, SDNP 우세·동률 2개.
- **Phase 7 iteration:** 6개 경로, 중앙값 0.18×, SDNP 우세·동률 6개.
- **squeeze·astype:** 6개 경로, 중앙값 0.60×, SDNP 우세·동률 4개.
- **NanPolicy::Ignore:** 10개 경로, 중앙값 0.58×, SDNP 우세·동률 8개.

## NanPolicy `Propagate` 회귀 확인

정책 추가 전의 마지막 측정값과 같은 경로를 선택 재측정했다. 양수는 느려짐,
음수는 빨라짐이며 작은 변동은 측정 노이즈 범위다.

| Propagate 경로 | 정책 추가 전 (µs) | 현재 (µs) | 변화 |
| --- | --- | --- | --- |
| sum · last axis contiguous | 80.17 | 80.25 | +0.1% |
| sum · first axis fixed stride | 128.2 | 126.5 | -1.3% |
| sum · multi-axis general | 575.9 | 558.0 | -3.1% |
| prod · last axis | 762.6 | 94.00 | -87.7% |
| prod · first axis fixed stride | 127.8 | 127.6 | -0.2% |
| var · last axis contiguous | 242.3 | 244.9 | +1.1% |
| min · last axis | 155.2 | 153.9 | -0.8% |
| argmin · last axis | 938.4 | 935.1 | -0.4% |
| cumsum · last axis contiguous | 1206.1 | 1156.9 | -4.1% |
| cumsum · first axis strided | 6994.9 | 6711.6 | -4.1% |

## 카테고리별 요약

| 카테고리 | 경로 수 | SDNP ≤ NumPy | 평균 비율 | 중앙값 비율 |
| --- | --- | --- | --- | --- |
| Cumulative | 4 | 2 | 0.77× | 0.75× |
| Diagonal | 2 | 1 | 1.33× | 1.33× |
| Indexing | 9 | 0 | 1.92× | 1.61× |
| Iteration | 6 | 6 | 0.25× | 0.18× |
| Join | 5 | 2 | 1.16× | 1.02× |
| Linalg | 8 | 1 | 11.21× | 10.83× |
| NaN policy | 10 | 8 | 1.16× | 0.58× |
| Reduction | 31 | 17 | 1.16× | 0.94× |
| Selection | 5 | 4 | 0.94× | 0.79× |
| Sorting | 4 | 2 | 1.14× | 0.99× |
| Spaces | 2 | 1 | 0.79× | 0.79× |
| Triangle | 4 | 0 | 1.70× | 1.49× |
| Ufunc | 23 | 10 | 0.92× | 1.03× |
| View·복사 | 7 | 5 | 0.76× | 0.64× |
| 생성 | 4 | 4 | 0.97× | 0.99× |
| 신규 기능 | 6 | 4 | 0.95× | 0.60× |

## SDNP 우세 Top 10 (비율 낮을수록 SDNP가 빠름)

| 카테고리 | 경로 | SDNP (µs) | NumPy (µs) | 비율 |
| --- | --- | --- | --- | --- |
| Iteration | flat · contiguous 1M | 548.8 | 24610.3 | **0.02×** ✓ |
| View·복사 | broadcast_to view | 0.04 | 0.98 | **0.04×** ✓ |
| Spaces | meshgrid · 1024×1024 view | 0.21 | 3.741 | **0.06×** ✓ |
| Iteration | flat · strided 1M | 1789.6 | 26659.5 | **0.07×** ✓ |
| View·복사 | broadcast_arrays | 0.17 | 1.722 | **0.10×** ✓ |
| Reduction | prod · last axis | 94.00 | 761.3 | **0.12×** ✓ |
| Ufunc | add · contiguous 32×32 | 0.20 | 1.257 | **0.16×** ✓ |
| Iteration | ndenumerate · contiguous 1M | 18739.0 | 107992.5 | **0.17×** ✓ |
| Iteration | nditer · broadcast 1024×1024 | 16662.0 | 88140.5 | **0.19×** ✓ |
| NaN policy | prod Ignore · suffix | 248.7 | 1085.9 | **0.23×** ✓ |

## 개선 우선순위 Top 10 (SDNP가 느린 순)

| 카테고리 | 경로 | SDNP (µs) | NumPy (µs) | 비율 |
| --- | --- | --- | --- | --- |
| Linalg | matmul · contiguous 256×256 | 2071.6 | 83.53 | 24.80× |
| Linalg | matmul · contiguous 128×128 | 242.6 | 12.34 | 19.66× |
| Linalg | matmul · strided 128×128 | 243.1 | 12.37 | 19.65× |
| Linalg | matmul · batch 8×64×64 | 252.7 | 14.63 | 17.27× |
| NaN policy | min Ignore · suffix | 430.2 | 87.02 | 4.94× |
| Reduction | sum · multi-axis general | 558.0 | 121.5 | 4.59× |
| Linalg | matmul · contiguous 32×32 | 5.052 | 1.151 | 4.39× |
| Indexing | scatter array · shared RHS | 210.0 | 51.06 | 4.11× |
| 신규 기능 | astype · strided f64→i64 | 2329.1 | 843.3 | 2.76× |
| NaN policy | sum Ignore · general | 1203.3 | 456.4 | 2.64× |

## 전체 결과

| 카테고리 | 경로 | SDNP (µs) | NumPy (µs) | 비율 |
| --- | --- | --- | --- | --- |
| Cumulative | cumprod · first axis strided | 7442.0 | 6327.9 | 1.18× |
| Cumulative | cumprod · last axis | 1273.4 | 2835.4 | **0.45×** ✓ |
| Cumulative | cumsum · first axis strided | 6711.6 | 6348.6 | 1.06× |
| Cumulative | cumsum · last axis contiguous | 1156.9 | 2813.1 | **0.41×** ✓ |
| Diagonal | diag build · 1024 | 35.78 | 38.46 | **0.93×** ✓ |
| Diagonal | diag extract · 1024×1024 | 0.69 | 0.40 | 1.74× |
| Indexing | basic half slice view | 0.13 | 0.11 | 1.21× |
| Indexing | boolean mask gather | 2721.6 | 1179.4 | 2.31× |
| Indexing | fancy gather · multidim | 1512.9 | 655.3 | 2.31× |
| Indexing | reverse slice view | 0.13 | 0.10 | 1.30× |
| Indexing | scalar scatter · fancy | 1413.8 | 590.5 | 2.39× |
| Indexing | scalar scatter · strided | 128.4 | 128.2 | 1.00× |
| Indexing | scatter array · shared RHS | 210.0 | 51.06 | 4.11× |
| Indexing | scatter array · strided basic | 231.3 | 143.7 | 1.61× |
| Indexing | scatter array · unshared RHS | 53.98 | 51.31 | 1.05× |
| Iteration | axis-0 iteration · 1024 rows | 34.15 | 46.76 | **0.73×** ✓ |
| Iteration | flat · contiguous 1M | 548.8 | 24610.3 | **0.02×** ✓ |
| Iteration | flat · strided 1M | 1789.6 | 26659.5 | **0.07×** ✓ |
| Iteration | ndenumerate · contiguous 1M | 18739.0 | 107992.5 | **0.17×** ✓ |
| Iteration | ndindex · 1024×1024 | 17820.0 | 55135.2 | **0.32×** ✓ |
| Iteration | nditer · broadcast 1024×1024 | 16662.0 | 88140.5 | **0.19×** ✓ |
| Join | concatenate · axis 0 contiguous | 209.9 | 214.0 | **0.98×** ✓ |
| Join | concatenate · axis 0 strided → C | 2583.5 | 1918.0 | 1.35× |
| Join | concatenate · axis 1 contiguous | 227.6 | 491.2 | **0.46×** ✓ |
| Join | concatenate · axis 1 strided → C | 3889.2 | 1945.4 | 2.00× |
| Join | stack · axis 0 contiguous | 214.0 | 210.4 | 1.02× |
| Linalg | dot · 1M vector | 141.7 | 77.87 | 1.82× |
| Linalg | matmul · batch 8×64×64 | 252.7 | 14.63 | 17.27× |
| Linalg | matmul · contiguous 128×128 | 242.6 | 12.34 | 19.66× |
| Linalg | matmul · contiguous 256×256 | 2071.6 | 83.53 | 24.80× |
| Linalg | matmul · contiguous 32×32 | 5.052 | 1.151 | 4.39× |
| Linalg | matmul · strided 128×128 | 243.1 | 12.37 | 19.65× |
| Linalg | outer · 1024×1024 | 417.7 | 343.7 | 1.22× |
| Linalg | trace · 1024×1024 | 0.64 | 0.74 | **0.87×** ✓ |
| NaN policy | argmin Ignore · suffix | 512.6 | 853.9 | **0.60×** ✓ |
| NaN policy | cumsum Ignore · contiguous | 1665.8 | 3115.4 | **0.53×** ✓ |
| NaN policy | cumsum Ignore · strided | 5951.5 | 8538.2 | **0.70×** ✓ |
| NaN policy | mean Ignore · suffix | 247.3 | 631.7 | **0.39×** ✓ |
| NaN policy | min Ignore · suffix | 430.2 | 87.02 | 4.94× |
| NaN policy | prod Ignore · suffix | 248.7 | 1085.9 | **0.23×** ✓ |
| NaN policy | sum Ignore · general | 1203.3 | 456.4 | 2.64× |
| NaN policy | sum Ignore · prefix | 358.0 | 460.9 | **0.78×** ✓ |
| NaN policy | sum Ignore · suffix | 249.1 | 450.5 | **0.55×** ✓ |
| NaN policy | var Ignore · suffix | 465.3 | 1864.5 | **0.25×** ✓ |
| Reduction | all · first axis fixed stride | 79.25 | 290.8 | **0.27×** ✓ |
| Reduction | all · last axis | 75.40 | 282.3 | **0.27×** ✓ |
| Reduction | any · bool last axis | 10.23 | 4.727 | 2.16× |
| Reduction | any · first axis fixed stride | 79.29 | 291.2 | **0.27×** ✓ |
| Reduction | any · last axis | 75.71 | 273.7 | **0.28×** ✓ |
| Reduction | argmax · last axis | 939.0 | 536.3 | 1.75× |
| Reduction | argmin · last axis | 935.1 | 521.2 | 1.79× |
| Reduction | max · bool first axis fixed stride | 16.46 | 21.79 | **0.76×** ✓ |
| Reduction | max · bool last axis contiguous | 10.26 | 7.151 | 1.43× |
| Reduction | max · first axis fixed stride | 262.8 | 136.6 | 1.92× |
| Reduction | max · i64 first axis fixed stride | 252.1 | 135.4 | 1.86× |
| Reduction | max · i64 last axis contiguous | 116.3 | 87.62 | 1.33× |
| Reduction | max · last axis | 157.5 | 87.47 | 1.80× |
| Reduction | mean · first axis fixed stride | 127.2 | 137.1 | **0.93×** ✓ |
| Reduction | mean · last axis | 87.01 | 128.7 | **0.68×** ✓ |
| Reduction | min · bool first axis fixed stride | 16.54 | 22.05 | **0.75×** ✓ |
| Reduction | min · bool last axis contiguous | 10.49 | 7.666 | 1.37× |
| Reduction | min · first axis fixed stride | 258.8 | 137.6 | 1.88× |
| Reduction | min · i64 first axis fixed stride | 252.0 | 135.8 | 1.86× |
| Reduction | min · i64 last axis contiguous | 115.6 | 87.57 | 1.32× |
| Reduction | min · last axis | 153.9 | 85.50 | 1.80× |
| Reduction | prod · first axis fixed stride | 127.6 | 134.5 | **0.95×** ✓ |
| Reduction | prod · last axis | 94.00 | 761.3 | **0.12×** ✓ |
| Reduction | std · first axis fixed stride | 279.5 | 714.8 | **0.39×** ✓ |
| Reduction | std · last axis | 240.9 | 662.0 | **0.36×** ✓ |
| Reduction | sum · first axis fixed stride | 126.5 | 134.4 | **0.94×** ✓ |
| Reduction | sum · last axis contiguous | 80.25 | 124.2 | **0.65×** ✓ |
| Reduction | sum · multi-axis general | 558.0 | 121.5 | 4.59× |
| Reduction | sum · total contiguous | 91.57 | 121.1 | **0.76×** ✓ |
| Reduction | var · first axis fixed stride | 280.5 | 721.4 | **0.39×** ✓ |
| Reduction | var · last axis contiguous | 244.9 | 654.8 | **0.37×** ✓ |
| Selection | clip · contiguous | 141.3 | 268.1 | **0.53×** ✓ |
| Selection | nonzero · bool contiguous | 1735.9 | 1994.9 | **0.87×** ✓ |
| Selection | where · contiguous | 236.8 | 300.7 | **0.79×** ✓ |
| Selection | where · scalar broadcast | 201.3 | 281.7 | **0.71×** ✓ |
| Selection | where · strided | 2308.5 | 1266.0 | 1.82× |
| Sorting | argsort · last axis contiguous | 1252.5 | 627.9 | 1.99× |
| Sorting | sort · last axis contiguous | 514.6 | 413.0 | 1.25× |
| Sorting | sort · last axis strided → C | 4014.1 | 6920.9 | **0.58×** ✓ |
| Sorting | unique · flattened | 13695.0 | 18785.6 | **0.73×** ✓ |
| Spaces | linspace · 1M | 747.0 | 491.1 | 1.52× |
| Spaces | meshgrid · 1024×1024 view | 0.21 | 3.741 | **0.06×** ✓ |
| Triangle | tri · 1024×1024 | 830.5 | 346.9 | 2.39× |
| Triangle | tril · contiguous 1024×1024 | 559.0 | 384.6 | 1.45× |
| Triangle | tril · strided 1024×1024 | 1786.0 | 1262.0 | 1.42× |
| Triangle | triu · contiguous 1024×1024 | 568.7 | 374.3 | 1.52× |
| Ufunc | absolute contiguous | 134.7 | 116.1 | 1.16× |
| Ufunc | add · contiguous 1M 1-D | 207.0 | 205.3 | 1.01× |
| Ufunc | add · contiguous 256×256 | 12.94 | 10.70 | 1.21× |
| Ufunc | add · contiguous 32×32 | 0.20 | 1.257 | **0.16×** ✓ |
| Ufunc | add · contiguous 64×128×128 | 207.6 | 208.3 | **1.00×** ✓ |
| Ufunc | add · contiguous × contiguous | 208.6 | 210.0 | **0.99×** ✓ |
| Ufunc | add · contiguous × strided | 1242.3 | 1097.9 | 1.13× |
| Ufunc | add · i64 + f64 | 215.5 | 503.3 | **0.43×** ✓ |
| Ufunc | add · strided × strided | 2890.2 | 2132.8 | 1.36× |
| Ufunc | divide contiguous | 208.8 | 202.2 | 1.03× |
| Ufunc | divide · i64 fallible | 1169.7 | 784.9 | 1.49× |
| Ufunc | divide · i64 fallible strided | 2922.8 | 2925.7 | **1.00×** ✓ |
| Ufunc | greater contiguous | 145.9 | 500.1 | **0.29×** ✓ |
| Ufunc | isnan contiguous | 78.48 | 76.45 | 1.03× |
| Ufunc | multiply · column broadcast | 138.5 | 276.6 | **0.50×** ✓ |
| Ufunc | multiply · row broadcast | 143.7 | 312.4 | **0.46×** ✓ |
| Ufunc | multiply · scalar broadcast | 139.0 | 122.4 | 1.14× |
| Ufunc | negative contiguous | 134.2 | 119.0 | 1.13× |
| Ufunc | power contiguous | 4003.1 | 3811.0 | 1.05× |
| Ufunc | remainder contiguous | 1665.1 | 2589.3 | **0.64×** ✓ |
| Ufunc | subtract · contiguous | 212.6 | 204.5 | 1.04× |
| Ufunc | subtract · strided × strided | 2813.8 | 2137.6 | 1.32× |
| Ufunc | trunc_divide contiguous | 210.1 | 353.6 | **0.59×** ✓ |
| View·복사 | broadcast_arrays | 0.17 | 1.722 | **0.10×** ✓ |
| View·복사 | broadcast_to view | 0.04 | 0.98 | **0.04×** ✓ |
| View·복사 | copy contiguous | 100.9 | 101.1 | **1.00×** ✓ |
| View·복사 | copy transposed → C-order | 1372.5 | 863.7 | 1.59× |
| View·복사 | reshape contiguous view | 0.04 | 0.08 | **0.44×** ✓ |
| View·복사 | to_vec transposed | 1353.3 | 878.9 | 1.54× |
| View·복사 | transpose view | 0.03 | 0.05 | **0.64×** ✓ |
| 생성 | eye 1024 | 35.26 | 39.78 | **0.89×** ✓ |
| 생성 | full 1024×1024 | 62.92 | 62.99 | **1.00×** ✓ |
| 생성 | ones 1024×1024 | 62.75 | 63.47 | **0.99×** ✓ |
| 생성 | zeros 1024×1024 | 30.49 | 30.89 | **0.99×** ✓ |
| 신규 기능 | astype · complex→f64 | 198.3 | 262.0 | **0.76×** ✓ |
| 신규 기능 | astype · contiguous f64→i64 | 121.4 | 279.8 | **0.43×** ✓ |
| 신규 기능 | astype · same dtype copy | 99.75 | 98.37 | 1.01× |
| 신규 기능 | astype · strided f64→i64 | 2329.1 | 843.3 | 2.76× |
| 신규 기능 | squeeze · all singleton axes | 0.06 | 0.17 | **0.33×** ✓ |
| 신규 기능 | squeeze · selected axes | 0.08 | 0.20 | **0.43×** ✓ |

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
