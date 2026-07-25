# SDNP Paths 벤치마크 리포트

> 자동 생성: `benches/.bench-rust.log` + `benches/.bench-numpy.json`
> 생성일: 2026-07-25
> 재생성: `python benches/generate_benchmark_report.py --skip-run`

## 측정 환경

| 항목         | 값                                      |
| ------------ | --------------------------------------- |
| 플랫폼       | arm64 (Apple Silicon)                   |
| NumPy        | 2.5.1                                   |
| Python       | 3.12.11                                 |
| Rust 벤치    | `cargo bench --bench paths` (Criterion) |
| NumPy 벤치   | `python benches/numpy_paths.py`         |
| 비교 경로 수 | 92 (양쪽 모두 존재하는 키만)            |

### 측정 방법론

- Rust와 NumPy 벤치는 **순차 실행**한다. 동시 실행 시 CPU 간섭으로 결과가 왜곡된다.
- 표의 시간은 Criterion **median** / NumPy **단일 측정값**(초 → µs 변환)이다.
- **비율** = SDNP ÷ NumPy. **1.00× 이하**이면 SDNP가 같거나 빠름(✓).
- View 계열은 allocator·복사 비용이 없어 µs 미만으로 매우 작게 나올 수 있다.

## Executive Summary

| 지표                     | 값                |
| ------------------------ | ----------------- |
| 비교 경로                | 92                |
| SDNP 우세·동률 (≤ 1.00×) | **42** / 92 (46%) |
| SDNP 열세 (> 1.00×)      | 50 / 92           |

### 주요 결과

- **Reduction first axis:** prefix `TraversalSchedule` 덕분에 `sum/mean/prod · first axis`, `any/all · first axis`, `var/std · first axis`가 NumPy 대비 **0.28–0.97×**.
- **Reduction last axis:** `sum · last axis` **0.66×**, `var/std · last axis` **~0.40×** — suffix 8-lane + two-pass var가 효과적.
- **min/max:** f64/i64 last axis는 여전히 NumPy **~1.8×** 열세. first axis prefix는 f64 **~1.9×**, bool은 **0.77×**.
- **최대 격차:** `sum · multi-axis general` **5.02×**, `scatter array · shared RHS` **4.12×**, `sort · last axis contiguous` **2.90×**.
- **View·Spaces:** `meshgrid · 1024×1024 view` **0.06×**, broadcast/view 경로 전반 우세.

## 카테고리별 요약

| 카테고리   | 경로 수 | SDNP ≤ NumPy | 평균 비율 | 중앙값 비율 |
| ---------- | ------- | ------------ | --------- | ----------- |
| Cumulative | 4       | 2            | 0.74×     | 0.75×       |
| Indexing   | 9       | 0            | 1.92×     | 1.63×       |
| Join       | 5       | 1            | 1.20×     | 1.01×       |
| Reduction  | 30      | 17           | 1.20×     | 0.96×       |
| Selection  | 5       | 3            | 1.00×     | 0.80×       |
| Sorting    | 4       | 2            | 1.60×     | 1.43×       |
| Spaces     | 2       | 1            | 0.80×     | 0.80×       |
| Ufunc      | 22      | 11           | 0.96×     | 1.00×       |
| View·복사  | 7       | 4            | 0.87×     | 0.67×       |
| 생성       | 4       | 1            | 1.03×     | 1.06×       |

## SDNP 우세 Top 10 (비율 낮을수록 SDNP가 빠름)

| 카테고리  | 경로                          | SDNP (µs) | NumPy (µs) | 비율        |
| --------- | ----------------------------- | --------- | ---------- | ----------- |
| View·복사 | broadcast_to view             | 0.03      | 0.99       | **0.04×** ✓ |
| Spaces    | meshgrid · 1024×1024 view     | 0.21      | 3.702      | **0.06×** ✓ |
| View·복사 | broadcast_arrays              | 0.13      | 1.747      | **0.07×** ✓ |
| Ufunc     | add · contiguous 32×32        | 0.21      | 1.245      | **0.17×** ✓ |
| Reduction | all · first axis fixed stride | 79.65     | 287.4      | **0.28×** ✓ |
| Reduction | any · first axis fixed stride | 80.28     | 287.9      | **0.28×** ✓ |
| Reduction | all · last axis               | 78.19     | 278.9      | **0.28×** ✓ |
| Reduction | any · last axis               | 77.15     | 271.3      | **0.28×** ✓ |
| Ufunc     | greater contiguous            | 146.2     | 505.2      | **0.29×** ✓ |
| Reduction | std · last axis               | 257.1     | 654.3      | **0.39×** ✓ |

## 개선 우선순위 Top 10 (SDNP가 느린 순)

| 카테고리  | 경로                              | SDNP (µs) | NumPy (µs) | 비율  |
| --------- | --------------------------------- | --------- | ---------- | ----- |
| Reduction | sum · multi-axis general          | 611.7     | 121.9      | 5.02× |
| Indexing  | scatter array · shared RHS        | 215.3     | 52.31      | 4.12× |
| Sorting   | sort · last axis contiguous       | 1152.5    | 397.5      | 2.90× |
| Indexing  | scalar scatter · fancy            | 1403.2    | 557.0      | 2.52× |
| Indexing  | fancy gather · multidim           | 1504.6    | 655.1      | 2.30× |
| Sorting   | argsort · last axis contiguous    | 1248.7    | 591.0      | 2.11× |
| Indexing  | boolean mask gather               | 2462.0    | 1178.9     | 2.09× |
| Reduction | min · i64 first axis fixed stride | 267.6     | 135.1      | 1.98× |
| Join      | concatenate · axis 1 strided → C  | 3866.9    | 1978.3     | 1.95× |
| Reduction | max · i64 first axis fixed stride | 262.3     | 135.0      | 1.94× |

## 전체 결과

| 카테고리   | 경로                               | SDNP (µs) | NumPy (µs) | 비율        |
| ---------- | ---------------------------------- | --------- | ---------- | ----------- |
| Cumulative | cumprod · first axis strided       | 6456.3    | 6245.1     | 1.03×       |
| Cumulative | cumprod · last axis                | 1353.2    | 2814.2     | **0.48×** ✓ |
| Cumulative | cumsum · first axis strided        | 6510.5    | 6438.0     | 1.01×       |
| Cumulative | cumsum · last axis contiguous      | 1214.3    | 2813.8     | **0.43×** ✓ |
| Indexing   | basic half slice view              | 0.13      | 0.11       | 1.23×       |
| Indexing   | boolean mask gather                | 2462.0    | 1178.9     | 2.09×       |
| Indexing   | fancy gather · multidim            | 1504.6    | 655.1      | 2.30×       |
| Indexing   | reverse slice view                 | 0.13      | 0.10       | 1.30×       |
| Indexing   | scalar scatter · fancy             | 1403.2    | 557.0      | 2.52×       |
| Indexing   | scalar scatter · strided           | 129.6     | 122.2      | 1.06×       |
| Indexing   | scatter array · shared RHS         | 215.3     | 52.31      | 4.12×       |
| Indexing   | scatter array · strided basic      | 230.7     | 141.3      | 1.63×       |
| Indexing   | scatter array · unshared RHS       | 54.14     | 51.42      | 1.05×       |
| Join       | concatenate · axis 0 contiguous    | 217.7     | 217.3      | 1.00×       |
| Join       | concatenate · axis 0 strided → C   | 2772.5    | 1868.6     | 1.48×       |
| Join       | concatenate · axis 1 contiguous    | 283.0     | 516.7      | **0.55×** ✓ |
| Join       | concatenate · axis 1 strided → C   | 3866.9    | 1978.3     | 1.95×       |
| Join       | stack · axis 0 contiguous          | 219.1     | 217.2      | 1.01×       |
| Reduction  | all · first axis fixed stride      | 79.65     | 287.4      | **0.28×** ✓ |
| Reduction  | all · last axis                    | 78.19     | 278.9      | **0.28×** ✓ |
| Reduction  | any · first axis fixed stride      | 80.28     | 287.9      | **0.28×** ✓ |
| Reduction  | any · last axis                    | 77.15     | 271.3      | **0.28×** ✓ |
| Reduction  | argmax · last axis                 | 962.7     | 526.2      | 1.83×       |
| Reduction  | argmin · last axis                 | 955.4     | 522.9      | 1.83×       |
| Reduction  | max · bool first axis fixed stride | 16.69     | 22.00      | **0.76×** ✓ |
| Reduction  | max · bool last axis contiguous    | 10.68     | 7.208      | 1.48×       |
| Reduction  | max · first axis fixed stride      | 266.8     | 137.3      | 1.94×       |
| Reduction  | max · i64 first axis fixed stride  | 262.3     | 135.0      | 1.94×       |
| Reduction  | max · i64 last axis contiguous     | 121.4     | 87.74      | 1.38×       |
| Reduction  | max · last axis                    | 158.9     | 87.30      | 1.82×       |
| Reduction  | mean · first axis fixed stride     | 130.7     | 137.1      | **0.95×** ✓ |
| Reduction  | mean · last axis                   | 87.67     | 127.9      | **0.69×** ✓ |
| Reduction  | min · bool first axis fixed stride | 16.94     | 21.90      | **0.77×** ✓ |
| Reduction  | min · bool last axis contiguous    | 10.85     | 7.678      | 1.41×       |
| Reduction  | min · first axis fixed stride      | 264.4     | 137.0      | 1.93×       |
| Reduction  | min · i64 first axis fixed stride  | 267.6     | 135.1      | 1.98×       |
| Reduction  | min · i64 last axis contiguous     | 120.0     | 87.55      | 1.37×       |
| Reduction  | min · last axis                    | 160.0     | 87.24      | 1.83×       |
| Reduction  | prod · first axis fixed stride     | 129.7     | 136.4      | **0.95×** ✓ |
| Reduction  | prod · last axis                   | 751.0     | 760.8      | **0.99×** ✓ |
| Reduction  | std · first axis fixed stride      | 285.2     | 709.1      | **0.40×** ✓ |
| Reduction  | std · last axis                    | 257.1     | 654.3      | **0.39×** ✓ |
| Reduction  | sum · first axis fixed stride      | 131.8     | 135.5      | **0.97×** ✓ |
| Reduction  | sum · last axis contiguous         | 83.49     | 125.9      | **0.66×** ✓ |
| Reduction  | sum · multi-axis general           | 611.7     | 121.9      | 5.02×       |
| Reduction  | sum · total contiguous             | 92.45     | 119.7      | **0.77×** ✓ |
| Reduction  | var · first axis fixed stride      | 286.6     | 709.5      | **0.40×** ✓ |
| Reduction  | var · last axis contiguous         | 263.5     | 652.0      | **0.40×** ✓ |
| Selection  | clip · contiguous                  | 140.6     | 268.1      | **0.52×** ✓ |
| Selection  | nonzero · bool contiguous          | 2056.5    | 1951.9     | 1.05×       |
| Selection  | where · contiguous                 | 238.8     | 299.3      | **0.80×** ✓ |
| Selection  | where · scalar broadcast           | 200.3     | 282.5      | **0.71×** ✓ |
| Selection  | where · strided                    | 2427.6    | 1265.1     | 1.92×       |
| Sorting    | argsort · last axis contiguous     | 1248.7    | 591.0      | 2.11×       |
| Sorting    | sort · last axis contiguous        | 1152.5    | 397.5      | 2.90×       |
| Sorting    | sort · last axis strided → C       | 4463.2    | 6897.6     | **0.65×** ✓ |
| Sorting    | unique · flattened                 | 14011.0   | 18600.5    | **0.75×** ✓ |
| Spaces     | linspace · 1M                      | 756.4     | 491.6      | 1.54×       |
| Spaces     | meshgrid · 1024×1024 view          | 0.21      | 3.702      | **0.06×** ✓ |
| Ufunc      | absolute contiguous                | 143.9     | 109.9      | 1.31×       |
| Ufunc      | add · contiguous 1M 1-D            | 201.5     | 207.3      | **0.97×** ✓ |
| Ufunc      | add · contiguous 256×256           | 12.22     | 10.75      | 1.14×       |
| Ufunc      | add · contiguous 32×32             | 0.21      | 1.245      | **0.17×** ✓ |
| Ufunc      | add · contiguous 64×128×128        | 205.9     | 207.9      | **0.99×** ✓ |
| Ufunc      | add · contiguous × contiguous      | 204.2     | 206.5      | **0.99×** ✓ |
| Ufunc      | add · contiguous × strided         | 1450.0    | 1069.6     | 1.36×       |
| Ufunc      | add · i64 + f64                    | 214.5     | 504.1      | **0.43×** ✓ |
| Ufunc      | add · strided × strided            | 3163.4    | 2134.8     | 1.48×       |
| Ufunc      | divide contiguous                  | 216.0     | 214.2      | 1.01×       |
| Ufunc      | divide · i64 fallible              | 1111.3    | 793.5      | 1.40×       |
| Ufunc      | greater contiguous                 | 146.2     | 505.2      | **0.29×** ✓ |
| Ufunc      | isnan contiguous                   | 80.12     | 76.63      | 1.05×       |
| Ufunc      | multiply · column broadcast        | 147.1     | 266.5      | **0.55×** ✓ |
| Ufunc      | multiply · row broadcast           | 206.5     | 305.3      | **0.68×** ✓ |
| Ufunc      | multiply · scalar broadcast        | 145.5     | 114.3      | 1.27×       |
| Ufunc      | negative contiguous                | 142.2     | 111.1      | 1.28×       |
| Ufunc      | power contiguous                   | 4016.4    | 3798.6     | 1.06×       |
| Ufunc      | remainder contiguous               | 1718.0    | 2603.5     | **0.66×** ✓ |
| Ufunc      | subtract · contiguous              | 204.6     | 206.6      | **0.99×** ✓ |
| Ufunc      | subtract · strided × strided       | 3243.8    | 2124.5     | 1.53×       |
| Ufunc      | trunc_divide contiguous            | 215.8     | 355.7      | **0.61×** ✓ |
| View·복사  | broadcast_arrays                   | 0.13      | 1.747      | **0.07×** ✓ |
| View·복사  | broadcast_to view                  | 0.03      | 0.99       | **0.04×** ✓ |
| View·복사  | copy contiguous                    | 106.7     | 99.92      | 1.07×       |
| View·복사  | copy transposed → C-order          | 1594.4    | 852.0      | 1.87×       |
| View·복사  | reshape contiguous view            | 0.04      | 0.08       | **0.46×** ✓ |
| View·복사  | to_vec transposed                  | 1648.1    | 853.3      | 1.93×       |
| View·복사  | transpose view                     | 0.04      | 0.05       | **0.67×** ✓ |
| 생성       | eye 1024                           | 36.10     | 39.08      | **0.92×** ✓ |
| 생성       | full 1024×1024                     | 67.02     | 62.85      | 1.07×       |
| 생성       | ones 1024×1024                     | 67.18     | 63.31      | 1.06×       |
| 생성       | zeros 1024×1024                    | 32.82     | 30.70      | 1.07×       |

## Rust 코드에 적용된 최적화 기법

SDNP(stardust-numpy) 전역에서 사용하는 성능 전략을 서브시스템별로 정리한다. 공통 설계 원칙은 다음과 같다.

- **Contiguous fast path 우선:** logical C-order slice가 있으면 flat zip/map/`chunks_exact`로 처리하고, strided machinery는 건너뛴다.
- **Strided 경로 통합:** `CoalescedLayout` → `RunPlan` → outer traversal + fixed-stride inner run. reduction·ufunc·indexing·join·selection이 같은 인프라를 공유한다.
- **컴파일러 벡터화 유도:** platform SIMD intrinsics·`unsafe`·`#[target_feature]`는 사용하지 않는다. 8-lane partial accumulator, `chunks_exact`, tight loop 등으로 LLVM auto-vectorization / ILP를 유도한다.
- **Dispatch는 경계에서 한 번:** fast path 판정(`as_c_contiguous_slice`)과 layout coalescing은 kernel 진입 시 수행; hot loop 안에서 재귀 dispatch 없음.
