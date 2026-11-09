# stardust-numpy

NumPy의 일부를 처음부터 다시 만든 교육용 프로젝트다 ("rebuilding the wheel").
- 계산 코어는 C++20, Python 바인딩은 nanobind로 만들었다.
- 대상 기기는 Apple M4 Pro 한 대다. NEON, Accelerate(BLAS, vForce)를 쓰고, 스레드풀은 직접 만들었다.
- 설계를 먼저 확정하고, 마일스톤마다 구현하고 측정하고 해설을 썼다.

```python
import stardust_numpy as sdnp

x = sdnp.arange(12.0).reshape((3, 4))
y = sdnp.sin(x.T) @ x            # vForce, BLAS
x[x > 5] = 0.0                    # boolean mask scatter
sdnp.sum(x, axis=0), x.argmax(axis=1), sdnp.sort(x, descending=True)
```

## 무엇이 다른가
API는 Python Array API Standard를 기준으로 했고, numpy의 비일관성 일부를 고쳤다. 자세한 내용은 [API.md](API.md)에 있다.
- **0차원 배열이 없다.** 결과가 스칼라면 Python 스칼라를 돌려준다.
- **정수 연산은 실패하지 않는다.**
  - `a // 0 == 0`, `a % 0 == a`
  - 오버플로는 wrap한다.
  - 경고를 내지 않는다.
- **산술 연산에서 bool은 int64로 계산한다.**
- **암묵적 하향 변환은 에러다.** 대입, `out=`, in-place 연산 모두 해당한다.
- **dtype은 4개다**: `bool`, `int64`, `float64`, `complex128`. 승격 순서는 일직선이다.
- **결과는 스레드 수와 무관하게 비트까지 같다.**

## 빌드와 테스트
macOS (Apple Silicon), Python 3.15에서 확인했다.
```bash
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core pytest hypothesis numpy
```
```bash
.venv/bin/pip install --no-build-isolation -e .
```
```bash
.venv/bin/python -m pytest -q
```

## 성능 (원소 100만 개, 자세한 내용은 [최종 보고서](docs/learn/M10-final-report.md))

| | 1스레드 / numpy | 10스레드 / numpy |
|---|---:|---:|
| `f + g` | 0.99배 | 0.20배 |
| `sin(f)` | 0.20배 | 0.024배 |
| `sum(f)` | 0.95배 | 0.21배 |
| `sort(f)` | 0.53배 | 0.53배 |
| int64 행렬곱 300³ | 0.63배 | 0.095배 |
| 작은 배열 `x[1, 2]` | 1.8배 | 1.8배 |

배율은 numpy 시간 대비다. 1보다 작으면 우리가 빠르다.

## 문서
- 설계: [DESIGN.md](DESIGN.md), [API.md](API.md), [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md)
- 마일스톤별 해설: [docs/learn/](docs/learn/README.md)
- 벤치마크: `bench/`, 측정 기록은 `bench/results/`
- 독립 실험: `experiments/`
