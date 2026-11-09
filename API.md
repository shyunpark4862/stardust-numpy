# stardust-numpy 공개 API

> 상태: **확정**
>
> 원소별 함수의 구현 방식은 DESIGN.md §2.6을 따른다.

## 0. 시그니처 규칙
- **positional-only (`/` 앞)**
  - 배열 인자
  - 그 함수의 "본 인자"
    - 예: `reshape`의 `shape`, `full`의 `fill_value`, `permute_dims`의 `axes`
- **keyword-only (`*` 뒤)**
  - 이름이 붙은 옵션은 항상 keyword-only다.
  - 옵션 이름: `dtype`, `axis`, `keepdims`, `out`, `copy`, `stable`, `descending`, `correction`, `min`, `max`, `k`, `endpoint`, `include_initial`
- **이름**
  - Array API에 있는 함수는 Array API 이름을 쓴다.
  - Array API에 없는 함수만 numpy 이름을 쓴다 (`ravel`).
- **dtype 지정**: dtype 객체만 받는다.
  - 사용 가능한 객체: `sdnp.bool`, `sdnp.int64`, `sdnp.float64`, `sdnp.complex128`
  - Python 타입(`float` 등)이나 문자열은 받지 않는다.
- **0차원 처리**: 결과가 0차원이 되면 Python 스칼라를 반환한다 (Z1).

## 1. 모듈 상수와 dtype
```
bool, int64, float64, complex128     # dtype 객체
e, pi, inf, nan                      # Python float
newaxis                              # = None
```

## 2. 배열 생성
```
asarray(obj, /, *, dtype=None, copy=None)
zeros(shape, /, *, dtype=float64)
ones(shape, /, *, dtype=float64)
empty(shape, /, *, dtype=float64)
full(shape, fill_value, /, *, dtype=None)       # dtype을 생략하면 fill_value의 타입을 따른다
zeros_like(x, /, *, dtype=None)
ones_like(x, /, *, dtype=None)
empty_like(x, /, *, dtype=None)
full_like(x, fill_value, /, *, dtype=None)
arange(start, /, stop=None, step=1, *, dtype=None)
linspace(start, stop, num, /, *, dtype=None, endpoint=True)   # num은 필수
eye(n_rows, n_cols=None, /, *, k=0, dtype=float64)
```
- `array` 함수는 없다. 변환은 `asarray`로 하고, 복사가 필요하면 `asarray(x, copy=True)`를 쓴다.
- `asarray`가 받는 입력
  - 중첩 시퀀스
  - ndarray
  - 버퍼 프로토콜 객체 (예: numpy 배열)
  - Python 스칼라를 넣으면 0차원이 되므로 에러다 (Z2).
- 중첩 리스트의 타입 추론
  - 원소 중 가장 높은 kind를 따른다.
  - 들쭉날쭉한(ragged) 리스트는 에러다.
  - `[]`는 shape `(0,)`, dtype float64가 된다.

## 3. shape 조작과 뷰
```
reshape(x, shape, /, *, copy=None)
ravel(x, /)                              # 가능하면 뷰를 반환한다
permute_dims(x, axes, /)
matrix_transpose(x, /)                   # .mT와 같다
moveaxis(x, source, destination, /)
squeeze(x, /, *, axis)                   # axis 필수
expand_dims(x, /, *, axis)               # axis는 int 또는 tuple, 필수
broadcast_to(x, shape, /)
broadcast_arrays(*arrays)
astype(x, dtype, /, *, copy=True)
concat(arrays, /, *, axis=0)
stack(arrays, /, *, axis=0)
```

## 4. 인덱싱
- **기본 인덱스**: `int`, `slice`, `...`, `newaxis`. 결과는 뷰다.
- **고급 인덱스**: 정수 배열(리스트 포함), bool 마스크. 결과는 복사본이다.
  - numpy의 의미론을 따른다.
  - 단, 고급 인덱스들이 basic 인덱스를 사이에 두고 떨어져 있는 조합은 `IndexError`로 막는다.
    - numpy는 이 경우 결과 축을 맨 앞으로 옮기는데, 이 규칙을 없애기 위해서다.
    - 허용되는 나머지 경우는 결과가 numpy와 같다.
- **인덱스 범위**: 범위 밖 인덱스는 `IndexError`다. 음수 인덱스는 허용한다.
- **대입**: `x[idx] = value`
  - `value`는 브로드캐스팅된다.
  - 하향 캐스팅은 에러다 (S6과 같은 규칙).

## 5. 원소별 함수 (ufunc)
모든 원소별 함수는 `out=`를 keyword-only로 받는다. `where=` 마스크 인자는 없다.

```
# 산술
add  subtract  multiply  divide  floor_divide  remainder  pow
negative  positive  abs  square  sqrt  sign
# 지수와 로그
exp  expm1  log  log1p  log2  log10
# 삼각함수와 쌍곡선함수
sin  cos  tan  asin  acos  atan  atan2  sinh  cosh  tanh  asinh  acosh  atanh
# 반올림
floor  ceil  trunc  round            # round는 half-to-even
# 비교
equal  not_equal  less  less_equal  greater  greater_equal
# 논리
logical_and  logical_or  logical_xor  logical_not
# 비트 (int64, bool)
bitwise_and  bitwise_or  bitwise_xor  bitwise_invert
bitwise_left_shift  bitwise_right_shift
# 기타
maximum  minimum  isnan  isinf  isfinite
real  imag  conj                     # complex 입력에 대해 real/imag는 stride를 이용한 뷰를 반환한다
clip(x, /, *, min=None, max=None)
where(condition, x1, x2, /)
```
- ufunc 객체는 없다. `add.reduce` 같은 메서드도 없다.
- `abs`에 complex를 넣으면 float64를 반환한다.
- `divide`는 정수 입력도 float64로 계산한다.
- **시프트 양**은 부호 없는 수로 해석한다.
  - 시프트 양이 64 이상이면(음수도 여기에 포함된다) 모든 비트가 밀려난다.
  - 그래서 왼쪽 시프트는 `0`이 되고, 오른쪽 시프트는 부호 비트로 채워져 `0` 또는 `-1`이 된다.

## 6. 리덕션
```
sum(x, /, *, axis=None, dtype=None, keepdims=False)
prod(x, /, *, axis=None, dtype=None, keepdims=False)
min(x, /, *, axis=None, keepdims=False)
max(x, /, *, axis=None, keepdims=False)
mean(x, /, *, axis=None, keepdims=False)
var(x, /, *, axis=None, correction=0, keepdims=False)
std(x, /, *, axis=None, correction=0, keepdims=False)
argmin(x, /, *, axis=None, keepdims=False)     # axis는 int만 받는다
argmax(x, /, *, axis=None, keepdims=False)
any(x, /, *, axis=None, keepdims=False)
all(x, /, *, axis=None, keepdims=False)
cumulative_sum(x, /, *, axis=None, dtype=None, include_initial=False)
cumulative_prod(x, /, *, axis=None, dtype=None, include_initial=False)
```
- **결과 dtype**
  - `sum`, `prod`: bool은 int64로 승격한다. 나머지는 입력 dtype을 유지한다.
  - `mean`, `var`, `std`: bool과 int는 float64로 계산한다.
  - `var`, `std`에 complex를 넣으면 결과는 float64다.
- **빈 배열**
  - `sum`은 `0`, `prod`는 `1`이다.
  - `any`는 `False`, `all`은 `True`다.
  - `mean`, `var`, `std`는 `nan`이다.
  - `min`, `max`, `argmin`, `argmax`는 `ValueError`다.

## 7. 정렬, 집합, 선형대수
```
sort(x, /, *, axis=-1, descending=False, stable=True)
argsort(x, /, *, axis=-1, descending=False, stable=True)
unique_values(x, /)
unique_counts(x, /)      # -> (values, counts)
unique_inverse(x, /)     # -> (values, inverse_indices)
unique_all(x, /)         # -> (values, indices, inverse_indices, counts)
matmul(x1, x2, /)
```
- `unique_*`
  - 입력을 평탄화해서 처리한다.
  - 결과는 정렬해서 반환한다.
  - NaN은 서로 다른 값으로 취급한다.
  - complex도 허용한다. 결과 순서는 (실수부, 허수부)의 사전순이다.

## 8. ndarray
**속성**
```
dtype  shape  ndim  size  strides  itemsize  nbytes  base  T  mT
```
- `strides`의 단위는 바이트다.

**메서드**
```
reshape  astype  copy  tolist
sum  prod  min  max  mean  var  std  argmin  argmax  any  all
```

**연산자**
- 산술: `+  -  *  /  //  %  **  @`
- 단항: `-x`, `+x`, `abs(x)`, `~x`
- 비트: `&  |  ^  <<  >>`
- 비교: `==  !=  <  <=  >  >=`
- 위 연산자들의 in-place 버전
- `~`를 bool에 쓰면 논리 부정이다.

**파이썬 프로토콜**
- `len(x)`는 `shape[0]`이다.
- 반복(iteration)은 축 0을 따라간다. 1차원 배열을 반복하면 Python 스칼라가 나온다.
- `repr`와 `str`을 지원한다.
- 버퍼 프로토콜을 지원한다.
  - 내보내기와 가져오기를 모두 지원한다.
  - numpy와 복사 없이 상호 변환할 수 있다.
- **TypeError**가 나는 연산
  - 배열을 스칼라로 바꾸는 변환: `bool(x)`, `int(x)`, `float(x)`, `complex(x)`, `operator.index(x)` (크기와 상관없이 항상)
  - 해싱
- `divmod`와 `in`은 지원하지 않는다.

## 9. 실행 설정
```
set_num_threads(n, /)    # n >= 1
get_num_threads()
```
- 기본값은 P코어 수(10)다.
- 스레드 수를 바꿔도 결과는 비트 단위로 같다. 리덕션도 결정적이다.
