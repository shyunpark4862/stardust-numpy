// Reduction loops (ARCHITECTURE.md §4.4).
//
// A reduction runs on the M2 iterator with two operands, the input and an
// output that has extent 1 on every reduced axis, broadcast back to the input
// shape (stride 0 on those axes). The kernel then does  out = f(out, x).
// What the iterator hands it decides the strategy:
//   * out stride 0  -> inner-axis reduction: fold a run into one accumulator
//   * out stride != 0 -> outer-axis reduction: add a whole row into the
//     result row, element by element (vectorizes along the kept axis)
// The binary operations are M4's (ops::Add, ops::Maximum, ...).
//
// M7.5: a float/complex sum folds each run with numpy's pairwise summation
// (8 accumulators in blocks of 128, halves added recursively): faster, since
// 8 independent chains replace one, and more accurate, since the rounding
// error grows like log n instead of n. max/min fold with NEON FMAX/FMIN.
#pragma once

#include "kernels/elementwise.hpp"
#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace sd {

// ---- pairwise summation (numpy's DOUBLE/CDOUBLE_pairwise_sum) --------------

inline constexpr std::int64_t kPairwiseBlock = 128;  // numpy's PW_BLOCKSIZE, in doubles

// Sum of get(begin), ..., get(begin + n - 1).
template <typename Get>
double pairwise_sum(std::int64_t begin, std::int64_t n, const Get& get) {
    if (n < 8) {
        double res = 0.;
        for (std::int64_t i = 0; i < n; ++i) res += get(begin + i);
        return res;
    }
    if (n <= kPairwiseBlock) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = get(begin + j);
        std::int64_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += get(begin + i + j);  // 8 independent chains
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += get(begin + i);
        return res;
    }
    std::int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(begin, n2, get) + pairwise_sum(begin + n2, n - n2, get);
}

// The complex version: real and imaginary parts summed side by side. numpy
// counts doubles, so a block holds 64 complex numbers and 4 accumulators.
template <typename Get>
complex128 pairwise_sum_complex(std::int64_t begin, std::int64_t n, const Get& get) {
    if (n < 4) {
        double rr = 0., ri = 0.;
        for (std::int64_t i = 0; i < n; ++i) {
            const complex128 z = get(begin + i);
            rr += z.real();
            ri += z.imag();
        }
        return {rr, ri};
    }
    if (n <= kPairwiseBlock / 2) {
        complex128 r[4];
        for (int j = 0; j < 4; ++j) r[j] = get(begin + j);
        std::int64_t i = 4;
        for (; i < n - (n % 4); i += 4)
            for (int j = 0; j < 4; ++j) {
                const complex128 z = get(begin + i + j);
                r[j] = {r[j].real() + z.real(), r[j].imag() + z.imag()};
            }
        double rr = (r[0].real() + r[1].real()) + (r[2].real() + r[3].real());
        double ri = (r[0].imag() + r[1].imag()) + (r[2].imag() + r[3].imag());
        for (; i < n; ++i) {
            const complex128 z = get(begin + i);
            rr += z.real();
            ri += z.imag();
        }
        return {rr, ri};
    }
    std::int64_t n2 = n;  // numpy: n2 = (2n)/2 - ((2n)/2) % 8 doubles
    n2 -= n2 % 8;
    n2 /= 2;
    const complex128 a = pairwise_sum_complex(begin, n2, get), b = pairwise_sum_complex(begin + n2, n - n2, get);
    return {a.real() + b.real(), a.imag() + b.imag()};
}

// ---- the same sums on many threads, with the same bits (M8) ----------------
// pairwise_sum's recursion tree depends on n only. Its subtrees of at most
// kLeaf elements are independent: they are summed in parallel, then combined
// along the very same tree. The result is bit-for-bit the serial one (and
// numpy's), for any thread count.

inline constexpr std::int64_t kLeaf = parallel::kTaskWork;

template <typename Split, typename SumLeaf, typename Add>
auto pairwise_parallel(std::int64_t n, std::int64_t leaf, Split split, SumLeaf sum_leaf, Add add) {
    std::vector<std::pair<std::int64_t, std::int64_t>> leaves;  // (begin, n), left to right
    auto collect = [&](auto&& self, std::int64_t b, std::int64_t m) -> void {
        if (m <= leaf) {
            leaves.emplace_back(b, m);
            return;
        }
        const std::int64_t m2 = split(m);
        self(self, b, m2);
        self(self, b + m2, m - m2);
    };
    collect(collect, 0, n);
    using R = decltype(sum_leaf(0, 0));
    std::vector<R> part(leaves.size());
    parallel::parallel_for(static_cast<std::int64_t>(leaves.size()),
                           [&](std::int64_t t) { part[t] = sum_leaf(leaves[t].first, leaves[t].second); });
    size_t next = 0;
    auto combine = [&](auto&& self, std::int64_t m) -> R {
        if (m <= leaf) return part[next++];
        const std::int64_t m2 = split(m);
        const R left = self(self, m2);
        return add(left, self(self, m - m2));
    };
    return combine(combine, n);
}

template <typename Get>
double pairwise_sum_mt(std::int64_t n, const Get& get) {
    if (n < parallel::kThreshold || !parallel::available()) return pairwise_sum(0, n, get);
    return pairwise_parallel(
        n, kLeaf, [](std::int64_t m) { return m / 2 - (m / 2) % 8; },
        [&](std::int64_t b, std::int64_t m) { return pairwise_sum(b, m, get); },
        [](double a, double b) { return a + b; });
}

template <typename Get>
complex128 pairwise_sum_complex_mt(std::int64_t n, const Get& get) {
    if (n < parallel::kThreshold || !parallel::available()) return pairwise_sum_complex(0, n, get);
    return pairwise_parallel(
        n, kLeaf / 2, [](std::int64_t m) { return (m - m % 8) / 2; },
        [&](std::int64_t b, std::int64_t m) { return pairwise_sum_complex(b, m, get); },
        [](complex128 a, complex128 b) { return complex128{a.real() + b.real(), a.imag() + b.imag()}; });
}

// ---- float and complex products in fixed blocks (M10+) --------------------
// A product folded left to right is one chain of multiplies, 4 cycles each
// (prod of 1M float64: 0.78 ms, as numpy). Instead a run is cut into pieces
// of kLeaf elements; within a piece, eight accumulators take every eighth
// element and are combined as ((a0*a1)*(a2*a3))*((a4*a5)*(a6*a7)); the pieces
// are then multiplied left to right. The structure depends on n only, the
// pieces may run in parallel, and the bits are the same for any thread count
// -- but not numpy's: the order of the multiplies changed (user decision
// 2026-11-10). It can also over- or underflow at different points than a
// left-to-right product of the same numbers.

template <typename T, typename Get>
T product_piece(std::int64_t b, std::int64_t m, const Get& get) {
    if (m < 8) {
        if (m == 0) return T(1);
        T r = get(b);
        for (std::int64_t i = 1; i < m; ++i) r = ops::Multiply::f(r, get(b + i));
        return r;
    }
    T a[8];
    for (int j = 0; j < 8; ++j) a[j] = get(b + j);
    std::int64_t i = 8;
    for (; i + 8 <= m; i += 8)
        for (int j = 0; j < 8; ++j) a[j] = ops::Multiply::f(a[j], get(b + i + j));
    for (int j = 0; i < m; ++i, ++j) a[j] = ops::Multiply::f(a[j], get(b + i));
    for (int w = 1; w < 8; w *= 2)
        for (int j = 0; j < 8; j += 2 * w) a[j] = ops::Multiply::f(a[j], a[j + w]);
    return a[0];
}

template <typename T, typename Get>
T product_mt(std::int64_t n, const Get& get) {
    const std::int64_t pieces = std::max<std::int64_t>((n + kLeaf - 1) / kLeaf, 1);
    auto piece = [&](std::int64_t t) { return product_piece<T>(t * kLeaf, std::min(kLeaf, n - t * kLeaf), get); };
    T r = piece(0);
    if (n < parallel::kThreshold || !parallel::available()) {
        for (std::int64_t t = 1; t < pieces; ++t) r = ops::Multiply::f(r, piece(t));
        return r;
    }
    auto part = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(pieces));
    parallel::parallel_for(pieces - 1, [&](std::int64_t t) { part[t + 1] = piece(t + 1); });
    for (std::int64_t t = 1; t < pieces; ++t) r = ops::Multiply::f(r, part[t]);
    return r;
}

// Folds that do not depend on the order at all -- wrapping integer + and *,
// max/min, and/or -- are cut into fixed pieces, folded in parallel, and the
// piece results folded in order.
template <typename Op, typename T>
inline constexpr bool order_free = std::is_same_v<Op, ops::Maximum> || std::is_same_v<Op, ops::Minimum> ||
                                   std::is_same_v<Op, ops::LogicalOr> || std::is_same_v<Op, ops::LogicalAnd> ||
                                   ((std::is_same_v<Op, ops::Add> || std::is_same_v<Op, ops::Multiply>) &&
                                    (ops::is_int<T> || ops::is_bool<T>));

// ---- max/min of a contiguous run with NEON ---------------------------------

template <bool Max>
double neon_fold(const double* x, std::int64_t n, double acc) {
    std::int64_t i = 0;
    if (n >= 8) {
        // Four independent registers (8 lanes), like the 8 sum accumulators:
        // FMAX's latency is hidden behind the other three.
        float64x2_t v[4];
        for (int j = 0; j < 4; ++j) v[j] = vld1q_f64(x + 2 * j);
        for (i = 8; i + 8 <= n; i += 8)
            for (int j = 0; j < 4; ++j) {
                const float64x2_t y = vld1q_f64(x + i + 2 * j);
                v[j] = Max ? vmaxq_f64(v[j], y) : vminq_f64(v[j], y);
            }
        const float64x2_t m = Max ? vmaxq_f64(vmaxq_f64(v[0], v[1]), vmaxq_f64(v[2], v[3]))
                                  : vminq_f64(vminq_f64(v[0], v[1]), vminq_f64(v[2], v[3]));
        const double lanes = Max ? vmaxvq_f64(m) : vminvq_f64(m);  // across the two lanes
        acc = Max ? ops::fmax_nan(acc, lanes) : ops::fmin_nan(acc, lanes);
    }
    for (; i < n; ++i) acc = Max ? ops::fmax_nan(acc, x[i]) : ops::fmin_nan(acc, x[i]);
    return acc;
}

// int64 max/min: NEON has no 64-bit integer max/min instruction, so each
// step is a compare and a select, ~5 cycles of latency. The plain loop was
// one dependency chain (1.6x slower than numpy, measured after M10); four
// registers still left it latency-bound (no change, measured), eight
// independent registers (16 lanes) keep enough steps in flight.
template <bool Max>
std::int64_t neon_fold_int(const std::int64_t* x, std::int64_t n, std::int64_t acc) {
    constexpr int kRegs = 8;
    std::int64_t i = 0;
    if (n >= 2 * kRegs) {
        int64x2_t v[kRegs];
        for (int j = 0; j < kRegs; ++j) v[j] = vld1q_s64(x + 2 * j);
        auto pick = [](int64x2_t a, int64x2_t b) {
            return Max ? vbslq_s64(vcgtq_s64(a, b), a, b) : vbslq_s64(vcgtq_s64(a, b), b, a);
        };
        for (i = 2 * kRegs; i + 2 * kRegs <= n; i += 2 * kRegs)
            for (int j = 0; j < kRegs; ++j) v[j] = pick(v[j], vld1q_s64(x + i + 2 * j));
        for (int w = kRegs / 2; w >= 1; w /= 2)
            for (int j = 0; j < w; ++j) v[j] = pick(v[j], v[j + w]);
        for (const std::int64_t lane : {vgetq_lane_s64(v[0], 0), vgetq_lane_s64(v[0], 1)})
            acc = Max ? std::max(acc, lane) : std::min(acc, lane);
    }
    for (; i < n; ++i) acc = Max ? std::max(acc, x[i]) : std::min(acc, x[i]);
    return acc;
}

// args: x, out (read and written)
template <typename Op, typename T>
void reduce_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    std::byte* o = args[1];
    if (s[1] == 0) {  // inner-axis: fold the run into one value
        T acc = load_as<T>(o);
        auto at = [&](std::int64_t i) { return load_as<T>(x + i * s[0]); };
        if constexpr (ops::is_bool<T> && (std::is_same_v<Op, ops::LogicalOr> || std::is_same_v<Op, ops::LogicalAnd>)) {
            // any/all stop as soon as the answer is known: any at the first
            // True, all at the first False. Checked once per block of 4096,
            // so the block itself stays a branch-free (vectorized) fold. A
            // later run of the same output returns at once (M10+).
            constexpr bool decided = std::is_same_v<Op, ops::LogicalOr>;
            for (std::int64_t off = 0; off < n && acc != decided; off += 4096) {
                const std::int64_t m = std::min<std::int64_t>(4096, n - off);
                bool part = !decided;
                if (s[0] == 1) {
                    // Folded as bytes: with `bool` the compiler turned the OR
                    // into a find-the-first-true loop, one element per branch
                    // (0.9 ns/element, measured); on uint8_t it is a plain vector OR.
                    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(x) + off;
                    std::uint8_t bits = decided ? 0 : 1;
                    for (std::int64_t i = 0; i < m; ++i) bits = decided ? (bits | p[i]) : (bits & p[i]);
                    part = bits != 0;
                } else {
                    for (std::int64_t i = 0; i < m; ++i) part = Op::f(part, at(off + i));
                }
                acc = Op::f(acc, part);
            }
            *reinterpret_cast<T*>(o) = acc;
            return;
        }
        if constexpr (order_free<Op, T>) {
            if (n >= parallel::kThreshold && parallel::available()) {
                // A long run with one output (a full reduction): pieces in parallel.
                const std::int64_t pieces = (n + parallel::kTaskWork - 1) / parallel::kTaskWork;
                auto part = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(pieces));  // not vector<bool>
                parallel::parallel_for(pieces, [&](std::int64_t t) {
                    const std::int64_t begin = t * parallel::kTaskWork;
                    const std::int64_t m = std::min(parallel::kTaskWork, n - begin);
                    const std::byte* first = x + begin * s[0];
                    part[t] = load_as<T>(first);  // each piece folds onto its own first element
                    std::byte* piece[] = {const_cast<std::byte*>(first + s[0]), reinterpret_cast<std::byte*>(&part[t])};
                    const std::int64_t piece_strides[] = {s[0], 0};
                    reduce_loop<Op, T>(piece, m - 1, piece_strides, nullptr);  // serial: we are inside a task
                });
                for (std::int64_t t = 0; t < pieces; ++t) acc = Op::f(acc, part[t]);
                *reinterpret_cast<T*>(o) = acc;
                return;
            }
        }
        if constexpr (std::is_same_v<Op, ops::Add> && ops::is_float<T>) {
            acc = acc + pairwise_sum_mt(n, at);
        } else if constexpr (std::is_same_v<Op, ops::Add> && ops::is_cplx<T>) {
            const complex128 p = pairwise_sum_complex_mt(n, at);
            acc = {acc.real() + p.real(), acc.imag() + p.imag()};
        } else if constexpr (std::is_same_v<Op, ops::Multiply> && (ops::is_float<T> || ops::is_cplx<T>)) {
            if (s[0] == sizeof(T)) {
                const T* xp = reinterpret_cast<const T*>(x);
                acc = Op::f(acc, product_mt<T>(n, [xp](std::int64_t i) { return xp[i]; }));
            } else {
                acc = Op::f(acc, product_mt<T>(n, at));
            }
        } else if constexpr ((std::is_same_v<Op, ops::Maximum> || std::is_same_v<Op, ops::Minimum>) &&
                             ops::is_float<T>) {
            if (s[0] == sizeof(T)) acc = neon_fold<std::is_same_v<Op, ops::Maximum>>(reinterpret_cast<const T*>(x), n, acc);
            else for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
        } else if constexpr ((std::is_same_v<Op, ops::Maximum> || std::is_same_v<Op, ops::Minimum>) &&
                             ops::is_int<T>) {
            if (s[0] == sizeof(T)) {
                acc = neon_fold_int<std::is_same_v<Op, ops::Maximum>>(reinterpret_cast<const T*>(x), n, acc);
            } else {
                for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
            }
        } else if (s[0] == sizeof(T)) {
            const T* xp = reinterpret_cast<const T*>(x);  // indexed: integer folds vectorize
            for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, xp[i]);
        } else {
            for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
        }
        *reinterpret_cast<T*>(o) = acc;
        return;
    }
    if (s[0] == sizeof(T) && s[1] == sizeof(T)) {  // outer-axis, contiguous rows: o = f(o, x)
        T* op = reinterpret_cast<T*>(o);
        const T* xp = reinterpret_cast<const T*>(x);
        if constexpr (neon_binary<Op, T>::available) neon_binary<Op, T>::run(op, xp, op, n);
        else for (std::int64_t i = 0; i < n; ++i) op[i] = Op::f(op[i], xp[i]);
        return;
    }
    for (std::int64_t i = 0; i < n; ++i, x += s[0], o += s[1])
        *reinterpret_cast<T*>(o) = Op::f(load_as<T>(o), load_as<T>(x));
}

// Sum of squared deviations, the second pass of var: out += |x - m|^2.
// args: x, m (the mean, broadcast), out (float64)
template <typename T>
void sq_dev_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    const std::byte* m = args[1];
    std::byte* o = args[2];
    auto sq = [&](std::int64_t i) {
        const T d = load_as<T>(x + i * s[0]) - load_as<T>(m + i * s[1]);
        if constexpr (ops::is_cplx<T>) return d.real() * d.real() + d.imag() * d.imag();
        else return d * d;
    };
    if (s[2] == 0) {  // inner-axis: pairwise, like sum
        *reinterpret_cast<double*>(o) += pairwise_sum_mt(n, sq);
        return;
    }
    for (std::int64_t i = 0; i < n; ++i) *reinterpret_cast<double*>(o + i * s[2]) += sq(i);
}

// ---- argmin / argmax over one lane (index order, so ties keep the first) ----

// float64, contiguous: two lanes, each keeping its best value and where it
// was. Strict > keeps the first index within a lane; between the lanes a tie
// goes to the smaller index. NaNs are only noted on the way (any NaN wins,
// the first one) and then located by a scalar rescan, as they are rare.
template <bool Max>
std::int64_t neon_arg_extreme(const double* x, std::int64_t n) {
    float64x2_t best = vld1q_f64(x);
    int64x2_t at = {0, 1}, idx = {0, 1};
    const int64x2_t two = vdupq_n_s64(2);
    uint64x2_t ordered = vceqq_f64(best, best);  // all-ones while a lane has seen no NaN (x == x)
    std::int64_t i = 2;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t v = vld1q_f64(x + i);
        idx = vaddq_s64(idx, two);
        const uint64x2_t better = Max ? vcgtq_f64(v, best) : vcltq_f64(v, best);
        best = vbslq_f64(better, v, best);
        at = vbslq_s64(better, idx, at);
        ordered = vandq_u64(ordered, vceqq_f64(v, v));
    }
    if (vminvq_u32(vreinterpretq_u32_u64(ordered)) == 0)  // some lane met a NaN
        for (std::int64_t k = 0;; ++k)
            if (std::isnan(x[k])) return k;
    double b0 = vgetq_lane_f64(best, 0), b1 = vgetq_lane_f64(best, 1);
    std::int64_t a0 = vgetq_lane_s64(at, 0), a1 = vgetq_lane_s64(at, 1);
    double b = b0;
    std::int64_t a = a0;
    if (Max ? (b1 > b0 || (b1 == b0 && a1 < a0)) : (b1 < b0 || (b1 == b0 && a1 < a0))) {
        b = b1;
        a = a1;
    }
    for (; i < n; ++i) {  // the odd last element
        if (std::isnan(x[i])) return i;
        if (Max ? x[i] > b : x[i] < b) {
            b = x[i];
            a = i;
        }
    }
    return a;
}

template <typename T, bool Max>
std::int64_t arg_extreme(const std::byte* p, std::int64_t n, std::int64_t stride) {
    if constexpr (ops::is_float<T>) {
        if (stride == sizeof(double) && n >= 2) return neon_arg_extreme<Max>(reinterpret_cast<const double*>(p), n);
    }
    T best = load_as<T>(p);
    std::int64_t at = 0;
    if constexpr (ops::is_float<T>) {
        if (std::isnan(best)) return 0;  // the first NaN wins outright
    }
    p += stride;
    for (std::int64_t i = 1; i < n; ++i, p += stride) {
        const T v = load_as<T>(p);
        if constexpr (ops::is_float<T>) {
            if (std::isnan(v)) return i;
        }
        if (Max ? v > best : v < best) {
            best = v;
            at = i;
        }
    }
    return at;
}

// ---- cumulative sum / product along one lane ----
// The first output is the first input itself (not identity + x), so -0.0
// survives, as in numpy.

template <typename Op, typename T>
void scan_lane(const std::byte* x, std::int64_t xs, std::byte* o, std::int64_t os, std::int64_t n) {
    if (n == 0) return;
    T acc = load_as<T>(x);
    *reinterpret_cast<T*>(o) = acc;
    for (std::int64_t i = 1; i < n; ++i) {
        x += xs;
        o += os;
        acc = Op::f(acc, load_as<T>(x));
        *reinterpret_cast<T*>(o) = acc;
    }
}

}  // namespace sd
