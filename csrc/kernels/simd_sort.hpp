// NEON sorting: sorting networks in registers, then bitonic merging (M7.9c).
//
// A 128-bit register holds two float64 or two int64. The sort has two phases:
//
//  1. Blocks of 16: load 8 registers and run an 8-input sorting network
//     *vertically* -- every compare-exchange is one min and one max over whole
//     registers, so both lanes are sorted at once. Each lane then holds a
//     sorted column of 8; a 2x2 transpose (zip) writes them out as two sorted
//     runs of 8. No branches at all.
//  2. Merging runs: a bitonic network merges 4 + 4 values in registers; the
//     lower 4 go out, the upper 4 stay and meet the next 4 from whichever run
//     has the smaller head. One branch per 4 outputs instead of one per value.
//
// The network compare-exchanges every pair it touches, so equal values may be
// reordered: the result is a correct sort but not a stable one. sort.cpp uses
// it only where that cannot be seen (DESIGN decision in M7.9c).
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

namespace sd::simd_sort {

// Two-lane vector operations for one element type.
struct F64 {
    using T = double;
    using V = float64x2_t;
    static constexpr T kMax = std::numeric_limits<double>::infinity();  // padding: sorts last
    static V load(const T* p) { return vld1q_f64(p); }
    static void store(T* p, V v) { vst1q_f64(p, v); }
    static V min(V a, V b) { return vminq_f64(a, b); }
    static V max(V a, V b) { return vmaxq_f64(a, b); }
    static V zip1(V a, V b) { return vzip1q_f64(a, b); }
    static V zip2(V a, V b) { return vzip2q_f64(a, b); }
    static V swap(V a) { return vextq_f64(a, a, 1); }  // (x, y) -> (y, x)
};

// NEON has no 64-bit integer min/max: compare, then select.
struct I64 {
    using T = std::int64_t;
    using V = int64x2_t;
    static constexpr T kMax = std::numeric_limits<std::int64_t>::max();
    static V load(const T* p) { return vld1q_s64(p); }
    static void store(T* p, V v) { vst1q_s64(p, v); }
    static V min(V a, V b) { return vbslq_s64(vcgtq_s64(a, b), b, a); }
    static V max(V a, V b) { return vbslq_s64(vcgtq_s64(a, b), a, b); }
    static V zip1(V a, V b) { return vzip1q_s64(a, b); }
    static V zip2(V a, V b) { return vzip2q_s64(a, b); }
    static V swap(V a) { return vextq_s64(a, a, 1); }
};

template <typename Tr>
inline void cmpx(typename Tr::V& a, typename Tr::V& b) {  // compare-exchange: a <- min, b <- max
    const typename Tr::V lo = Tr::min(a, b);
    b = Tr::max(a, b);
    a = lo;
}

// 16 values at p -> two sorted runs of 8 at p and p + 8.
template <typename Tr>
inline void sort_block16(typename Tr::T* p) {
    typename Tr::V v[8];
    for (int i = 0; i < 8; ++i) v[i] = Tr::load(p + 2 * i);
    // The optimal 19-comparator network for 8 inputs (checked exhaustively
    // with the 0-1 principle: it sorts all 256 inputs of zeros and ones).
    cmpx<Tr>(v[0], v[2]); cmpx<Tr>(v[1], v[3]); cmpx<Tr>(v[4], v[6]); cmpx<Tr>(v[5], v[7]);
    cmpx<Tr>(v[0], v[4]); cmpx<Tr>(v[1], v[5]); cmpx<Tr>(v[2], v[6]); cmpx<Tr>(v[3], v[7]);
    cmpx<Tr>(v[0], v[1]); cmpx<Tr>(v[2], v[3]); cmpx<Tr>(v[4], v[5]); cmpx<Tr>(v[6], v[7]);
    cmpx<Tr>(v[2], v[4]); cmpx<Tr>(v[3], v[5]);
    cmpx<Tr>(v[1], v[4]); cmpx<Tr>(v[3], v[6]);
    cmpx<Tr>(v[1], v[2]); cmpx<Tr>(v[3], v[4]); cmpx<Tr>(v[5], v[6]);
    // Lane 0 of v[0..7] is now a sorted column, and so is lane 1: transpose.
    for (int i = 0; i < 4; ++i) {
        Tr::store(p + 2 * i, Tr::zip1(v[2 * i], v[2 * i + 1]));
        Tr::store(p + 8 + 2 * i, Tr::zip2(v[2 * i], v[2 * i + 1]));
    }
}

// Sorts the 4 values of a bitonic sequence (x0, x1) in place.
template <typename Tr>
inline void bitonic_clean4(typename Tr::V& x0, typename Tr::V& x1) {
    cmpx<Tr>(x0, x1);  // distance 2: (0,2) and (1,3)
    const typename Tr::V p = Tr::zip1(x0, x1), q = Tr::zip2(x0, x1);  // (0,2) (1,3) -> pairs at distance 1
    const typename Tr::V lo = Tr::min(p, q), hi = Tr::max(p, q);
    x0 = Tr::zip1(lo, hi);
    x1 = Tr::zip2(lo, hi);
}

// a = (a0, a1) and b = (b0, b1) each sorted (4 values); afterwards a holds
// the lowest 4 and b the highest 4, both sorted.
template <typename Tr>
inline void merge4x4(typename Tr::V& a0, typename Tr::V& a1, typename Tr::V& b0, typename Tr::V& b1) {
    // a followed by reversed b is bitonic; one half-cleaner splits it into
    // the lower and the upper half, each again bitonic.
    typename Tr::V r0 = Tr::swap(b1), r1 = Tr::swap(b0);
    const typename Tr::V l0 = Tr::min(a0, r0), l1 = Tr::min(a1, r1);
    b0 = Tr::max(a0, r0);
    b1 = Tr::max(a1, r1);
    a0 = l0;
    a1 = l1;
    bitonic_clean4<Tr>(a0, a1);
    bitonic_clean4<Tr>(b0, b1);
}

// Merges sorted runs a[0, na) and b[0, nb) (lengths multiples of 4) into out.
template <typename Tr>
inline void merge_runs(const typename Tr::T* a, std::int64_t na, const typename Tr::T* b, std::int64_t nb,
                       typename Tr::T* out) {
    if (na == 0 || nb == 0) {
        std::copy(a, a + na, out);
        std::copy(b, b + nb, out + na);
        return;
    }
    typename Tr::V x0 = Tr::load(a), x1 = Tr::load(a + 2), y0 = Tr::load(b), y1 = Tr::load(b + 2);
    std::int64_t ia = 4, ib = 4;
    for (;;) {
        merge4x4<Tr>(x0, x1, y0, y1);  // x: lowest 4 so far, y: carried highest 4
        Tr::store(out, x0);
        Tr::store(out + 2, x1);
        out += 4;
        if (ia < na && (ib >= nb || a[ia] <= b[ib])) {  // the next 4 come from the smaller head
            x0 = Tr::load(a + ia);
            x1 = Tr::load(a + ia + 2);
            ia += 4;
        } else if (ib < nb) {
            x0 = Tr::load(b + ib);
            x1 = Tr::load(b + ib + 2);
            ib += 4;
        } else {
            break;
        }
    }
    Tr::store(out, y0);
    Tr::store(out + 2, y1);
}

// Sorts data[0, n) ascending. Needs no NaN in the input (NaNs are split off
// before; FMIN/FMAX would spread them).
template <typename Tr>
void sort(typename Tr::T* data, std::int64_t n) {
    using T = typename Tr::T;
    const std::int64_t padded = (n + 15) / 16 * 16;  // whole blocks; the padding (kMax) sorts to the end
    auto buf = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(2 * padded));
    T* src = buf.get();
    T* dst = src + padded;
    std::copy(data, data + n, src);
    std::fill(src + n, src + padded, Tr::kMax);
    for (std::int64_t i = 0; i < padded; i += 16) sort_block16<Tr>(src + i);
    for (std::int64_t width = 8; width < padded; width *= 2) {  // merge passes, ping-pong between buffers
        for (std::int64_t i = 0; i < padded; i += 2 * width) {
            const std::int64_t na = std::min(width, padded - i), nb = std::min(width, padded - i - na);
            merge_runs<Tr>(src + i, na, src + i + na, nb, dst + i);
        }
        std::swap(src, dst);
    }
    std::copy(src, src + n, data);
}

}  // namespace sd::simd_sort
