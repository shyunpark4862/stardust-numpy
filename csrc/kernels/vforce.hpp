// float64 transcendental functions through Accelerate's vForce (M9;
// DESIGN.md §2.6).
//
// vForce computes a whole array per call (vvexp(y, x, &n)) with SIMD code
// tuned for Apple chips. M0 checked the 16 functions used here: the result
// for an element does not depend on its position, the array length or
// alignment (so exp(x)[i] == exp(x[i]), and thread chunking cannot change
// results), they keep C99 Annex F special values, use no threads of their
// own, and stay within a few ULP of libm (contract: 8 ULP). expm1, log1p, cos
// and tan failed those checks and stay on libm; complex functions are libm.
//
// vForce wants contiguous arrays. A strided operand (or a broadcast scalar)
// goes through a small stack buffer first -- the "buffer casting" of M7, here
// done inside the kernel.
#pragma once

#include "kernels/scalar_ops.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstdint>

namespace sd {

using VUnary = void (*)(double*, const double*, const int*);

template <typename Op> inline constexpr VUnary vforce_unary = nullptr;
template <> inline constexpr VUnary vforce_unary<ops::Exp> = &vvexp;
template <> inline constexpr VUnary vforce_unary<ops::Log> = &vvlog;
template <> inline constexpr VUnary vforce_unary<ops::Log2> = &vvlog2;
template <> inline constexpr VUnary vforce_unary<ops::Log10> = &vvlog10;
template <> inline constexpr VUnary vforce_unary<ops::Sin> = &vvsin;
template <> inline constexpr VUnary vforce_unary<ops::Asin> = &vvasin;
template <> inline constexpr VUnary vforce_unary<ops::Acos> = &vvacos;
template <> inline constexpr VUnary vforce_unary<ops::Atan> = &vvatan;
template <> inline constexpr VUnary vforce_unary<ops::Sinh> = &vvsinh;
template <> inline constexpr VUnary vforce_unary<ops::Cosh> = &vvcosh;
template <> inline constexpr VUnary vforce_unary<ops::Tanh> = &vvtanh;
template <> inline constexpr VUnary vforce_unary<ops::Asinh> = &vvasinh;
template <> inline constexpr VUnary vforce_unary<ops::Acosh> = &vvacosh;
template <> inline constexpr VUnary vforce_unary<ops::Atanh> = &vvatanh;

// Binary functions, normalized to f(out, a, b, n) = op(a, b) elementwise.
using VBinary = void (*)(double*, const double*, const double*, const int*);

inline void vf_pow(double* o, const double* a, const double* b, const int* n) { vvpow(o, b, a, n); }  // vvpow(z, y, x): x^y
inline void vf_atan2(double* o, const double* a, const double* b, const int* n) { vvatan2(o, a, b, n); }  // atan2(y, x)

template <typename Op> inline constexpr VBinary vforce_binary = nullptr;
template <> inline constexpr VBinary vforce_binary<ops::Pow> = &vf_pow;
template <> inline constexpr VBinary vforce_binary<ops::Atan2> = &vf_atan2;

namespace vforce_detail {

inline constexpr std::int64_t kChunk = 1024;        // stack buffer, in elements (8 KB)
inline constexpr std::int64_t kMaxCall = 1 << 24;   // vForce takes an int count

// Operand i of a chunk as a contiguous pointer: in place when it already is,
// else gathered into `buf` (a stride-0 scalar fills it).
inline const double* gather(const std::byte* p, std::int64_t stride, std::int64_t m, double* buf) {
    if (stride == sizeof(double)) return reinterpret_cast<const double*>(p);
    for (std::int64_t i = 0; i < m; ++i) buf[i] = *reinterpret_cast<const double*>(p + i * stride);
    return buf;
}

inline void scatter(const double* buf, std::byte* p, std::int64_t stride, std::int64_t m) {
    for (std::int64_t i = 0; i < m; ++i) *reinterpret_cast<double*>(p + i * stride) = buf[i];
}

}  // namespace vforce_detail

// The kernel bodies: any strides, n elements.
inline void vforce_unary_run(VUnary f, const std::byte* x, std::int64_t sx, std::byte* o, std::int64_t so,
                             std::int64_t n) {
    using namespace vforce_detail;
    if (sx == sizeof(double) && so == sizeof(double)) {  // contiguous: straight to vForce
        for (std::int64_t off = 0; off < n; off += kMaxCall) {
            const int m = static_cast<int>(std::min(kMaxCall, n - off));
            f(reinterpret_cast<double*>(o) + off, reinterpret_cast<const double*>(x) + off, &m);
        }
        return;
    }
    double in[kChunk], out[kChunk];
    for (std::int64_t off = 0; off < n; off += kChunk) {
        const int m = static_cast<int>(std::min(kChunk, n - off));
        const double* src = gather(x + off * sx, sx, m, in);
        const bool direct = so == sizeof(double);
        double* dst = direct ? reinterpret_cast<double*>(o + off * so) : out;
        f(dst, src, &m);
        if (!direct) scatter(out, o + off * so, so, m);
    }
}

inline void vforce_binary_run(VBinary f, const std::byte* a, std::int64_t sa, const std::byte* b, std::int64_t sb,
                              std::byte* o, std::int64_t so, std::int64_t n) {
    using namespace vforce_detail;
    double ba[kChunk], bb[kChunk], out[kChunk];
    for (std::int64_t off = 0; off < n; off += kChunk) {
        const int m = static_cast<int>(std::min(kChunk, n - off));
        const double* pa = gather(a + off * sa, sa, m, ba);
        const double* pb = gather(b + off * sb, sb, m, bb);
        const bool direct = so == sizeof(double);
        double* dst = direct ? reinterpret_cast<double*>(o + off * so) : out;
        f(dst, pa, pb, &m);
        if (!direct) scatter(out, o + off * so, so, m);
    }
}

}  // namespace sd
