// Hand-written NEON kernels for contiguous runs (M7.3), used where the
// compiler's own vectorization of the scalar definition falls short.
//
// Every kernel here must give exactly the bits of its scalar definition in
// scalar_ops.hpp (and so of numpy): tests compare all layouts bit for bit.
//
// A 128-bit NEON register holds two float64 or two int64 (or one complex128).
#pragma once

#include "kernels/scalar_ops.hpp"

#include <arm_neon.h>

#include <cstdint>

namespace sd {

// neon_binary<Op, T>::run(a, b, out, n) for the (operation, dtype) pairs that
// have one; `available` is false elsewhere and the generic loop runs.
template <typename Op, typename T>
struct neon_binary {
    static constexpr bool available = false;
};

// maximum/minimum: FMAX/FMIN implement IEEE 754-2019 maximum/minimum in one
// instruction -- NaN propagates and -0 < +0 -- which is exactly fmax_nan.
// The scalar version needs three branches per element, which the compiler
// vectorizes into extra compares and selects (M7.2 measured it slower).
template <>
struct neon_binary<ops::Maximum, double> {
    static constexpr bool available = true;
    static void run(const double* a, const double* b, double* o, std::int64_t n) {
        std::int64_t i = 0;
        for (; i + 4 <= n; i += 4) {  // two registers per iteration: more loads in flight
            vst1q_f64(o + i, vmaxq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
            vst1q_f64(o + i + 2, vmaxq_f64(vld1q_f64(a + i + 2), vld1q_f64(b + i + 2)));
        }
        for (; i < n; ++i) o[i] = ops::fmax_nan(a[i], b[i]);
    }
};

template <>
struct neon_binary<ops::Minimum, double> {
    static constexpr bool available = true;
    static void run(const double* a, const double* b, double* o, std::int64_t n) {
        std::int64_t i = 0;
        for (; i + 4 <= n; i += 4) {
            vst1q_f64(o + i, vminq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
            vst1q_f64(o + i + 2, vminq_f64(vld1q_f64(a + i + 2), vld1q_f64(b + i + 2)));
        }
        for (; i < n; ++i) o[i] = ops::fmin_nan(a[i], b[i]);
    }
};

// complex multiply with FCMLA (the FCMA extension): two fused multiply-adds
// per complex number, each covering both lanes.
//   rot 90:  re += -(a.im * b.im)   im += a.im * b.re
//   rot 0:   re +=   a.re * b.re    im += a.re * b.im
// numpy's result is re = fma(ar, br, -(ai*bi)), im = fma(ar, bi, ai*br)
// (scalar_ops.hpp, cmul): the a.im products rounded, the a.re products
// fused. Doing rot 90 first gives exactly that. The accumulator starts at
// -0.0, not +0.0: -0 + x == x for every x, so the first step yields the
// rounded product itself, signed zero included.
template <>
struct neon_binary<ops::Multiply, complex128> {
    static constexpr bool available = true;
    static void run(const complex128* a, const complex128* b, complex128* o, std::int64_t n) {
        const double* pa = reinterpret_cast<const double*>(a);
        const double* pb = reinterpret_cast<const double*>(b);
        double* po = reinterpret_cast<double*>(o);
        const float64x2_t neg_zero = vdupq_n_f64(-0.0);
        for (std::int64_t i = 0; i < n; ++i) {
            const float64x2_t x = vld1q_f64(pa + 2 * i), y = vld1q_f64(pb + 2 * i);
            float64x2_t acc = vcmlaq_rot90_f64(neg_zero, x, y);
            acc = vcmlaq_f64(acc, x, y);
            vst1q_f64(po + 2 * i, acc);
        }
    }
};

}  // namespace sd
