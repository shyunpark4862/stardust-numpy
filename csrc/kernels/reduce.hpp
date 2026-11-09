// Reduction loops (ARCHITECTURE.md §4.4).
//
// A reduction runs on the M2 iterator with two operands, the input and an
// output that has extent 1 on every reduced axis, broadcast back to the input
// shape (stride 0 on those axes). The kernel then does  out = f(out, x).
// What the iterator hands it decides the strategy:
//   * out stride 0  -> inner-axis reduction: fold a run into one accumulator
//   * out stride != 0 -> outer-axis reduction: add a whole row into the
//     result row, element by element (vectorizes along the kept axis)
// The binary operations are M4's (ops::Add, ops::Maximum, ...), so a sum is
// exactly a chain of add() calls.
//
// M5 keeps these as plain sequential loops; M7 adds multiple accumulators and
// pairwise summation, M8 fixed chunks for threads.
#pragma once

#include "kernels/elementwise.hpp"

#include <cmath>
#include <cstdint>

namespace sd {

// args: x, out (read and written)
template <typename Op, typename T>
void reduce_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    std::byte* o = args[1];
    if (s[1] == 0) {
        T acc = load_as<T>(o);
        for (std::int64_t i = 0; i < n; ++i, x += s[0]) acc = Op::f(acc, load_as<T>(x));
        *reinterpret_cast<T*>(o) = acc;
    } else {
        for (std::int64_t i = 0; i < n; ++i, x += s[0], o += s[1])
            *reinterpret_cast<T*>(o) = Op::f(load_as<T>(o), load_as<T>(x));
    }
}

// Sum of squared deviations, the second pass of var: out += |x - m|^2.
// args: x, m (the mean, broadcast), out (float64)
template <typename T>
void sq_dev_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    const std::byte* m = args[1];
    std::byte* o = args[2];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], m += s[1], o += s[2]) {
        const T d = load_as<T>(x) - load_as<T>(m);
        double sq;
        if constexpr (ops::is_cplx<T>) sq = d.real() * d.real() + d.imag() * d.imag();
        else sq = d * d;
        *reinterpret_cast<double*>(o) += sq;
    }
}

// ---- argmin / argmax over one lane (index order, so ties keep the first) ----

template <typename T, bool Max>
std::int64_t arg_extreme(const std::byte* p, std::int64_t n, std::int64_t stride) {
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
