// Inner loops for the operations in scalar_ops.hpp, and the per-operation
// loop tables (ARCHITECTURE.md §4.1: table[op][dtype]).
//
// One template per arity turns Op::f into a 1-D strided loop with the M2
// kernel signature. Like the cast loops these are plain scalar loops for now:
// M7 adds contiguous NEON paths inside the same signatures and measures them
// against this baseline.
#pragma once

#include "core/iter.hpp"
#include "kernels/scalar_ops.hpp"

#include <array>
#include <type_traits>

namespace sd {

template <typename T>
inline T load_as(const std::byte* p) { return *reinterpret_cast<const T*>(p); }

template <typename Op, typename T>
using unary_result_t = decltype(Op::f(std::declval<T>()));
template <typename Op, typename T>
using binary_result_t = decltype(Op::f(std::declval<T>(), std::declval<T>()));

// args: x, out
template <typename Op, typename T>
void unary_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    using R = unary_result_t<Op, T>;
    const std::byte* x = args[0];
    std::byte* o = args[1];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], o += s[1])
        *reinterpret_cast<R*>(o) = Op::f(load_as<T>(x));
}

// args: a, b, out
template <typename Op, typename T>
void binary_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    using R = binary_result_t<Op, T>;
    const std::byte* a = args[0];
    const std::byte* b = args[1];
    std::byte* o = args[2];
    for (std::int64_t i = 0; i < n; ++i, a += s[0], b += s[1], o += s[2])
        *reinterpret_cast<R*>(o) = Op::f(load_as<T>(a), load_as<T>(b));
}

// args: condition (bool), x1, x2, out
template <typename T>
void where_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* c = args[0];
    const std::byte* a = args[1];
    const std::byte* b = args[2];
    std::byte* o = args[3];
    for (std::int64_t i = 0; i < n; ++i, c += s[0], a += s[1], b += s[2], o += s[3])
        *reinterpret_cast<T*>(o) = load_as<bool>(c) ? load_as<T>(a) : load_as<T>(b);
}

// clip = minimum(maximum(x, lo), hi), fused into one pass, with the same NaN
// and signed-zero rules as maximum/minimum. args: x, lo, hi, out
template <typename T>
void clip_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    const std::byte* lo = args[1];
    const std::byte* hi = args[2];
    std::byte* o = args[3];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], lo += s[1], hi += s[2], o += s[3])
        *reinterpret_cast<T*>(o) =
            ops::Minimum::f(ops::Maximum::f(load_as<T>(x), load_as<T>(lo)), load_as<T>(hi));
}

// One table row: the loop for each computation dtype (null where Op has
// none), and the dtype that loop writes.
struct LoopEntry {
    Loop loop = nullptr;
    DType out = DType::Bool;
};
using LoopTable = std::array<LoopEntry, kNumDTypes>;

namespace detail {

template <typename Op, typename T, int Arity>
constexpr LoopEntry entry() {
    if constexpr (!Op::template ok<T>) {
        return {};
    } else if constexpr (Arity == 1) {
        return {&unary_loop<Op, T>, dtype_of<unary_result_t<Op, T>>};
    } else {
        return {&binary_loop<Op, T>, dtype_of<binary_result_t<Op, T>>};
    }
}

template <typename Op, int Arity>
constexpr LoopTable make_table() {
    return {entry<Op, bool, Arity>(), entry<Op, std::int64_t, Arity>(), entry<Op, double, Arity>(),
            entry<Op, complex128, Arity>()};
}

}  // namespace detail

template <typename Op> inline constexpr LoopTable unary_table = detail::make_table<Op, 1>();
template <typename Op> inline constexpr LoopTable binary_table = detail::make_table<Op, 2>();

}  // namespace sd
