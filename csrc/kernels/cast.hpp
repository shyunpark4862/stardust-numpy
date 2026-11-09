// Cast kernels: one inner loop per (from, to) dtype pair, 4 x 4 = 16 loops,
// generated from a single template and looked up in a table at runtime.
//
// M2 keeps these as plain strided scalar loops on purpose: they are the
// baseline that M7's contiguous/NEON fast paths are measured against.
#pragma once

#include "core/convert.hpp"
#include "core/iter.hpp"

#include <array>

namespace sd {

// args[0] = source, args[1] = destination.
template <typename From, typename To>
void cast_loop(std::byte* const* args, std::int64_t n, const std::int64_t* strides, void*) {
    const std::byte* src = args[0];
    std::byte* dst = args[1];
    for (std::int64_t i = 0; i < n; ++i, src += strides[0], dst += strides[1])
        *reinterpret_cast<To*>(dst) = cast_value<To>(*reinterpret_cast<const From*>(src));
}

namespace detail {
template <typename From>
inline constexpr std::array<Loop, kNumDTypes> cast_row = {
    cast_loop<From, bool>, cast_loop<From, std::int64_t>, cast_loop<From, double>, cast_loop<From, complex128>};
inline constexpr std::array<std::array<Loop, kNumDTypes>, kNumDTypes> cast_table = {
    cast_row<bool>, cast_row<std::int64_t>, cast_row<double>, cast_row<complex128>};
}  // namespace detail

// complex -> real entries exist in the table but are unreachable:
// check_castable runs before any kernel is chosen.
inline Loop cast_kernel(DType from, DType to) {
    return detail::cast_table[static_cast<int>(from)][static_cast<int>(to)];
}

}  // namespace sd
