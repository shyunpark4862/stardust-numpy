// The four element types and their promotion order (DESIGN.md §1.3).
#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

namespace sd {

// Ordered by "kind": promotion of two dtypes is simply the larger one.
enum class DType : std::uint8_t { Bool, Int64, Float64, Complex128 };

inline constexpr int kNumDTypes = 4;

using complex128 = std::complex<double>;

constexpr DType promote(DType a, DType b) { return a > b ? a : b; }

constexpr std::int64_t itemsize(DType d) {
    switch (d) {
        case DType::Bool: return 1;
        case DType::Int64: return 8;
        case DType::Float64: return 8;
        case DType::Complex128: return 16;
    }
    __builtin_unreachable();
}

constexpr const char* dtype_name(DType d) {
    switch (d) {
        case DType::Bool: return "bool";
        case DType::Int64: return "int64";
        case DType::Float64: return "float64";
        case DType::Complex128: return "complex128";
    }
    __builtin_unreachable();
}

// C++ storage type of each dtype.
template <typename T> inline constexpr DType dtype_of = [] {
    if constexpr (std::is_same_v<T, bool>) return DType::Bool;
    else if constexpr (std::is_same_v<T, std::int64_t>) return DType::Int64;
    else if constexpr (std::is_same_v<T, double>) return DType::Float64;
    else if constexpr (std::is_same_v<T, complex128>) return DType::Complex128;
    else static_assert(!sizeof(T), "not a stardust storage type");
}();

// Runtime dtype -> compile-time type. Calls f(std::type_identity<T>{}) where T
// is the storage type of `d`; this is how one templated function body is
// instantiated once per dtype and selected at runtime.
template <typename F>
decltype(auto) visit_dtype(DType d, F&& f) {
    switch (d) {
        case DType::Bool: return f(std::type_identity<bool>{});
        case DType::Int64: return f(std::type_identity<std::int64_t>{});
        case DType::Float64: return f(std::type_identity<double>{});
        case DType::Complex128: return f(std::type_identity<complex128>{});
    }
    __builtin_unreachable();
}

}  // namespace sd
