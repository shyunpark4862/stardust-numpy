// Value conversions between dtypes and between C++ and Python scalars.
//
// One rule set serves astype, `dtype=` arguments and buffer import
// (DESIGN.md §2.4, "explicit conversion"):
//   * anything -> bool       : x != 0   (nan -> true)
//   * float    -> int64      : truncate toward zero, saturate out-of-range, nan -> 0
//   * complex  -> float/int  : TypeError (would drop the imaginary part)
#pragma once

#include "core/dtype.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>

#include <cmath>
#include <cstddef>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace sd {

namespace nb = nanobind;

template <typename T> inline constexpr bool is_complex_v = false;
template <typename T> inline constexpr bool is_complex_v<std::complex<T>> = true;

// Dtype-level check; value-level conversion below never fails.
void check_castable(DType from, DType to);

// float -> int64 by S3: truncate toward zero, saturate out-of-range values,
// nan -> 0. That is exactly what the ARM `FCVTZS` instruction does, so it is
// called directly. (A plain C++ cast is undefined behaviour for nan, inf and
// out-of-range values; the three branches that used to spell the rule out
// kept the conversion from vectorizing: 1.8x slower than numpy, measured
// after M10.) The compiler turns a loop of these into the vector FCVTZS.
inline std::int64_t saturate_to_int64(double x) { return vcvtd_s64_f64(x); }

// Converts one value of any source element type (including the narrower
// types met in buffer import, e.g. int16 or float32) to storage type To.
template <typename To, typename From>
To cast_value(From v) {
    if constexpr (std::is_same_v<To, bool>) {
        if constexpr (is_complex_v<From>) return v.real() != 0 || v.imag() != 0;
        else return v != From(0);
    } else if constexpr (is_complex_v<To>) {
        if constexpr (is_complex_v<From>) return To(static_cast<double>(v.real()), static_cast<double>(v.imag()));
        else return To(static_cast<double>(v), 0.0);
    } else if constexpr (is_complex_v<From>) {
        // Instantiated by generic dtype x dtype loops but never reached:
        // check_castable rejects complex -> real before any element is touched.
        check_castable(DType::Complex128, dtype_of<To>);
        __builtin_unreachable();
    } else if constexpr (std::is_same_v<To, std::int64_t> && std::is_floating_point_v<From>) {
        return saturate_to_int64(static_cast<double>(v));
    } else {
        return static_cast<To>(v);
    }
}

// ---- Python scalars ----

// The dtype a Python scalar maps to (bool, int, float, complex and their
// subclasses), or nullopt if `o` is not one of those.
std::optional<DType> scalar_dtype(nb::handle o);

// Converts Python scalar `o` (whose dtype is `src`, from scalar_dtype) to T.
// Raises OverflowError for an int that does not fit in int64 (Z4).
template <typename T> T scalar_from_python(nb::handle o, DType src);

// Builds the Python scalar for a stored value (Z1: no 0-d arrays, ever).
nb::object to_python(bool v);
nb::object to_python(std::int64_t v);
nb::object to_python(double v);
nb::object to_python(complex128 v);

// Reads the element of dtype `d` stored at `p` (any alignment) as a Python scalar.
nb::object load_python(DType d, const std::byte* p);

}  // namespace sd
