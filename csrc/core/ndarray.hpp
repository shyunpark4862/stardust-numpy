// The n-dimensional array object (ARCHITECTURE.md §3.3).
//
// An array is a *description* of how to read a Buffer: start at `data`, and
// element (i0, i1, ..., ik) lives at  data + i0*strides[0] + ... + ik*strides[k]
// (strides in bytes). Transposes, slices and broadcasts only change this
// description, never the memory -- that is what makes them free.
#pragma once

#include "core/buffer.hpp"
#include "core/dtype.hpp"

#include <nanobind/nanobind.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace sd {

namespace nb = nanobind;

inline constexpr int kMaxNdim = 32;

// Stored inline (no heap allocation per array). Only the first ndim entries
// are meaningful.
using Dims = std::array<std::int64_t, kMaxNdim>;

struct NDArray {
    std::shared_ptr<Buffer> buffer;
    std::byte* data = nullptr;
    DType dtype = DType::Float64;
    int ndim = 0;
    Dims shape{};
    Dims strides{};
    bool c_contiguous = true;
    bool f_contiguous = true;
    bool writeable = true;
    // What Python sees as `.base`: the object that owns the memory, or a
    // null handle when this array allocated `buffer` itself.
    nb::object base;

    std::int64_t itemsize() const { return sd::itemsize(dtype); }
    std::int64_t size() const;
    std::int64_t nbytes() const { return size() * itemsize(); }
    std::span<const std::int64_t> shape_span() const { return {shape.data(), static_cast<size_t>(ndim)}; }
    std::span<const std::int64_t> strides_span() const { return {strides.data(), static_cast<size_t>(ndim)}; }

    // Recomputes c_contiguous / f_contiguous from shape and strides.
    void update_flags();
};

// "(2, 3)", "(4,)": shapes in error messages, spelled like Python tuples.
std::string shape_str(std::span<const std::int64_t> shape);

// Validates a requested shape (0 < ndim <= kMaxNdim, no negative extents, no
// overflow of the byte size) and returns the element count.
std::int64_t checked_size(std::span<const std::int64_t> shape, DType dtype);

// New C-contiguous (row-major) array.
NDArray new_array(DType dtype, std::span<const std::int64_t> shape, bool zeroed = false);

// New contiguous array whose memory axis order follows `proto`'s strides
// (numpy's order='K'): a Fortran-ordered prototype gives a Fortran-ordered
// result. Element order of *values* is unaffected; only the layout is.
NDArray new_array_like(const NDArray& proto, DType dtype, bool zeroed = false);

// Visits every element in row-major index order, with N arrays of the same
// shape walked in lockstep: f(p0, p1, ...) receives the element pointers.
//
// This is the "odometer": the last axis spins fastest; when it wraps, the
// next axis advances by one, exactly like the digits of a car's odometer.
// M2 replaces most uses with a smarter iterator; this simple form is enough
// for conversions and tolist.
template <std::size_t N, typename F>
void for_each_element(std::span<const std::int64_t> shape,
                      std::array<std::byte*, N> ptrs,
                      std::array<const std::int64_t*, N> strides, F&& f) {
    const int ndim = static_cast<int>(shape.size());
    for (std::int64_t extent : shape)
        if (extent == 0) return;
    const int last = ndim - 1;
    std::array<std::int64_t, kMaxNdim> index{};
    for (;;) {
        std::array<std::byte*, N> p = ptrs;
        for (std::int64_t i = 0; i < shape[last]; ++i) {
            [&]<std::size_t... I>(std::index_sequence<I...>) { f(p[I]...); }(std::make_index_sequence<N>{});
            for (std::size_t k = 0; k < N; ++k) p[k] += strides[k][last];
        }
        int d = last - 1;
        for (; d >= 0; --d) {
            for (std::size_t k = 0; k < N; ++k) ptrs[k] += strides[k][d];
            if (++index[d] < shape[d]) break;
            for (std::size_t k = 0; k < N; ++k) ptrs[k] -= strides[k][d] * shape[d];
            index[d] = 0;
        }
        if (d < 0) return;
    }
}

}  // namespace sd
