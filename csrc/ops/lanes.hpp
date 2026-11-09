// Lane-by-lane processing along one axis.
//
// Some operations must see the elements of an axis in *index* order, not in
// the memory order the iterator prefers: argmin/argmax (first tie wins), the
// cumulative scans, sort. They run on lanes: the iterator walks every axis
// *except* `axis` (in memory order, as usual), and each element it visits is
// the start of one lane, which `fn` then processes from index 0 to n-1.
#pragma once

#include "core/iter.hpp"
#include "core/ndarray.hpp"

namespace sd {

struct LaneCtx {
    std::int64_t n, x_stride, o_stride;  // lane length; step along the axis in x and in out
    void (*fn)(const std::byte* x, std::byte* o, const LaneCtx& c);
    void* user = nullptr;  // per-operation state (function pointers, scratch buffers)
};

namespace detail {

inline void lane_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const LaneCtx& c = *static_cast<const LaneCtx*>(ctx);
    for (std::int64_t i = 0; i < n; ++i) c.fn(args[0] + i * s[0], args[1] + i * s[1], c);
}

// a without `axis` (shape and strides), as an iterator operand.
struct Outer {
    Dims shape{}, strides{};
    int ndim = 0;
    static constexpr std::int64_t kOne[1] = {1}, kZero[1] = {0};
    IterOperand op(std::byte* data) const {
        if (ndim == 0) return {data, {kOne, 1}, {kZero, 1}};
        return {data, {shape.data(), static_cast<size_t>(ndim)}, {strides.data(), static_cast<size_t>(ndim)}};
    }
};

inline Outer drop_axis(const NDArray& a, int axis) {
    Outer o;
    for (int d = 0; d < a.ndim; ++d) {
        if (d == axis) continue;
        o.shape[o.ndim] = a.shape[d];
        o.strides[o.ndim] = a.strides[d];
        ++o.ndim;
    }
    return o;
}

}  // namespace detail

// Calls ctx.fn once per lane. `out` has the same ndim as x, with the same
// extents on every axis but `axis` (keepdims, or the axis kept/resized).
inline void run_lanes(const NDArray& x, NDArray& out, int axis, LaneCtx& ctx) {
    const detail::Outer xo = detail::drop_axis(x, axis);
    const detail::Outer oo = detail::drop_axis(out, axis);
    const IterOperand ops[] = {xo.op(x.data), oo.op(out.data)};
    NdIter(ops[0].shape, ops).run(&detail::lane_loop, &ctx);
}

}  // namespace sd
