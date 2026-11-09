#include "core/ndarray.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>

namespace sd {

std::int64_t NDArray::size() const {
    std::int64_t n = 1;
    for (int d = 0; d < ndim; ++d) n *= shape[d];
    return n;
}

// An axis of extent 1 is never stepped over, so its stride is irrelevant to
// contiguity; an empty array is trivially contiguous (numpy uses the same
// "relaxed strides" rule).
void NDArray::update_flags() {
    if (size() == 0) {
        c_contiguous = f_contiguous = true;
        return;
    }
    std::int64_t expect = itemsize();
    c_contiguous = true;
    for (int d = ndim - 1; d >= 0; --d) {
        if (shape[d] == 1) continue;
        if (strides[d] != expect) { c_contiguous = false; break; }
        expect *= shape[d];
    }
    expect = itemsize();
    f_contiguous = true;
    for (int d = 0; d < ndim; ++d) {
        if (shape[d] == 1) continue;
        if (strides[d] != expect) { f_contiguous = false; break; }
        expect *= shape[d];
    }
}

std::string shape_str(std::span<const std::int64_t> shape) {
    std::string out = "(";
    for (size_t i = 0; i < shape.size(); ++i) out += (i ? ", " : "") + std::to_string(shape[i]);
    return out + (shape.size() == 1 ? ",)" : ")");
}

std::int64_t checked_size(std::span<const std::int64_t> shape, DType dtype) {
    if (shape.empty())
        throw nb::value_error("0-dimensional arrays are not supported (stardust-numpy has no 0-d arrays)");
    if (shape.size() > static_cast<size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    const std::int64_t limit = std::numeric_limits<std::int64_t>::max() / itemsize(dtype);
    std::int64_t n = 1;
    for (std::int64_t extent : shape) {
        if (extent < 0) throw nb::value_error("negative dimensions are not allowed");
        if (extent != 0 && n > limit / extent) throw nb::value_error("array is too big");
        n *= extent;
    }
    return n;
}

namespace {

// Allocates a contiguous array whose axes are laid out in memory in the order
// `order` (slowest-varying axis first).
NDArray allocate(DType dtype, std::span<const std::int64_t> shape, std::span<const int> order, bool zeroed) {
    const std::int64_t n = checked_size(shape, dtype);
    NDArray a;
    a.dtype = dtype;
    a.ndim = static_cast<int>(shape.size());
    std::copy(shape.begin(), shape.end(), a.shape.begin());
    a.buffer = Buffer::allocate(n * itemsize(dtype), zeroed);
    a.data = a.buffer->data();
    std::int64_t stride = itemsize(dtype);
    for (int k = a.ndim - 1; k >= 0; --k) {
        a.strides[order[k]] = stride;
        stride *= std::max<std::int64_t>(shape[order[k]], 1);
    }
    a.update_flags();
    return a;
}

}  // namespace

NDArray new_array(DType dtype, std::span<const std::int64_t> shape, bool zeroed) {
    std::array<int, kMaxNdim> order;
    std::iota(order.begin(), order.end(), 0);
    return allocate(dtype, shape, {order.data(), shape.size()}, zeroed);
}

NDArray new_array_like(const NDArray& proto, DType dtype, bool zeroed) {
    // Sort axes by decreasing |stride|; ties keep the original (C) order.
    std::array<int, kMaxNdim> order;
    std::iota(order.begin(), order.begin() + proto.ndim, 0);
    std::stable_sort(order.begin(), order.begin() + proto.ndim, [&](int a, int b) {
        return std::llabs(proto.strides[a]) > std::llabs(proto.strides[b]);
    });
    return allocate(dtype, proto.shape_span(), {order.data(), static_cast<size_t>(proto.ndim)}, zeroed);
}

}  // namespace sd
