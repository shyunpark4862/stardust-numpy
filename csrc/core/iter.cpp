#include "core/iter.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace sd {

int broadcast_into(std::span<const std::span<const std::int64_t>> shapes, Dims& out) {
    size_t ndim = 0;
    for (auto s : shapes) ndim = std::max(ndim, s.size());
    if (ndim > static_cast<size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    std::fill(out.begin(), out.begin() + ndim, 1);
    for (auto s : shapes) {
        const size_t offset = ndim - s.size();
        for (size_t d = 0; d < s.size(); ++d) {
            std::int64_t& o = out[offset + d];
            if (s[d] == o || s[d] == 1) continue;
            if (o == 1) { o = s[d]; continue; }
            std::string msg = "shapes could not be broadcast together:";
            for (auto t : shapes) msg += " " + shape_str(t);
            throw nb::value_error(msg.c_str());
        }
    }
    return static_cast<int>(ndim);
}

std::vector<std::int64_t> broadcast_shapes(std::span<const std::span<const std::int64_t>> shapes) {
    Dims out;
    const int ndim = broadcast_into(shapes, out);
    return {out.begin(), out.begin() + ndim};
}

NdIter::NdIter(std::span<const std::int64_t> shape, std::span<const IterOperand> ops, IterMode mode) {
    nop_ = static_cast<int>(ops.size());
    const int ndim = static_cast<int>(shape.size());

    // 1. Broadcast: right-align each operand; missing or extent-1 axes read the
    //    same element over and over, i.e. stride 0.
    for (int k = 0; k < nop_; ++k) {
        data_[k] = ops[k].data;
        const int offset = ndim - static_cast<int>(ops[k].shape.size());
        if (offset < 0) throw nb::value_error("operand has more dimensions than the iteration shape");
        for (int d = 0; d < ndim; ++d) {
            const int od = d - offset;
            if (od < 0 || ops[k].shape[od] == 1) {
                strides_[k][d] = 0;
            } else if (ops[k].shape[od] == shape[d]) {
                strides_[k][d] = ops[k].strides[od];
            } else {
                throw nb::value_error(("operand of shape " + shape_str(ops[k].shape) +
                                       " cannot be broadcast to " + shape_str(shape)).c_str());
            }
        }
    }
    std::copy(shape.begin(), shape.end(), shape_.begin());
    ndim_ = ndim;

    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        empty_ = true;
        return;
    }

    if (mode != IterMode::Naive) {
        // 2. Drop extent-1 axes.
        int n = 0;
        for (int d = 0; d < ndim_; ++d) {
            if (shape_[d] == 1) continue;
            shape_[n] = shape_[d];
            for (int k = 0; k < nop_; ++k) strides_[k][n] = strides_[k][d];
            ++n;
        }
        ndim_ = n;

        // 3. Reorder: insertion sort, outermost = largest |stride|. The last
        //    operand (the output, by convention) decides first, so writes are
        //    sequential; a stride of 0 (broadcast) carries no information and
        //    is skipped. Equal strides keep the original order, so a
        //    C-contiguous array is untouched.
        std::array<int, kMaxNdim> perm;
        for (int d = 0; d < ndim_; ++d) perm[d] = d;
        auto outer = [&](int a, int b) {
            for (int k = nop_ - 1; k >= 0; --k) {
                const std::int64_t sa = std::llabs(strides_[k][a]), sb = std::llabs(strides_[k][b]);
                if (sa == 0 || sb == 0) continue;
                if (sa != sb) return sa > sb;
            }
            return false;
        };
        for (int i = 1; i < ndim_; ++i)
            for (int j = i; j > 0 && outer(perm[j], perm[j - 1]); --j) std::swap(perm[j], perm[j - 1]);
        Dims shape_p;  // copies of the live part only
        std::array<Dims, kMaxOperands> strides_p;
        std::copy(shape_.begin(), shape_.begin() + ndim_, shape_p.begin());
        for (int k = 0; k < nop_; ++k) std::copy(strides_[k].begin(), strides_[k].begin() + ndim_, strides_p[k].begin());
        for (int d = 0; d < ndim_; ++d) {
            shape_[d] = shape_p[perm[d]];
            for (int k = 0; k < nop_; ++k) strides_[k][d] = strides_p[k][perm[d]];
        }

        // 4. Coalesce: axis d (outer) and d+1 (inner) are one run when, for
        //    every operand, stepping once along d equals stepping extent(d+1)
        //    times along d+1.
        if (mode == IterMode::Full && ndim_ > 1) {
            int out = 0;
            for (int d = 1; d < ndim_; ++d) {
                bool mergeable = true;
                for (int k = 0; k < nop_; ++k)
                    mergeable = mergeable && strides_[k][out] == strides_[k][d] * shape_[d];
                if (mergeable) {
                    shape_[out] *= shape_[d];
                    for (int k = 0; k < nop_; ++k) strides_[k][out] = strides_[k][d];
                } else {
                    ++out;
                    shape_[out] = shape_[d];
                    for (int k = 0; k < nop_; ++k) strides_[k][out] = strides_[k][d];
                }
            }
            ndim_ = out + 1;
        }

        if (ndim_ == 0) {  // every axis had extent 1: a single element
            ndim_ = 1;
            shape_[0] = 1;
            for (int k = 0; k < nop_; ++k) strides_[k][0] = 0;
        }
    }
}

void NdIter::run(Loop kernel, void* ctx) const {
    if (empty_) return;
    const int last = ndim_ - 1;
    std::array<std::byte*, kMaxOperands> p = data_;
    std::array<std::int64_t, kMaxOperands> inner{};
    for (int k = 0; k < nop_; ++k) inner[k] = strides_[k][last];
    Dims index;
    std::fill(index.begin(), index.begin() + ndim_, 0);
    for (;;) {
        kernel(p.data(), shape_[last], inner.data(), ctx);
        // Odometer over the outer axes.
        int d = last - 1;
        for (; d >= 0; --d) {
            for (int k = 0; k < nop_; ++k) p[k] += strides_[k][d];
            if (++index[d] < shape_[d]) break;
            for (int k = 0; k < nop_; ++k) p[k] -= strides_[k][d] * shape_[d];
            index[d] = 0;
        }
        if (d < 0) return;
    }
}

}  // namespace sd
