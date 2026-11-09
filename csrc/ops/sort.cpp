// sort, argsort (API.md §7) and the unique_* family (C8).

#include "ops/sort.hpp"

#include "bind/pyutil.hpp"
#include "kernels/elementwise.hpp"
#include "kernels/simd_sort.hpp"
#include "kernels/sort.hpp"
#include "ops/copy.hpp"
#include "ops/lanes.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <memory>
#include <string>
#include <vector>

namespace sd {

namespace {

// ------------------------------------------------------------ ordering
// NaN goes last in both directions, and with descending=True equal keys still
// keep their original order (the result is never a reversed ascending sort).
// ARCHITECTURE.md §6.2.
//
// M7.9: instead of a comparator that tests for NaN on every comparison, the
// NaNs are first moved to the end (in their original order); the rest is
// then sorted with a plain comparison.

template <typename T>
bool is_nan(const T& v) {
    if constexpr (ops::is_float<T>) return std::isnan(v);
    else if constexpr (ops::is_cplx<T>) return std::isnan(v.real()) || std::isnan(v.imag());
    else return false;
}

// The order of the non-NaN values.
template <typename T>
struct Order {
    bool descending;
    bool operator()(const T& a, const T& b) const {
        if constexpr (ops::is_cplx<T>) {  // unique only: lexicographic (real, imag)
            return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
        } else {
            return descending ? b < a : a < b;
        }
    }
};

template <typename T>
struct Keyed {  // argsort sorts (value, original index) pairs: better locality than sorting indices
    T value;
    std::int64_t index;
};

// A plain heap buffer: std::vector<bool> is a bit-packed specialization with
// no data() pointer, so it cannot back a T* sort.
template <typename T>
std::unique_ptr<T[]> buffer(std::int64_t n) {
    return std::make_unique_for_overwrite<T[]>(static_cast<size_t>(n));
}

// Fills buf[0, n) from a strided lane with make(i, value): values that are
// not NaN first, then the NaNs, each part in its original order. Returns
// how many are not NaN (the part that still needs sorting).
template <typename T, typename E, typename Make>
std::int64_t load_nans_last(const std::byte* x, std::int64_t stride, std::int64_t n, E* buf, Make make) {
    if constexpr (!ops::is_float<T> && !ops::is_cplx<T>) {
        for (std::int64_t i = 0; i < n; ++i) buf[i] = make(i, load_as<T>(x + i * stride));
        return n;
    } else {
        std::int64_t nans = 0;
        for (std::int64_t i = 0; i < n; ++i) nans += is_nan(load_as<T>(x + i * stride));
        std::int64_t front = 0, back = n - nans;
        for (std::int64_t i = 0; i < n; ++i) {
            const T v = load_as<T>(x + i * stride);
            buf[is_nan(v) ? back++ : front++] = make(i, v);
        }
        return n - nans;
    }
}

template <typename T>
void sort_buffer(T* a, std::int64_t n, bool stable, auto less) {
    if (stable) {
        auto tmp = buffer<T>(n / 2 + 1);
        sorting::merge_sort(a, tmp.get(), n, less);
    } else {
        sorting::unstable_sort(a, n, less);
    }
}

// Values alone (sort, unique_values). The NEON sort reorders equal values,
// so it is used only when that cannot be seen (M7.9c decision): for int64
// equal values are identical, and for float64 too -- NaNs are already split
// off -- except that -0.0 == +0.0. So a stable float sort takes it unless the
// data holds zeros of both signs. stable=False takes it always.
inline constexpr std::int64_t kSimdMin = 32;  // below this, insertion/merge sort is as fast

template <typename T>
bool zeros_of_both_signs(const T* a, std::int64_t n) {
    bool neg = false, pos = false;
    for (std::int64_t i = 0; i < n; ++i)
        if (a[i] == 0) (std::signbit(a[i]) ? neg : pos) = true;
    return neg && pos;
}

template <typename T>
void sort_values(T* a, std::int64_t n, bool stable, bool descending) {
    if constexpr (ops::is_float<T> || ops::is_int<T>) {
        bool simd = n >= kSimdMin;
        if constexpr (ops::is_float<T>) simd = simd && (!stable || !zeros_of_both_signs(a, n));
        if (simd) {
            simd_sort::sort<std::conditional_t<ops::is_float<T>, simd_sort::F64, simd_sort::I64>>(a, n);
            if (descending) std::reverse(a, a + n);  // ties are indistinguishable here
            return;
        }
    }
    sort_buffer(a, n, stable, Order<T>{descending});
}

// ------------------------------------------------------------ sort lanes

struct SortCfg {
    bool descending, stable;
};

template <typename T>
void sort_lane(const std::byte* x, std::byte* o, const LaneCtx& c) {
    const SortCfg& cfg = *static_cast<const SortCfg*>(c.user);
    auto buf = buffer<T>(c.n);
    const std::int64_t m = load_nans_last<T>(x, c.x_stride, c.n, buf.get(), [](std::int64_t, T v) { return v; });
    sort_values(buf.get(), m, cfg.stable, cfg.descending);
    for (std::int64_t i = 0; i < c.n; ++i) *reinterpret_cast<T*>(o + i * c.o_stride) = buf[i];
}

template <typename T>
void argsort_lane(const std::byte* x, std::byte* o, const LaneCtx& c) {
    const SortCfg& cfg = *static_cast<const SortCfg*>(c.user);
    auto buf = buffer<Keyed<T>>(c.n);
    const std::int64_t m = load_nans_last<T>(x, c.x_stride, c.n, buf.get(),
                                             [](std::int64_t i, T v) { return Keyed<T>{v, i}; });
    const Order<T> order{cfg.descending};
    sort_buffer(buf.get(), m, cfg.stable,
                [&](const Keyed<T>& a, const Keyed<T>& b) { return order(a.value, b.value); });
    for (std::int64_t i = 0; i < c.n; ++i) *reinterpret_cast<std::int64_t*>(o + i * c.o_stride) = buf[i].index;
}

template <bool Arg>
nb::object sort_impl(nb::handle x_obj, nb::handle axis_obj, bool descending, bool stable) {
    const char* name = Arg ? "argsort" : "sort";
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128)
        throw nb::type_error((std::string(name) + "() is not supported for complex128 (no ordering)").c_str());  // S5
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), x.ndim);
    NDArray out = new_array_like(x, Arg ? DType::Int64 : x.dtype);
    SortCfg cfg{descending, stable};
    LaneCtx ctx{x.shape[axis], x.strides[axis], out.strides[axis], nullptr, &cfg};
    visit_dtype(x.dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (!ops::is_cplx<T>) ctx.fn = Arg ? &argsort_lane<T> : &sort_lane<T>;
    });
    if (ctx.n > 0) run_lanes(x, out, axis, ctx);
    return wrap(std::move(out));
}

// ------------------------------------------------------------------ unique
// flatten -> stable argsort -> compare neighbours (ARCHITECTURE.md §6.3).
// Stable, so within a group of equal values the first one is the first
// occurrence in x: that gives `indices`, and the value that is kept (-0.0 or
// +0.0, whichever came first). NaN never equals anything, so each NaN is its
// own group (N3).

template <typename T>
bool same(const T& a, const T& b) {
    return a == b;  // false for NaN (and for complex with a NaN part)
}

struct UniqueParts {
    NDArray values, indices, inverse, counts;
};

UniqueParts unique_parts(const NDArray& x) {
    NDArray flat = x;  // C order, contiguous
    if (!x.c_contiguous) {
        flat = new_array(x.dtype, x.shape_span());
        assign(flat, x);
    }
    const std::int64_t n = x.size();
    UniqueParts r;
    std::vector<std::int64_t> starts;  // positions in sorted order where a new group begins
    std::vector<std::int64_t> order(n);
    visit_dtype(x.dtype, [&]<typename T>(std::type_identity<T>) {
        auto buf = buffer<Keyed<T>>(n);
        const std::int64_t m = load_nans_last<T>(flat.data, flat.itemsize(), n, buf.get(),
                                                 [](std::int64_t i, T v) { return Keyed<T>{v, i}; });
        const Order<T> ascending{false};
        sort_buffer(buf.get(), m, /*stable=*/true,
                    [&](const Keyed<T>& a, const Keyed<T>& b) { return ascending(a.value, b.value); });
        for (std::int64_t i = 0; i < n; ++i) {
            order[i] = buf[i].index;
            if (i == 0 || !same(buf[i - 1].value, buf[i].value)) starts.push_back(i);
        }
        const std::int64_t g[] = {static_cast<std::int64_t>(starts.size())};
        r.values = new_array(x.dtype, g);
        for (size_t k = 0; k < starts.size(); ++k) reinterpret_cast<T*>(r.values.data)[k] = buf[starts[k]].value;
    });
    const std::int64_t groups = static_cast<std::int64_t>(starts.size());
    const std::int64_t g[] = {groups};
    r.indices = new_array(DType::Int64, g);
    r.counts = new_array(DType::Int64, g);
    r.inverse = new_array(DType::Int64, x.shape_span());  // same shape as x (Array API)
    auto* idx = reinterpret_cast<std::int64_t*>(r.indices.data);
    auto* cnt = reinterpret_cast<std::int64_t*>(r.counts.data);
    auto* inv = reinterpret_cast<std::int64_t*>(r.inverse.data);  // C order, like `flat`
    for (std::int64_t k = 0; k < groups; ++k) {
        const std::int64_t begin = starts[k], end = k + 1 < groups ? starts[k + 1] : n;
        idx[k] = order[begin];
        cnt[k] = end - begin;
        for (std::int64_t i = begin; i < end; ++i) inv[order[i]] = k;
    }
    return r;
}

nb::object result_type(const char* name) { return nb::module_::import_("stardust_numpy._results").attr(name); }

// unique_values needs neither positions nor counts, so it sorts the values
// alone: half the data to move, and no (value, index) pairs (M7.9b). Still a
// stable sort, so of -0.0 and +0.0 the first occurrence is kept, as in
// unique_parts.
nb::object unique_values(nb::handle x_obj) {
    const NDArray& x = require_ndarray(x_obj);
    const std::int64_t n = x.size();
    NDArray flat = x;
    if (!x.c_contiguous) {
        flat = new_array(x.dtype, x.shape_span());
        assign(flat, x);
    }
    return visit_dtype(x.dtype, [&]<typename T>(std::type_identity<T>) {
        auto buf = buffer<T>(n);
        const std::int64_t m =
            load_nans_last<T>(flat.data, flat.itemsize(), n, buf.get(), [](std::int64_t, T v) { return v; });
        sort_values(buf.get(), m, /*stable=*/true, /*descending=*/false);
        std::int64_t groups = 0;  // compact in place: keep the first of every run of equal values
        for (std::int64_t i = 0; i < n; ++i)
            if (i == 0 || !same(buf[groups - 1], buf[i])) buf[groups++] = buf[i];
        const std::int64_t g[] = {groups};
        NDArray out = new_array(x.dtype, g);
        std::copy(buf.get(), buf.get() + groups, reinterpret_cast<T*>(out.data));
        return wrap(std::move(out));
    });
}

nb::object unique_counts(nb::handle x) {
    UniqueParts p = unique_parts(require_ndarray(x));
    return result_type("UniqueCountsResult")(wrap(std::move(p.values)), wrap(std::move(p.counts)));
}

nb::object unique_inverse(nb::handle x) {
    UniqueParts p = unique_parts(require_ndarray(x));
    return result_type("UniqueInverseResult")(wrap(std::move(p.values)), wrap(std::move(p.inverse)));
}

nb::object unique_all(nb::handle x) {
    UniqueParts p = unique_parts(require_ndarray(x));
    return result_type("UniqueAllResult")(wrap(std::move(p.values)), wrap(std::move(p.indices)),
                                          wrap(std::move(p.inverse)), wrap(std::move(p.counts)));
}

}  // namespace

void bind_sort(nb::module_& m) {
    using nb::arg;
    m.def("sort", &sort_impl<false>, arg(), nb::kw_only(), arg("axis") = -1, arg("descending") = false,
          arg("stable") = true);
    m.def("argsort", &sort_impl<true>, arg(), nb::kw_only(), arg("axis") = -1, arg("descending") = false,
          arg("stable") = true);
    m.def("unique_values", &unique_values, arg());
    m.def("unique_counts", &unique_counts, arg());
    m.def("unique_inverse", &unique_inverse, arg());
    m.def("unique_all", &unique_all, arg());
}

}  // namespace sd
