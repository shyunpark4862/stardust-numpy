// sort, argsort (API.md §7) and the unique_* family (C8).

#include "ops/sort.hpp"

#include "bind/pyutil.hpp"
#include "kernels/elementwise.hpp"
#include "kernels/simd_sort.hpp"
#include "kernels/sort.hpp"
#include "ops/copy.hpp"
#include "ops/lanes.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
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

// A lane's working buffer: short lanes use per-thread scratch, so sorting
// many short lanes (sort(x, axis=1) of a (100000, 8) array) does not
// allocate twice per lane. Slot 0 holds the lane, slot 1 the merge sort's
// half copy. Longer lanes allocate, so a thread keeps at most 2 x 64 KB
// per element type.
inline constexpr std::int64_t kScratchBytes = 64 * 1024;

template <typename T>
class LaneBuffer {
public:
    LaneBuffer(std::int64_t n, int slot) {
        if (n * static_cast<std::int64_t>(sizeof(T)) <= kScratchBytes) {
            thread_local std::unique_ptr<std::byte[]> mem =
                std::make_unique_for_overwrite<std::byte[]>(static_cast<size_t>(2 * kScratchBytes));
            ptr_ = reinterpret_cast<T*>(mem.get() + slot * kScratchBytes);
        } else {
            heap_ = buffer<T>(n);
            ptr_ = heap_.get();
        }
    }
    T* get() const { return ptr_; }
    T& operator[](std::int64_t i) const { return ptr_[i]; }

private:
    std::unique_ptr<T[]> heap_;
    T* ptr_;
};

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

// ---- int64: LSD radix sort (Tier 3 item 12, user decision 2026-11-10) ----
// Picked by dtype and lane length only, never by looking at the data. Stable,
// so it serves sort and argsort alike. The key maps signed order to unsigned
// order (sign bit flipped); for descending it is inverted, so larger values
// come first and equal ones keep their order -- what the comparator gives.
//
// Keys are counted relative to the smallest one, so only the 8-bit digits
// of (max - min) are sorted: a lane of values in [-1000, 1000) takes two
// passes, not eight. (Without that, mixed signs made every digit vary --
// 0x7F/0x80 on top, 0xFF/0x00 below -- and the narrow range was the slower
// one, measured.) One pass counts every digit; a digit where all keys have
// the same byte moves nothing and is skipped.

// Below these lengths the fixed cost (clearing and summing up to 8 x 256
// counts per lane) loses. Measured on 2^20 int64 in lanes of n, one thread:
// values against the NEON sort, n = 512: 6.4 ms radix vs 5.9 (wide range),
// n = 1024: 6.0 vs 6.8; pairs against merge sort, n = 64: 15.2 vs 13.2,
// n = 128: 10.6 vs 16.0.
inline constexpr std::int64_t kRadixMinValues = 1024;
inline constexpr std::int64_t kRadixMinPairs = 128;

inline std::uint64_t radix_key(std::int64_t v, bool descending) {
    const std::uint64_t u = static_cast<std::uint64_t>(v) ^ (std::uint64_t{1} << 63);
    return descending ? ~u : u;
}

// A long lane is cut into pieces of kSortPiece elements for the pool (Tier
// 3 item 13, below); only when the pool can run them, unless the pieces
// show in the result.
inline constexpr std::int64_t kSortPiece = 1 << 16;

inline bool use_pieces(std::int64_t n, bool pieces_show) {
    return n >= 2 * kSortPiece && (pieces_show || parallel::available());
}

// The radix sort on the pool: every pass counts its digit piece by piece,
// turns the counts into each piece's first slot per byte value (byte value
// first, then piece, in order) and scatters the pieces in parallel. The
// result is the stable order, whatever the pieces.
template <typename E, typename Key>
void radix_sort_parallel(E* a, std::int64_t n, Key key) {
    const std::int64_t pieces = (n + kSortPiece - 1) / kSortPiece;
    auto range = [&](std::int64_t t) { return std::pair{t * kSortPiece, std::min(n, (t + 1) * kSortPiece)}; };
    std::vector<std::uint64_t> los(pieces), his(pieces);
    parallel::parallel_for(pieces, [&](std::int64_t t) {
        std::uint64_t lo = ~std::uint64_t{0}, hi = 0;
        for (auto [i, end] = range(t); i < end; ++i) {
            const std::uint64_t k = key(a[i]);
            lo = std::min(lo, k);
            hi = std::max(hi, k);
        }
        los[t] = lo;
        his[t] = hi;
    });
    const std::uint64_t lo = *std::min_element(los.begin(), los.end());
    const std::uint64_t hi = *std::max_element(his.begin(), his.end());
    int digits = 0;
    while (digits < 8 && ((hi - lo) >> (8 * digits)) != 0) ++digits;
    auto tmp = buffer<E>(n);
    E* src = a;
    E* dst = tmp.get();
    std::vector<std::array<std::int64_t, 256>> slot(pieces);
    for (int d = 0; d < digits; ++d) {
        auto digit = [&](const E& e) { return ((key(e) - lo) >> (8 * d)) & 255; };
        parallel::parallel_for(pieces, [&](std::int64_t t) {
            slot[t].fill(0);
            for (auto [i, end] = range(t); i < end; ++i) ++slot[t][digit(src[i])];
        });
        std::int64_t sum = 0;
        bool moves = true;
        for (int b = 0; b < 256; ++b) {
            std::int64_t total = 0;
            for (std::int64_t t = 0; t < pieces; ++t) total += slot[t][b];
            if (total == n) moves = false;  // one byte value for all keys
            for (std::int64_t t = 0; t < pieces; ++t) sum += std::exchange(slot[t][b], sum);
        }
        if (!moves) continue;
        parallel::parallel_for(pieces, [&](std::int64_t t) {
            std::array<std::int64_t, 256>& c = slot[t];
            for (auto [i, end] = range(t); i < end; ++i) dst[c[digit(src[i])]++] = src[i];
        });
        std::swap(src, dst);
    }
    if (src != a) std::copy(src, src + n, a);
}

template <typename E, typename Key>
void radix_sort(E* a, std::int64_t n, Key key) {
    if (use_pieces(n, false)) return radix_sort_parallel(a, n, key);
    std::uint64_t lo = ~std::uint64_t{0}, hi = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::uint64_t k = key(a[i]);
        lo = std::min(lo, k);
        hi = std::max(hi, k);
    }
    int digits = 0;
    while (digits < 8 && ((hi - lo) >> (8 * digits)) != 0) ++digits;
    if (digits == 0) return;  // every key equal: a stable sort leaves the lane as it is
    std::array<std::array<std::int64_t, 256>, 8> count;
    for (int d = 0; d < digits; ++d) count[d].fill(0);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::uint64_t k = key(a[i]) - lo;
        for (int d = 0; d < digits; ++d) ++count[d][(k >> (8 * d)) & 255];
    }
    const LaneBuffer<E> tmp(n, 1);
    E* src = a;
    E* dst = tmp.get();
    for (int d = 0; d < digits; ++d) {
        std::array<std::int64_t, 256>& c = count[d];
        auto digit = [&](const E& e) { return ((key(e) - lo) >> (8 * d)) & 255; };
        if (c[digit(a[0])] == n) continue;  // one byte value for all keys
        std::int64_t sum = 0;
        for (std::int64_t& v : c) sum += std::exchange(v, sum);  // counts -> first slots
        for (std::int64_t i = 0; i < n; ++i) dst[c[digit(src[i])]++] = src[i];
        std::swap(src, dst);
    }
    if (src != a) std::copy(src, src + n, a);
}

template <typename T>
void sort_buffer(T* a, std::int64_t n, bool stable, auto less) {
    if (stable) {
        const LaneBuffer<T> tmp(n / 2 + 1, 1);
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
void sort_values_serial(T* a, std::int64_t n, bool stable, bool descending) {
    if constexpr (ops::is_int<T>) {
        if (n >= kRadixMinValues) return radix_sort(a, n, [descending](T v) { return radix_key(v, descending); });
    }
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

// (value, index) pairs: argsort and unique_parts.
template <typename T>
void sort_keyed_serial(Keyed<T>* a, std::int64_t n, bool stable, bool descending) {
    if constexpr (ops::is_int<T>) {
        if (n >= kRadixMinPairs)
            return radix_sort(a, n, [descending](const Keyed<T>& e) { return radix_key(e.value, descending); });
    }
    const Order<T> order{descending};
    sort_buffer(a, n, stable, [&](const Keyed<T>& x, const Keyed<T>& y) { return order(x.value, y.value); });
}

// ---- one long lane on several threads (Tier 3 item 13) ---------------------
// ARCHITECTURE §6.2 used to sort a single lane on one thread (user decision
// 2026-11-10 to change it). A lane of 2 x kSortPiece or more is cut into
// pieces of kSortPiece elements -- by length alone, never by thread count --
// each sorted by the serial algorithm on the pool. Then the sorted runs are
// merged pairwise, level by level. A merge is cut into output chunks of
// kMergeChunk; where a chunk starts in each run is found by binary search
// (the "merge path"), so the chunks of every level run in parallel too.
// Ties take the left run first: merging stable runs gives the stable result
// exactly, and a values-only sort gives the same values.
//
// The merges cost passes a one-thread sort does not make (sort(i) of 1M:
// 5.8 ms radix, 13.6 ms in pieces on one thread), so pieces are used only
// when the pool can run them -- the result is the same either way. Except
// for a float sort with stable=False: there the NEON sort's order of -0.0
// and +0.0 depends on the pieces, so it always takes them, keeping the
// result independent of the thread count (DESIGN §2.4).

inline constexpr std::int64_t kMergeChunk = 1 << 15;

// How many elements of A come first among the first k of merge(A, B), A
// winning ties.
template <typename E, typename Less>
std::int64_t co_rank(std::int64_t k, const E* A, std::int64_t na, const E* B, std::int64_t nb, Less less) {
    std::int64_t lo = std::max<std::int64_t>(0, k - nb), hi = std::min(k, na);
    while (lo < hi) {
        const std::int64_t i = lo + (hi - lo) / 2, j = k - i;
        if (j > 0 && i < na && !less(B[j - 1], A[i])) lo = i + 1;  // A[i] goes before B[j-1]: take more of A
        else hi = i;
    }
    return lo;
}

template <typename E, typename SortPiece, typename Less>
void sort_long(E* a, std::int64_t n, SortPiece sort_piece, Less less) {
    const std::int64_t pieces = (n + kSortPiece - 1) / kSortPiece;
    parallel::parallel_for(pieces, [&](std::int64_t t) {
        const std::int64_t begin = t * kSortPiece;
        sort_piece(a + begin, std::min(kSortPiece, n - begin));
    });
    auto tmp = buffer<E>(n);
    E* src = a;
    E* dst = tmp.get();
    struct Task {
        std::int64_t left, mid, right, k0, k1;  // runs [left, mid) and [mid, right); output [left+k0, left+k1)
    };
    std::vector<Task> tasks;
    for (std::int64_t w = kSortPiece; w < n; w *= 2) {
        tasks.clear();
        for (std::int64_t left = 0; left < n; left += 2 * w) {
            const std::int64_t mid = std::min(left + w, n), right = std::min(left + 2 * w, n);
            for (std::int64_t k = 0; k < right - left; k += kMergeChunk)
                tasks.push_back({left, mid, right, k, std::min(k + kMergeChunk, right - left)});
        }
        parallel::parallel_for(static_cast<std::int64_t>(tasks.size()), [&](std::int64_t t) {
            const Task& c = tasks[t];
            const E* A = src + c.left;
            const E* B = src + c.mid;
            const std::int64_t na = c.mid - c.left, nb = c.right - c.mid;
            std::int64_t i = co_rank(c.k0, A, na, B, nb, less), j = c.k0 - i;
            const std::int64_t i1 = co_rank(c.k1, A, na, B, nb, less), j1 = c.k1 - i1;
            E* out = dst + c.left + c.k0;
            while (i < i1 && j < j1) {  // branch-free: random data mispredicts half the time
                const bool take_b = less(B[j], A[i]);
                *out++ = take_b ? B[j] : A[i];
                j += take_b;
                i += !take_b;
            }
            out = std::copy(A + i, A + i1, out);
            std::copy(B + j, B + j1, out);
        });
        std::swap(src, dst);
    }
    if (src != a) std::copy(src, src + n, a);
}

template <typename T>
void sort_values(T* a, std::int64_t n, bool stable, bool descending) {
    // int64 takes the radix sort, parallel by itself
    if (ops::is_int<T> || !use_pieces(n, ops::is_float<T> && !stable))
        return sort_values_serial(a, n, stable, descending);
    sort_long(a, n, [&](T* p, std::int64_t m) { sort_values_serial(p, m, stable, descending); }, Order<T>{descending});
}

template <typename T>
void sort_keyed(Keyed<T>* a, std::int64_t n, bool stable, bool descending) {
    if (ops::is_int<T> || !use_pieces(n, false)) return sort_keyed_serial(a, n, stable, descending);  // pairs: always stable
    const Order<T> order{descending};
    sort_long(a, n, [&](Keyed<T>* p, std::int64_t m) { sort_keyed_serial(p, m, stable, descending); },
              [order](const Keyed<T>& x, const Keyed<T>& y) { return order(x.value, y.value); });
}

// ------------------------------------------------------------ sort lanes

struct SortCfg {
    bool descending, stable;
};

template <typename T>
void sort_lane(const std::byte* x, std::byte* o, const LaneCtx& c) {
    const SortCfg& cfg = *static_cast<const SortCfg*>(c.user);
    const LaneBuffer<T> buf(c.n, 0);
    const std::int64_t m = load_nans_last<T>(x, c.x_stride, c.n, buf.get(), [](std::int64_t, T v) { return v; });
    sort_values(buf.get(), m, cfg.stable, cfg.descending);
    for (std::int64_t i = 0; i < c.n; ++i) *reinterpret_cast<T*>(o + i * c.o_stride) = buf[i];
}

template <typename T>
void argsort_lane(const std::byte* x, std::byte* o, const LaneCtx& c) {
    const SortCfg& cfg = *static_cast<const SortCfg*>(c.user);
    const LaneBuffer<Keyed<T>> buf(c.n, 0);
    const std::int64_t m = load_nans_last<T>(x, c.x_stride, c.n, buf.get(),
                                             [](std::int64_t i, T v) { return Keyed<T>{v, i}; });
    sort_keyed(buf.get(), m, cfg.stable, cfg.descending);
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
        sort_keyed(buf.get(), m, /*stable=*/true, /*descending=*/false);
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
