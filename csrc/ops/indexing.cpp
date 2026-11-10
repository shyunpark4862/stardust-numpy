#include "ops/indexing.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "ops/copy.hpp"
#include "kernels/cast.hpp"
#include "kernels/elementwise.hpp"
#include "ops/creation.hpp"
#include "parallel/pool.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace sd {

namespace {

enum class Kind { Int, Slice, Ellipsis, NewAxis, Array };

struct Item {
    Item(Kind k, nb::handle o, std::int64_t v = 0) : kind(k), obj(o), value(v) {}
    Kind kind;
    nb::handle obj;
    std::int64_t value = 0;  // Kind::Int
    nb::object array;        // Kind::Array: an int64 index array or a bool mask
    bool mask = false;
    int axes() const { return kind == Kind::Array && mask ? as_ndarray(array).ndim : 1; }  // axes consumed
};

[[noreturn]] void raise(PyObject* type, const std::string& msg) {
    PyErr_SetString(type, msg.c_str());
    throw nb::python_error();
}

Item array_item(nb::handle o, nb::object arr) {
    const DType d = as_ndarray(arr).dtype;
    if (d != DType::Int64 && d != DType::Bool)
        raise(PyExc_IndexError, std::string("arrays used as indices must be of integer or boolean type, not ") +
                                    dtype_name(d));
    Item it(Kind::Array, o);
    it.array = std::move(arr);
    it.mask = d == DType::Bool;
    return it;
}

Item classify(nb::handle o) {
    PyObject* p = o.ptr();
    if (p == Py_None) return Item(Kind::NewAxis, o);
    if (p == Py_Ellipsis) return Item(Kind::Ellipsis, o);
    if (PySlice_Check(p)) return Item(Kind::Slice, o);
    // numpy reads a bool as a 0-d mask; with no 0-d arrays that is just
    // confusing next to x[1] (M6 decision).
    if (PyBool_Check(p)) raise(PyExc_IndexError, "a bool is not a valid index; use an int, or a boolean array");
    if (is_ndarray(o)) return array_item(o, nb::borrow(o));
    const bool text = PyUnicode_Check(p) || PyBytes_Check(p);  // sequences, but never indices
    if (!text && PySequence_Check(p)) {
        // A list (or numpy array, ...) is an integer array or a mask. An empty
        // list would come out float64 from asarray; as an index it means "no
        // positions".
        if ((PyList_Check(p) || PyTuple_Check(p)) && PySequence_Size(p) == 0) {
            const std::int64_t zero[] = {0};
            return array_item(o, wrap(new_array(DType::Int64, zero)));
        }
        return array_item(o, asarray(o, nb::none(), nb::none()));
    }
    if (!text && PyIndex_Check(p)) {  // int, or anything with __index__ (e.g. numpy.int64)
        nb::object i = nb::steal(PyNumber_Index(p));
        if (!i.is_valid()) throw nb::python_error();
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
        if (overflow != 0) raise(PyExc_IndexError, "index is out of bounds");
        return Item(Kind::Int, o, v);
    }
    raise(PyExc_IndexError,
          "only integers, slices (`:`), ellipsis (`...`), newaxis (`None`) and integer or boolean arrays are "
          "valid indices");
}

// The items of one key, stored in place: a key has at most 32 axis-consuming
// items, 32 newaxis and one ellipsis, so a fixed capacity is enough, and a
// std::vector's growth allocations were most of the cost of x[1, 2] (M7.7).
class Items {
public:
    static constexpr int kCapacity = 2 * kMaxNdim + 1;
    Items() = default;
    Items(const Items&) = delete;
    Items& operator=(const Items&) = delete;
    ~Items() {
        for (int i = 0; i < n_; ++i) (*this)[i].~Item();
    }
    void push_back(Item&& it) { new (slot(n_++)) Item(std::move(it)); }
    int size() const { return n_; }
    Item& operator[](int i) { return *std::launder(reinterpret_cast<Item*>(slot(i))); }
    const Item& operator[](int i) const { return *std::launder(reinterpret_cast<const Item*>(slot(i))); }
    Item* begin() { return &(*this)[0]; }
    Item* end() { return begin() + n_; }

private:
    std::byte* slot(int i) { return storage_ + static_cast<size_t>(i) * sizeof(Item); }
    const std::byte* slot(int i) const { return storage_ + static_cast<size_t>(i) * sizeof(Item); }
    alignas(Item) std::byte storage_[kCapacity * sizeof(Item)];  // uninitialized until pushed
    int n_ = 0;
};

// How x[key] is computed. The basic items (slices, newaxis, ...) are applied
// to make `view`; the axes the advanced items aim at are kept whole in it, as
// the consecutive run [first, first + indices.size()). Each of those axes gets
// one int64 index array; they broadcast together to the "index shape" B, and
// the result is  view.shape[:first] + B + view.shape[first + count:].
//
// A single boolean mask as the only advanced item is kept as the mask itself
// (`mask`, covering axes [first, first + mask.ndim)): it is applied by
// walking it once and copying where it is true ("compress", ARCHITECTURE.md
// §6.1) instead of first turning it into coordinate arrays (M7.8).
struct Plan {
    NDArray view;  // ndim may be 0 internally (all ints): never escapes
    int first = 0;
    std::vector<NDArray> indices;  // empty: basic indexing only (unless mask_only)
    bool mask_only = false;
    NDArray mask;
    bool advanced() const { return mask_only || !indices.empty(); }
};

// The positions of the true elements of a mask, one int64 array per mask axis
// (in C order: what numpy's nonzero gives).
std::vector<NDArray> mask_positions(const NDArray& m) {
    std::int64_t count = 0;
    for_each_element<1>(m.shape_span(), {m.data}, {m.strides.data()},
                        [&](std::byte* p) { count += *reinterpret_cast<const bool*>(p); });
    std::vector<NDArray> out;
    const std::int64_t shape[] = {count};
    for (int d = 0; d < m.ndim; ++d) out.push_back(new_array(DType::Int64, shape));
    std::array<std::int64_t, kMaxNdim> idx{};
    std::int64_t k = 0;
    for_each_element<1>(m.shape_span(), {m.data}, {m.strides.data()}, [&](std::byte* p) {
        if (*reinterpret_cast<const bool*>(p)) {
            for (int d = 0; d < m.ndim; ++d) reinterpret_cast<std::int64_t*>(out[d].data)[k] = idx[d];
            ++k;
        }
        for (int d = m.ndim - 1; d >= 0; --d) {  // advance the C-order multi-index
            if (++idx[d] < m.shape[d]) break;
            idx[d] = 0;
        }
    });
    return out;
}

Plan make_plan(nb::handle x, nb::handle key) {
    const NDArray& a = as_ndarray(x);
    Items items;
    if (PyTuple_Check(key.ptr())) {
        if (PyTuple_GET_SIZE(key.ptr()) > Items::kCapacity)
            raise(PyExc_IndexError, "too many indices (more than " + std::to_string(Items::kCapacity) + ")");
        for (nb::handle h : nb::borrow<nb::tuple>(key)) items.push_back(classify(h));
    } else {
        items.push_back(classify(key));
    }

    int consumed = 0, ellipses = 0;
    bool any_array = false;
    for (const Item& it : items) {
        if (it.kind == Kind::Int || it.kind == Kind::Slice || it.kind == Kind::Array) consumed += it.axes();
        if (it.kind == Kind::Ellipsis) ++ellipses;
        any_array |= it.kind == Kind::Array;
    }
    if (ellipses > 1) raise(PyExc_IndexError, "an index can only have a single ellipsis ('...')");
    if (consumed > a.ndim)
        raise(PyExc_IndexError, "too many indices for array: array is " + std::to_string(a.ndim) +
                                    "-dimensional, but " + std::to_string(consumed) + " were indexed");

    // Next to an array, an int is an advanced index too (numpy's rule). The
    // advanced items must sit side by side: numpy moves the index dimensions
    // to the front otherwise, a rule stardust removes by refusing (API.md §4).
    auto advanced = [&](const Item& it) { return it.kind == Kind::Array || (any_array && it.kind == Kind::Int); };
    int first_item = -1, last_item = -1;
    for (int i = 0; i < items.size(); ++i)
        if (advanced(items[i])) {
            if (first_item < 0) first_item = i;
            last_item = i;
        }
    for (int i = first_item; any_array && i <= last_item; ++i)
        if (!advanced(items[i]))
            raise(PyExc_IndexError,
                  "advanced indices (arrays, and ints next to them) separated by a slice, newaxis or ellipsis are "
                  "not supported; index in two steps, e.g. x[0][:, idx]");

    int n_advanced = 0;
    for (const Item& it : items) n_advanced += advanced(it);

    Plan plan;
    NDArray& v = plan.view;
    v = view_of(x);
    v.ndim = 0;
    std::byte* data = a.data;
    int src = 0;  // next axis of `a` to consume
    auto push = [&](std::int64_t extent, std::int64_t stride) {
        if (v.ndim == kMaxNdim)
            raise(PyExc_IndexError, "the result would have more than " + std::to_string(kMaxNdim) + " dimensions");
        v.shape[v.ndim] = extent;
        v.strides[v.ndim] = stride;
        ++v.ndim;
    };
    auto keep_axis = [&] {
        push(a.shape[src], a.strides[src]);
        ++src;
    };

    bool started = false;
    for (const Item& it : items) {
        if (advanced(it)) {
            if (!started) plan.first = v.ndim;
            started = true;
            if (it.kind == Kind::Int) {
                const std::int64_t one[] = {1};
                NDArray s = new_array(DType::Int64, one);
                *reinterpret_cast<std::int64_t*>(s.data) = it.value;
                plan.indices.push_back(std::move(s));
                keep_axis();
            } else if (!it.mask) {
                plan.indices.push_back(as_ndarray(it.array));
                keep_axis();
            } else {
                const NDArray& m = as_ndarray(it.array);
                for (int d = 0; d < m.ndim; ++d)
                    if (m.shape[d] != a.shape[src + d])
                        raise(PyExc_IndexError, "boolean index did not match indexed array along axis " +
                                                    std::to_string(src + d) + "; size of axis is " +
                                                    std::to_string(a.shape[src + d]) +
                                                    " but size of corresponding boolean axis is " +
                                                    std::to_string(m.shape[d]));
                if (n_advanced == 1) {
                    plan.mask_only = true;
                    plan.mask = m;
                } else {
                    for (NDArray& pos : mask_positions(m)) plan.indices.push_back(std::move(pos));
                }
                for (int d = 0; d < m.ndim; ++d) keep_axis();
            }
            continue;
        }
        switch (it.kind) {
            case Kind::Int: {
                const std::int64_t n = a.shape[src];
                const std::int64_t i = it.value < 0 ? it.value + n : it.value;
                if (i < 0 || i >= n)
                    raise(PyExc_IndexError, "index " + std::to_string(it.value) + " is out of bounds for axis " +
                                                std::to_string(src) + " with size " + std::to_string(n));
                data += i * a.strides[src];
                ++src;
                break;
            }
            case Kind::Slice: {
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(it.obj.ptr(), &start, &stop, &step) < 0) throw nb::python_error();
                const Py_ssize_t n = PySlice_AdjustIndices(a.shape[src], &start, &stop, step);
                // An empty slice may "start" outside the array; leave the
                // pointer alone rather than point past the buffer.
                if (n > 0) data += start * a.strides[src];
                push(n, a.strides[src] * step);
                ++src;
                break;
            }
            case Kind::NewAxis:
                push(1, 0);
                break;
            case Kind::Ellipsis:
                for (int k = 0; k < a.ndim - consumed; ++k) keep_axis();
                break;
            case Kind::Array:
                break;  // handled above
        }
    }
    while (src < a.ndim) keep_axis();  // unindexed trailing axes are kept whole
    v.data = data;
    v.update_flags();
    return plan;
}

// ------------------------------------------------------- gather and scatter

// For every point of the index shape B (C order): the byte offset into the
// plan's view, and -- when `w_strides` is given -- the byte offset into a
// second array laid out over the result shape (the value of a scatter).
// B: the index arrays broadcast together.
std::vector<std::int64_t> index_shape(const Plan& plan) {
    std::vector<std::span<const std::int64_t>> shapes;
    for (const NDArray& ix : plan.indices) shapes.push_back(ix.shape_span());
    std::vector<std::int64_t> b = broadcast_shapes(shapes);
    if (plan.view.ndim - static_cast<int>(plan.indices.size()) + static_cast<int>(b.size()) > kMaxNdim)
        raise(PyExc_IndexError, "the result would have more than " + std::to_string(kMaxNdim) + " dimensions");
    return b;
}

// Walks the index shape B in C order, turning each point into the byte offset
// of its block in the plan's view (bounds-checked; negative indices wrap) and,
// when `w_strides` is given, into the offset in a second array laid out over
// the result shape (the value of a scatter). Offsets are produced a chunk at a
// time into small buffers and handed to emit(view_off, w_off, n, first_b)
// right away, so they never make a full pass through memory (M7.8).
template <typename Emit>
void for_each_offset_chunk(const Plan& plan, std::span<const std::int64_t> bshape, const std::int64_t* w_strides,
                           Emit&& emit) {
    constexpr std::int64_t kChunk = 1024;
    const NDArray& v = plan.view;
    const int k = static_cast<int>(plan.indices.size());
    const int bd = static_cast<int>(bshape.size());
    std::int64_t nb = 1;
    for (std::int64_t e : bshape) nb *= e;
    if (nb == 0) return;

    // Each index array broadcast to B: stride 0 on stretched axes.
    std::array<std::array<std::int64_t, kMaxNdim>, kMaxNdim> st;
    std::array<const std::byte*, kMaxNdim> ptr;
    for (int j = 0; j < k; ++j) {
        const NDArray& ix = plan.indices[j];
        const int offset = bd - ix.ndim;
        for (int d = 0; d < bd; ++d)
            st[j][d] = d < offset || ix.shape[d - offset] == 1 ? 0 : ix.strides[d - offset];
        ptr[j] = ix.data;
    }
    std::array<std::int64_t, kMaxNdim> bidx{};
    std::array<std::int64_t, kChunk> voff, woff;
    std::int64_t w = 0, fill = 0, chunk_start = 0;
    // The last B axis is a tight run; the odometer only moves between runs.
    const int last = bd - 1;
    const std::int64_t run = bshape[last];
    const std::int64_t w_last = w_strides ? w_strides[plan.first + last] : 0;
    auto check = [&](std::int64_t raw, int axis) {
        const std::int64_t n = v.shape[axis];
        const std::int64_t i = raw < 0 ? raw + n : raw;
        if (i < 0 || i >= n) [[unlikely]]
            raise(PyExc_IndexError, "index " + std::to_string(raw) + " is out of bounds for axis " +
                                        std::to_string(axis) + " with size " + std::to_string(n));
        return i;
    };
    for (std::int64_t done = 0; done < nb; done += run) {
        for (std::int64_t t = 0; t < run; ++t) {
            std::int64_t o = 0;
            for (int j = 0; j < k; ++j) {
                const std::int64_t raw = *reinterpret_cast<const std::int64_t*>(ptr[j] + t * st[j][last]);
                o += check(raw, plan.first + j) * v.strides[plan.first + j];
            }
            voff[fill] = o;
            woff[fill] = w + t * w_last;
            if (++fill == kChunk) {
                emit(voff.data(), woff.data(), fill, chunk_start);
                chunk_start += fill;
                fill = 0;
            }
        }
        for (int d = last - 1; d >= 0; --d) {  // C-order odometer over the other B axes
            for (int j = 0; j < k; ++j) ptr[j] += st[j][d];
            if (w_strides) w += w_strides[plan.first + d];
            if (++bidx[d] < bshape[d]) break;
            for (int j = 0; j < k; ++j) ptr[j] -= st[j][d] * bshape[d];
            if (w_strides) w -= w_strides[plan.first + d] * bshape[d];
            bidx[d] = 0;
        }
    }
    if (fill > 0) emit(voff.data(), woff.data(), fill, chunk_start);
}

// The result shape: view.shape[:first] + B + view.shape[first + count:].
std::vector<std::int64_t> result_shape(const Plan& plan, std::span<const std::int64_t> b) {
    const NDArray& v = plan.view;
    const int k = plan.mask_only ? plan.mask.ndim : static_cast<int>(plan.indices.size());
    std::vector<std::int64_t> shape(v.shape.begin(), v.shape.begin() + plan.first);
    shape.insert(shape.end(), b.begin(), b.end());
    shape.insert(shape.end(), v.shape.begin() + plan.first + k, v.shape.begin() + v.ndim);
    return shape;
}

// The fresh result of an advanced index, laid out like the view (M10+, user
// decision 2026-11-10), as the result of an elementwise operation is: the
// view's axes keep their memory order, and the index axes B take the place
// of the indexed axes (at the outermost of them), in C order among
// themselves. An F-ordered mT[:, rows] is then an F-ordered result, filled
// column by column, contiguous on both sides (it was 2.6x numpy into a C
// result). The gather loops step through B as one block (point b at
// b * stride of the last B axis); if the layout breaks that (axes of extent
// 1 or stride 0 can tie), the result falls back to C order.
NDArray advanced_result(const Plan& plan, std::span<const std::int64_t> bshape) {
    const NDArray& v = plan.view;
    const int k = plan.mask_only ? plan.mask.ndim : static_cast<int>(plan.indices.size());
    const int bd = static_cast<int>(bshape.size());
    const std::vector<std::int64_t> shape = result_shape(plan, bshape);
    std::int64_t outer = 0;
    for (int j = 0; j < k; ++j) outer = std::max(outer, std::abs(v.strides[plan.first + j]));
    NDArray proto;
    proto.ndim = static_cast<int>(shape.size());
    for (int d = 0; d < proto.ndim; ++d) {
        proto.shape[d] = shape[d];
        proto.strides[d] = d < plan.first ? v.strides[d] : d < plan.first + bd ? outer : v.strides[d - bd + k];
    }
    NDArray out = new_array_like(proto, v.dtype);
    std::int64_t expect = out.strides[plan.first + bd - 1];
    for (int j = bd - 1; j >= 0; --j) {
        if (bshape[j] > 1 && out.strides[plan.first + j] != expect) return new_array(v.dtype, shape);
        expect *= bshape[j];
    }
    return out;
}

// `a` restricted to the "block" axes (everything but the index axes):
// shape/strides of a's axes [0, first) and [first + skip, ndim).
NDArray block_of(const NDArray& a, int first, int skip) {
    NDArray b = a;
    b.ndim = 0;
    for (int d = 0; d < a.ndim; ++d) {
        if (d >= first && d < first + skip) continue;
        b.shape[b.ndim] = a.shape[d];
        b.strides[b.ndim] = a.strides[d];
        ++b.ndim;
    }
    return b;
}

// ---- the common case: one index array selecting single elements ----------
// x[idx] / x[idx] = v on a 1-D view, same dtype. Measured in M9+ (see
// experiments/gather_probe): the general path's per-element loop over "k
// index arrays", re-reading their strides and shapes, and its separate
// offset buffer cost 3.5x the work of the plain loop below (read an index,
// fix a negative one, copy one element). Checking every index first, in a
// pass of its own, keeps that loop free of error handling -- so it can run on
// several threads -- and makes x[idx] = v all-or-nothing: an out-of-bounds
// index raises before anything is written. (A gather has nothing to protect
// and checks inside its loop instead.)

struct OneIndex {
    const std::byte* base;  // the view's data
    std::int64_t len, stride;
    std::atomic<bool>* bad = nullptr;
};

inline std::int64_t wrap_index(std::int64_t raw, std::int64_t len) { return raw < 0 ? raw + len : raw; }

void check_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const OneIndex& c = *static_cast<const OneIndex*>(ctx);
    bool bad = false;
    for (std::int64_t t = 0; t < n; ++t)  // branch-free: vectorizes
        bad |= static_cast<std::uint64_t>(wrap_index(load_as<std::int64_t>(args[0] + t * s[0]), c.len)) >=
               static_cast<std::uint64_t>(c.len);
    if (bad) c.bad->store(true, std::memory_order_relaxed);
}

// Raises IndexError for the first (C order) index of `ix` outside [-len, len).
void check_indices(const NDArray& ix, std::int64_t len, int axis) {
    std::atomic<bool> bad{false};
    OneIndex c{nullptr, len, 0, &bad};
    const IterOperand ops[] = {operand(ix)};
    NdIter(ix.shape_span(), ops).run(&check_loop, &c);
    if (!bad.load()) return;
    for_each_element<1>(ix.shape_span(), {ix.data}, {ix.strides.data()}, [&](std::byte* p) {
        const std::int64_t raw = load_as<std::int64_t>(p);
        if (static_cast<std::uint64_t>(wrap_index(raw, len)) >= static_cast<std::uint64_t>(len))
            raise(PyExc_IndexError, "index " + std::to_string(raw) + " is out of bounds for axis " +
                                        std::to_string(axis) + " with size " + std::to_string(len));
    });
}

// args: index, out. A gather writes only its own fresh result, so it needs
// no separate check pass: a bad index is noted (branch-free) and read as
// index 0 to stay inside the array; the caller raises afterwards.
template <typename T>
void gather_one_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const OneIndex& c = *static_cast<const OneIndex*>(ctx);
    bool bad = false;
    for (std::int64_t t = 0; t < n; ++t) {
        std::int64_t i = wrap_index(load_as<std::int64_t>(args[0] + t * s[0]), c.len);
        const bool out_of_bounds = static_cast<std::uint64_t>(i) >= static_cast<std::uint64_t>(c.len);
        bad |= out_of_bounds;
        i = out_of_bounds ? 0 : i;
        *reinterpret_cast<T*>(args[1] + t * s[1]) = load_as<T>(c.base + i * c.stride);
    }
    if (bad) c.bad->store(true, std::memory_order_relaxed);
}

// args: index, value
template <typename T>
void scatter_one_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const OneIndex& c = *static_cast<const OneIndex*>(ctx);
    for (std::int64_t t = 0; t < n; ++t) {
        const std::int64_t i = wrap_index(load_as<std::int64_t>(args[0] + t * s[0]), c.len);
        *reinterpret_cast<T*>(const_cast<std::byte*>(c.base) + i * c.stride) = load_as<T>(args[1] + t * s[1]);
    }
}

bool one_index(const Plan& plan) { return plan.indices.size() == 1 && plan.view.ndim == 1; }

nb::object gather_one(const Plan& plan) {
    const NDArray& v = plan.view;
    const NDArray& ix = plan.indices[0];
    NDArray out = new_array(v.dtype, ix.shape_span());  // the result shape is the index shape
    std::atomic<bool> bad{false};
    OneIndex c{v.data, v.shape[0], v.strides[0], &bad};
    const IterOperand ops[] = {operand(ix), operand(out)};
    visit_dtype(v.dtype, [&]<typename T>(std::type_identity<T>) {
        NdIter(ix.shape_span(), ops).run(&gather_one_loop<T>, &c);  // independent outputs: may use threads
    });
    if (bad.load()) check_indices(ix, v.shape[0], plan.first);  // raises, naming the first bad index
    return wrap(std::move(out));
}

// ---- one index array with whole blocks around it (M10+) ------------------
// x[..., idx, ...] = take along one axis: result[pre, b, post] =
// view[pre, idx[b], post]. The general path walked the index points and
// copied each point's block through an iterator of its own -- fine for long
// contiguous blocks (m[rows]), slow when the blocks are short (p3[idx]: one
// iterator per 3 elements, 3.4x numpy) or far apart (m[:, rows]: 1000
// strided columns written into a C-order result, 1.4x numpy).
//
// Instead it is one elementwise loop over the result shape with three
// operands: the view without the indexed axis (stride 0 along B), the index
// array (stride 0 along the block axes) and the result. The iterator then
// picks the memory-friendly order and the parallel split, as for any
// elementwise operation; the kernel turns the index into a byte offset.
// Every index is checked first (B elements, not the result's), so the loop
// itself has no error path.

struct Take {
    std::int64_t len, stride;  // the indexed axis of the view
};

// args: view (block base), index, out
template <typename T>
void take_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const Take& c = *static_cast<const Take*>(ctx);
    if (s[1] == 0) {  // one index for the whole run: a block copy (m[rows])
        const std::byte* src = args[0] + wrap_index(load_as<std::int64_t>(args[1]), c.len) * c.stride;
        if (s[0] == sizeof(T) && s[2] == sizeof(T)) {
            std::memcpy(args[2], src, static_cast<size_t>(n) * sizeof(T));
        } else {
            for (std::int64_t t = 0; t < n; ++t) *reinterpret_cast<T*>(args[2] + t * s[2]) = load_as<T>(src + t * s[0]);
        }
        return;
    }
    for (std::int64_t t = 0; t < n; ++t) {
        const std::int64_t i = wrap_index(load_as<std::int64_t>(args[1] + t * s[1]), c.len);
        *reinterpret_cast<T*>(args[2] + t * s[2]) = load_as<T>(args[0] + t * s[0] + i * c.stride);
    }
}

nb::object gather_take(const Plan& plan) {
    const NDArray& v = plan.view;
    const NDArray& ix = plan.indices[0];
    check_indices(ix, v.shape[plan.first], plan.first);
    NDArray out = advanced_result(plan, ix.shape_span());
    // The view and the index array, both stretched to the result shape.
    Dims vs{}, is{};
    for (int d = 0; d < out.ndim; ++d) {
        if (d < plan.first) {
            vs[d] = v.strides[d];
        } else if (d < plan.first + ix.ndim) {
            is[d] = ix.strides[d - plan.first];
        } else {
            vs[d] = v.strides[d - ix.ndim + 1];
        }
    }
    const auto rshape = out.shape_span();
    const IterOperand ops[] = {{v.data, rshape, {vs.data(), rshape.size()}},
                               {ix.data, rshape, {is.data(), rshape.size()}},
                               operand(out)};
    Take c{v.shape[plan.first], v.strides[plan.first]};
    visit_dtype(v.dtype, [&]<typename T>(std::type_identity<T>) { NdIter(rshape, ops).run(&take_loop<T>, &c); });
    return wrap(std::move(out));
}

// Calls visit(copy), where copy(s, d) copies one block (`src_block`'s shape
// and strides at s) to d (`dst_block`'s layout), converting dtype. Choosing
// the copy once, outside the per-point loop, lets the common case -- single
// elements of the same dtype -- be a plain typed assignment the compiler
// inlines, instead of a call through a kernel pointer per element (M7.8).
template <typename Visit>
void with_block_copy(const NDArray& src_block, const NDArray& dst_block, Visit&& visit) {
    if (src_block.ndim == 0 && src_block.dtype == dst_block.dtype) {
        visit_dtype(src_block.dtype, [&]<typename T>(std::type_identity<T>) {
            visit([](const std::byte* s, std::byte* d) { *reinterpret_cast<T*>(d) = load_as<T>(s); });
        });
        return;
    }
    const Loop loop = cast_kernel(src_block.dtype, dst_block.dtype);
    if (src_block.ndim == 0) {
        visit([loop](const std::byte* s, std::byte* d) {
            static constexpr std::int64_t zero[] = {0, 0};
            std::byte* args[] = {const_cast<std::byte*>(s), d};
            loop(args, 1, zero, nullptr);
        });
        return;
    }
    visit([&](const std::byte* s, std::byte* d) {
        const IterOperand ops[] = {{const_cast<std::byte*>(s), src_block.shape_span(), src_block.strides_span()},
                                   {d, dst_block.shape_span(), dst_block.strides_span()}};
        NdIter(dst_block.shape_span(), ops).run(loop);
    });
}

// ---- boolean masks in fixed chunks (M10+) ----------------------------------
// The mask is cut, in C order, into chunks of kMaskChunk elements. Pass 1
// counts the true elements of every chunk; their running sum says where
// each chunk's selection starts in the result (or in the assigned value);
// pass 2 walks each chunk on its own. Both passes run on the pool. The
// chunks depend on the mask's size only and every element lands at its C
// order position, so the result is the serial one exactly.

inline constexpr std::int64_t kMaskChunk = 1 << 14;

// Calls on_run(mask, mask_step, view_offset, view_step, n) for the runs (parts
// of the last mask axis) covering C-order mask elements [begin, end).
// view_offset is the byte offset in the view, over the mask axes only.
template <typename F>
void mask_runs(const NDArray& m, const std::int64_t* vs, std::int64_t begin, std::int64_t end, F&& on_run) {
    const int last = m.ndim - 1;
    std::array<std::int64_t, kMaxNdim> idx{};
    const std::byte* mp = m.data;
    std::int64_t vo = 0;
    for (std::int64_t d = last, rem = begin; d >= 0; --d) {  // the multi-index of `begin`
        idx[d] = rem % m.shape[d];
        rem /= m.shape[d];
        mp += idx[d] * m.strides[d];
        vo += idx[d] * vs[d];
    }
    for (std::int64_t pos = begin; pos < end;) {
        const std::int64_t n = std::min(m.shape[last] - idx[last], end - pos);
        on_run(mp, m.strides[last], vo, vs[last], n);
        pos += n;
        idx[last] += n;
        mp += n * m.strides[last];
        vo += n * vs[last];
        if (idx[last] < m.shape[last]) continue;
        mp -= m.shape[last] * m.strides[last];
        vo -= m.shape[last] * vs[last];
        idx[last] = 0;
        for (int d = last - 1; d >= 0; --d) {  // C-order odometer over the other mask axes
            mp += m.strides[d];
            vo += vs[d];
            if (++idx[d] < m.shape[d]) break;
            mp -= m.shape[d] * m.strides[d];
            vo -= m.shape[d] * vs[d];
            idx[d] = 0;
        }
    }
}

// Calls f(view_offset) for every true element of one run, in order. A
// contiguous run skips eight false elements at a time (sparse masks: x[m]
// with 1% true was 3.5x numpy, one branch per element).
template <typename F>
void for_true(const std::byte* mp, std::int64_t ms, std::int64_t vo, std::int64_t vstep, std::int64_t n, F&& f) {
    std::int64_t t = 0;
    if (ms == 1) {
        for (; t + 8 <= n; t += 8) {
            std::uint64_t word;
            std::memcpy(&word, mp + t, 8);
            if (word == 0) continue;
            for (int k = 0; k < 8; ++k)
                if (mp[t + k] != std::byte{0}) f(vo + (t + k) * vstep);
        }
    }
    for (; t < n; ++t)
        if (*reinterpret_cast<const bool*>(mp + t * ms)) f(vo + t * vstep);
}

struct MaskChunks {
    std::int64_t size = 0, chunks = 0, total = 0;
    std::vector<std::int64_t> start;  // where each chunk's selection begins
};

MaskChunks count_mask(const Plan& plan) {
    const NDArray& m = plan.mask;
    MaskChunks mc;
    mc.size = m.size();
    mc.chunks = (mc.size + kMaskChunk - 1) / kMaskChunk;
    mc.start.assign(static_cast<size_t>(mc.chunks) + 1, 0);
    const std::int64_t* vs = plan.view.strides.data() + plan.first;
    parallel::parallel_for(mc.chunks, [&](std::int64_t c) {
        std::int64_t count = 0;
        mask_runs(m, vs, c * kMaskChunk, std::min(mc.size, (c + 1) * kMaskChunk),
                  [&](const std::byte* mp, std::int64_t ms, std::int64_t, std::int64_t, std::int64_t n) {
                      if (ms == 1) {
                          const std::uint8_t* q = reinterpret_cast<const std::uint8_t*>(mp);
                          std::uint32_t part = 0;  // a run never exceeds kMaskChunk elements
                          for (std::int64_t t = 0; t < n; ++t) part += q[t];
                          count += part;
                      } else {
                          for (std::int64_t t = 0; t < n; ++t) count += *reinterpret_cast<const bool*>(mp + t * ms);
                      }
                  });
        mc.start[c + 1] = count;
    });
    for (std::int64_t c = 0; c < mc.chunks; ++c) mc.start[c + 1] += mc.start[c];
    mc.total = mc.start[mc.chunks];
    return mc;
}

// Calls on_true(b, view_offset) for every true element of chunk c, b being
// its position in the whole selection.
template <typename F>
void walk_chunk(const Plan& plan, const MaskChunks& mc, std::int64_t c, F&& on_true) {
    std::int64_t b = mc.start[c];
    mask_runs(plan.mask, plan.view.strides.data() + plan.first, c * kMaskChunk, std::min(mc.size, (c + 1) * kMaskChunk),
              [&](const std::byte* mp, std::int64_t ms, std::int64_t vo, std::int64_t vstep, std::int64_t n) {
                  for_true(mp, ms, vo, vstep, n, [&](std::int64_t off) { on_true(b++, off); });
              });
}

// x[mask] for single elements of one dtype, chunk c. A run is taken 256
// elements at a time; the trues of each piece are counted first (a vector
// sum), then
//   - few (< 32): only those are visited, skipping eight falses at a time;
//   - more: every element is copied into a small buffer and the buffer
//     position moves by the mask byte -- no branch to mispredict (a random
//     half-true mask cost 2.2 ms per 1M with one branch per element) --
//     and the buffer goes out with one memcpy.
// Writing the buffer, not the result, keeps every write inside the chunk's
// own part of the result.
template <typename T>
void compress_chunk(const Plan& plan, const MaskChunks& mc, std::int64_t c, std::byte* out, std::int64_t step) {
    constexpr std::int64_t kPiece = 256;
    const std::byte* src = plan.view.data;
    std::int64_t b = mc.start[c];
    auto put = [&](const T* buf, std::int64_t cnt) {
        if (step == sizeof(T)) std::memcpy(out + b * step, buf, static_cast<size_t>(cnt) * sizeof(T));
        else for (std::int64_t j = 0; j < cnt; ++j) *reinterpret_cast<T*>(out + (b + j) * step) = buf[j];
        b += cnt;
    };
    mask_runs(plan.mask, plan.view.strides.data() + plan.first, c * kMaskChunk, std::min(mc.size, (c + 1) * kMaskChunk),
              [&](const std::byte* mp, std::int64_t ms, std::int64_t vo, std::int64_t vstep, std::int64_t n) {
                  if (ms != 1) {
                      for_true(mp, ms, vo, vstep, n, [&](std::int64_t off) {
                          *reinterpret_cast<T*>(out + b++ * step) = load_as<T>(src + off);
                      });
                      return;
                  }
                  const std::uint8_t* q = reinterpret_cast<const std::uint8_t*>(mp);
                  T buf[kPiece];
                  for (std::int64_t t0 = 0; t0 < n; t0 += kPiece) {
                      const std::int64_t m = std::min(kPiece, n - t0);
                      std::uint32_t trues = 0;
                      for (std::int64_t t = 0; t < m; ++t) trues += q[t0 + t];
                      if (trues == 0) continue;
                      std::int64_t cnt = 0;
                      if (trues < 32) {
                          for_true(mp + t0, 1, vo + t0 * vstep, vstep, m,
                                   [&](std::int64_t off) { buf[cnt++] = load_as<T>(src + off); });
                      } else {
                          const std::byte* p = src + vo + t0 * vstep;
                          for (std::int64_t t = 0; t < m; ++t) {
                              buf[cnt] = load_as<T>(p + t * vstep);
                              cnt += q[t0 + t];
                          }
                      }
                      put(buf, cnt);
                  }
              });
}

// x[mask] = w for single elements of one dtype, chunk c (w: the value laid
// over the selection, step bytes apart, 0 for a broadcast scalar). The same
// pieces as compress_chunk; with many trues every element of the piece is
// rewritten, with w's next value or its own: x[i] = mask[i] ? w[b] : x[i].
// Only this chunk's elements are touched, so chunks stay independent.
template <typename T>
void expand_chunk(const Plan& plan, const MaskChunks& mc, std::int64_t c, const std::byte* w, std::int64_t step) {
    constexpr std::int64_t kPiece = 256;
    std::byte* dst = plan.view.data;
    std::int64_t b = mc.start[c];
    mask_runs(plan.mask, plan.view.strides.data() + plan.first, c * kMaskChunk, std::min(mc.size, (c + 1) * kMaskChunk),
              [&](const std::byte* mp, std::int64_t ms, std::int64_t vo, std::int64_t vstep, std::int64_t n) {
                  auto sparse = [&](const std::byte* mq, std::int64_t mstep, std::int64_t off0, std::int64_t m) {
                      for_true(mq, mstep, off0, vstep, m, [&](std::int64_t off) {
                          *reinterpret_cast<T*>(dst + off) = load_as<T>(w + b++ * step);
                      });
                  };
                  if (ms != 1) return sparse(mp, ms, vo, n);
                  const std::uint8_t* q = reinterpret_cast<const std::uint8_t*>(mp);
                  for (std::int64_t t0 = 0; t0 < n; t0 += kPiece) {
                      const std::int64_t m = std::min(kPiece, n - t0);
                      std::uint32_t trues = 0;
                      for (std::int64_t t = 0; t < m; ++t) trues += q[t0 + t];
                      if (trues < 32) {
                          if (trues > 0) sparse(mp + t0, 1, vo + t0 * vstep, m);
                          continue;
                      }
                      std::byte* p = dst + vo + t0 * vstep;
                      if (step == 0) {
                          // A broadcast scalar: a select on the raw words with
                          // a mask made from the mask byte. Written as
                          // `q ? val : *e` the compiler turned it back into a
                          // branch around the store (2.2 ms per 1M, measured).
                          using W = std::conditional_t<sizeof(T) % 8 == 0, std::uint64_t, std::uint8_t>;
                          constexpr int kWords = sizeof(T) / sizeof(W);
                          W val[kWords];
                          std::memcpy(val, w, sizeof(T));
                          for (std::int64_t t = 0; t < m; ++t) {
                              W* e = reinterpret_cast<W*>(p + t * vstep);
                              const W keep = static_cast<W>(q[t0 + t]) - 1;  // all ones where false
                              for (int j = 0; j < kWords; ++j) e[j] = (e[j] & keep) | (val[j] & ~keep);
                          }
                      } else {
                          for (std::int64_t t = 0; t < m; ++t) {
                              T* e = reinterpret_cast<T*>(p + t * vstep);
                              const T val = load_as<T>(w + b * step);  // b stays inside the selection: < total
                              *e = q[t0 + t] ? val : *e;
                              b += q[t0 + t];
                          }
                      }
                      if (step == 0) b += trues;
                  }
              });
}

// x[mask]: count, allocate, then copy the selected blocks chunk by chunk.
nb::object gather_mask(const Plan& plan, const MaskChunks& mc) {
    const NDArray& v = plan.view;
    const int k = plan.mask.ndim;
    const std::int64_t count[] = {mc.total};
    NDArray out = advanced_result(plan, count);
    const std::int64_t step = out.strides[plan.first];
    const NDArray src = block_of(v, plan.first, k), dst = block_of(out, plan.first, 1);
    if (src.ndim == 0) {
        visit_dtype(v.dtype, [&]<typename T>(std::type_identity<T>) {
            parallel::parallel_for(mc.chunks, [&](std::int64_t c) { compress_chunk<T>(plan, mc, c, out.data, step); });
        });
        return wrap(std::move(out));
    }
    with_block_copy(src, dst, [&](auto copy) {
        parallel::parallel_for(mc.chunks, [&](std::int64_t c) {
            walk_chunk(plan, mc, c, [&](std::int64_t b, std::int64_t off) { copy(src.data + off, dst.data + b * step); });
        });
    });
    return wrap(std::move(out));
}

nb::object gather(const Plan& plan) {
    if (one_index(plan)) return gather_one(plan);
    if (plan.indices.size() == 1) return gather_take(plan);
    const std::vector<std::int64_t> bshape = index_shape(plan);
    const int k = static_cast<int>(plan.indices.size());
    const int bd = static_cast<int>(bshape.size());
    NDArray out = advanced_result(plan, bshape);
    // The B axes of the result form one block (advanced_result), so index
    // point b starts b * (stride of the last B axis) bytes in.
    const std::int64_t step = out.strides[plan.first + bd - 1];
    const NDArray src = block_of(plan.view, plan.first, k), dst = block_of(out, plan.first, bd);
    with_block_copy(src, dst, [&](auto copy) {
        for_each_offset_chunk(plan, bshape, nullptr, [&](const std::int64_t* voff, const std::int64_t*, std::int64_t n,
                                                        std::int64_t b0) {
            for (std::int64_t i = 0; i < n; ++i) copy(src.data + voff[i], dst.data + (b0 + i) * step);
        });
    });
    return wrap(std::move(out));
}

// The right-hand side of an assignment as an array: a Python scalar becomes a
// 1-element array of the target dtype (broadcast later by the engine);
// anything else goes through asarray.
nb::object value_array(nb::handle value, DType target) {
    if (std::optional<DType> kind = scalar_dtype(value)) {
        check_assignable(*kind, target);
        const std::int64_t one[] = {1};
        NDArray s = new_array(target, one);
        visit_dtype(target, [&]<typename T>(std::type_identity<T>) {
            *reinterpret_cast<T*>(s.data) = scalar_from_python<T>(value, *kind);
        });
        return wrap(std::move(s));
    }
    if (is_ndarray(value)) return nb::borrow(value);
    return asarray(value, nb::none(), nb::none());
}

}  // namespace

// x[i], x[i, j, ...]: Python ints on the leading axes only -- the most
// common tiny calls -- skip the key machinery (Items, Plan): the offset is
// computed directly and the result is the element itself (one int per axis)
// or a view of the remaining axes. x[1, 2] took 58 ns, numpy 31 ns, nearly
// all of it in building and walking the general plan (M10+). Anything
// unusual -- a bool, an int too large, an index out of range -- returns an
// invalid object and takes the general path, which raises as usual.
nb::object leading_ints_fast(nb::handle x, PyObject* key) {
    const NDArray& a = as_ndarray(x);
    PyObject* const* items = &key;
    Py_ssize_t n = 1;
    if (PyTuple_Check(key)) {
        items = &PyTuple_GET_ITEM(key, 0);
        n = PyTuple_GET_SIZE(key);
    }
    if (n == 0 || n > a.ndim) return {};
    std::byte* p = a.data;
    for (Py_ssize_t d = 0; d < n; ++d) {
        if (!PyLong_CheckExact(items[d])) return {};
        int overflow = 0;
        long long i = PyLong_AsLongLongAndOverflow(items[d], &overflow);
        if (overflow) return {};
        if (i < 0) i += a.shape[d];
        if (i < 0 || i >= a.shape[d]) return {};
        p += i * a.strides[d];
    }
    if (n == a.ndim) return load_python(a.dtype, p);
    NDArray v = view_of(x);
    v.data = p;
    v.ndim = a.ndim - static_cast<int>(n);
    std::copy(a.shape.begin() + n, a.shape.begin() + a.ndim, v.shape.begin());
    std::copy(a.strides.begin() + n, a.strides.begin() + a.ndim, v.strides.begin());
    v.update_flags();
    return wrap(std::move(v));
}

nb::object getitem(nb::handle x, nb::handle key) {
    if (nb::object r = leading_ints_fast(x, key.ptr()); r.is_valid()) return r;
    Plan plan = make_plan(x, key);
    if (plan.mask_only) return gather_mask(plan, count_mask(plan));  // advanced: always a copy
    if (!plan.indices.empty()) return gather(plan);
    if (plan.view.ndim == 0) return load_python(plan.view.dtype, plan.view.data);
    return wrap(std::move(plan.view));
}

namespace {

// x[advanced key] = value: the value is broadcast (strictly) to the result
// shape, then each index point's block is written into x.
void scatter(nb::handle x, const Plan& plan, nb::handle value) {
    nb::object src_obj = value_array(value, plan.view.dtype);
    NDArray src = as_ndarray(src_obj);
    check_assignable(src.dtype, plan.view.dtype);  // S6
    if (may_share_memory(src, as_ndarray(x))) src = cast_copy(src, src.dtype);  // e.g. x[i] = x[::-1] (R3)

    // The value broadcast over the result shape (strict: never more axes).
    std::vector<std::int64_t> shape;
    MaskChunks mc;
    if (plan.mask_only) {
        mc = count_mask(plan);
        const std::int64_t count[] = {mc.total};
        shape = result_shape(plan, count);
    } else {
        shape = result_shape(plan, index_shape(plan));
    }
    const int nd = static_cast<int>(shape.size());
    const int off = nd - src.ndim;
    auto fail = [&] {
        return nb::value_error(("cannot broadcast a value of shape " + shape_str(src.shape_span()) +
                                " to the indexed shape " + shape_str(shape))
                                   .c_str());
    };
    if (off < 0) throw fail();
    NDArray w = src;
    w.ndim = nd;
    for (int d = nd - 1; d >= 0; --d) {
        const int sd = d - off;
        if (sd < 0 || src.shape[sd] == 1) w.strides[d] = 0;
        else if (src.shape[sd] == shape[d]) w.strides[d] = src.strides[sd];
        else throw fail();
        w.shape[d] = shape[d];
    }
    // Every index is checked before anything is written: an IndexError
    // leaves x as it was.
    for (size_t j = 0; j < plan.indices.size(); ++j)
        check_indices(plan.indices[j], plan.view.shape[plan.first + j], plan.first + static_cast<int>(j));
    if (one_index(plan) && w.dtype == plan.view.dtype) {
        // Index order (C order, IterMode::Naive) on one thread: with repeated
        // indices the last write must win, as in the general path.
        const NDArray& ix = plan.indices[0];
        OneIndex c{plan.view.data, plan.view.shape[0], plan.view.strides[0]};
        const IterOperand ops[] = {operand(ix), operand(w)};
        visit_dtype(w.dtype, [&]<typename T>(std::type_identity<T>) {
            NdIter(ix.shape_span(), ops, IterMode::Naive).run(&scatter_one_loop<T>, &c, Split::None);
        });
        return;
    }
    if (plan.mask_only) {
        const NDArray from = block_of(w, plan.first, 1), to = block_of(plan.view, plan.first, plan.mask.ndim);
        const std::int64_t step = w.strides[plan.first];
        if (from.ndim == 0 && from.dtype == to.dtype) {
            visit_dtype(to.dtype, [&]<typename T>(std::type_identity<T>) {
                parallel::parallel_for(mc.chunks, [&](std::int64_t c) { expand_chunk<T>(plan, mc, c, from.data, step); });
            });
            return;
        }
        with_block_copy(from, to, [&](auto copy) {  // every element is written once: chunks are independent
            parallel::parallel_for(mc.chunks, [&](std::int64_t c) {
                walk_chunk(plan, mc, c,
                           [&](std::int64_t b, std::int64_t off) { copy(from.data + b * step, to.data + off); });
            });
        });
        return;
    }
    const int k = static_cast<int>(plan.indices.size());
    const int bd = nd - (plan.view.ndim - k);
    const NDArray from = block_of(w, plan.first, bd), to = block_of(plan.view, plan.first, k);
    with_block_copy(from, to, [&](auto copy) {  // in order: with repeated indices the last write wins
        for_each_offset_chunk(plan, std::span(shape).subspan(plan.first, bd), w.strides.data(),
                              [&](const std::int64_t* voff, const std::int64_t* woff, std::int64_t n, std::int64_t) {
                                  for (std::int64_t i = 0; i < n; ++i) copy(from.data + woff[i], to.data + voff[i]);
                              });
    });
}

}  // namespace

void setitem(nb::handle x, nb::handle key, nb::handle value) {
    if (!as_ndarray(x).writeable) throw nb::value_error("assignment destination is read-only");
    Plan plan = make_plan(x, key);
    if (plan.advanced()) {
        scatter(x, plan, value);
        return;
    }
    NDArray& dst = plan.view;
    const std::optional<DType> kind = scalar_dtype(value);
    if (dst.ndim == 0) {
        // One element. Only a Python scalar fits: an array, even of shape
        // (1,), does not broadcast to () (strict broadcasting). No engine
        // needed either -- convert the value and store it.
        if (!kind) throw nb::value_error("only a Python scalar can be assigned to a single element");
        check_assignable(*kind, dst.dtype);
        visit_dtype(dst.dtype, [&]<typename T>(std::type_identity<T>) {
            const T v = scalar_from_python<T>(value, *kind);
            std::memcpy(dst.data, &v, sizeof(T));
        });
        return;
    }
    nb::object src = value_array(value, dst.dtype);
    assign_checked(dst, as_ndarray(src));
}

nb::object item_along_axis0(nb::handle x, std::int64_t i) {
    const NDArray& a = as_ndarray(x);
    const std::byte* p = a.data + i * a.strides[0];
    if (a.ndim == 1) return load_python(a.dtype, p);
    NDArray v = view_of(x);
    v.data = const_cast<std::byte*>(p);
    v.ndim = a.ndim - 1;
    std::copy(a.shape.begin() + 1, a.shape.begin() + a.ndim, v.shape.begin());
    std::copy(a.strides.begin() + 1, a.strides.begin() + a.ndim, v.strides.begin());
    v.update_flags();
    return wrap(std::move(v));
}

}  // namespace sd
