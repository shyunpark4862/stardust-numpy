#include "ops/indexing.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "ops/copy.hpp"
#include "kernels/cast.hpp"
#include "ops/creation.hpp"

#include <array>
#include <cstring>
#include <string>
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

// How x[key] is computed. The basic items (slices, newaxis, ...) are applied
// to make `view`; the axes the advanced items aim at are kept whole in it, as
// the consecutive run [first, first + indices.size()). Each of those axes gets
// one int64 index array; they broadcast together to the "index shape" B, and
// the result is  view.shape[:first] + B + view.shape[first + count:].
struct Plan {
    NDArray view;  // ndim may be 0 internally (all ints): never escapes
    int first = 0;
    std::vector<NDArray> indices;  // empty: basic indexing only
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
    std::vector<Item> items;
    if (PyTuple_Check(key.ptr())) {
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
    for (int i = 0; i < static_cast<int>(items.size()); ++i)
        if (advanced(items[i])) {
            if (first_item < 0) first_item = i;
            last_item = i;
        }
    for (int i = first_item; any_array && i <= last_item; ++i)
        if (!advanced(items[i]))
            raise(PyExc_IndexError,
                  "advanced indices (arrays, and ints next to them) separated by a slice, newaxis or ellipsis are "
                  "not supported; index in two steps, e.g. x[0][:, idx]");

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

    for (const Item& it : items) {
        if (advanced(it)) {
            if (plan.indices.empty()) plan.first = v.ndim;
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
                for (NDArray& pos : mask_positions(m)) plan.indices.push_back(std::move(pos));
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
struct Offsets {
    std::vector<std::int64_t> shape;  // B
    std::vector<std::int64_t> view, other;
};

// B: the index arrays broadcast together.
std::vector<std::int64_t> index_shape(const Plan& plan) {
    std::vector<std::span<const std::int64_t>> shapes;
    for (const NDArray& ix : plan.indices) shapes.push_back(ix.shape_span());
    std::vector<std::int64_t> b = broadcast_shapes(shapes);
    if (plan.view.ndim - static_cast<int>(plan.indices.size()) + static_cast<int>(b.size()) > kMaxNdim)
        raise(PyExc_IndexError, "the result would have more than " + std::to_string(kMaxNdim) + " dimensions");
    return b;
}

Offsets index_offsets(const Plan& plan, const std::int64_t* w_strides) {
    const NDArray& v = plan.view;
    const int k = static_cast<int>(plan.indices.size());
    Offsets off;
    off.shape = index_shape(plan);
    const int bd = static_cast<int>(off.shape.size());
    std::int64_t nb = 1;
    for (std::int64_t e : off.shape) nb *= e;
    if (nb == 0) return off;

    // Each index array broadcast to B: stride 0 on stretched axes.
    std::vector<std::array<std::int64_t, kMaxNdim>> st(k);
    for (int j = 0; j < k; ++j) {
        const NDArray& ix = plan.indices[j];
        const int offset = bd - ix.ndim;
        for (int d = 0; d < bd; ++d)
            st[j][d] = d < offset || ix.shape[d - offset] == 1 ? 0 : ix.strides[d - offset];
    }
    std::vector<const std::byte*> ptr(k);
    for (int j = 0; j < k; ++j) ptr[j] = plan.indices[j].data;
    std::array<std::int64_t, kMaxNdim> bidx{};
    off.view.resize(nb);
    if (w_strides) off.other.resize(nb);
    for (std::int64_t b = 0; b < nb; ++b) {
        std::int64_t o = 0;
        for (int j = 0; j < k; ++j) {
            const int axis = plan.first + j;
            const std::int64_t n = v.shape[axis];
            const std::int64_t raw = *reinterpret_cast<const std::int64_t*>(ptr[j]);
            const std::int64_t i = raw < 0 ? raw + n : raw;
            if (i < 0 || i >= n)
                raise(PyExc_IndexError, "index " + std::to_string(raw) + " is out of bounds for axis " +
                                            std::to_string(axis) + " with size " + std::to_string(n));
            o += i * v.strides[axis];
        }
        off.view[b] = o;
        if (w_strides) {
            std::int64_t w = 0;
            for (int d = 0; d < bd; ++d) w += bidx[d] * w_strides[plan.first + d];
            off.other[b] = w;
        }
        for (int d = bd - 1; d >= 0; --d) {  // C-order odometer over B
            for (int j = 0; j < k; ++j) ptr[j] += st[j][d];
            if (++bidx[d] < off.shape[d]) break;
            for (int j = 0; j < k; ++j) ptr[j] -= st[j][d] * off.shape[d];
            bidx[d] = 0;
        }
    }
    return off;
}

// The result shape: view.shape[:first] + B + view.shape[first + count:].
std::vector<std::int64_t> result_shape(const Plan& plan, std::span<const std::int64_t> b) {
    const NDArray& v = plan.view;
    const int k = static_cast<int>(plan.indices.size());
    std::vector<std::int64_t> shape(v.shape.begin(), v.shape.begin() + plan.first);
    shape.insert(shape.end(), b.begin(), b.end());
    shape.insert(shape.end(), v.shape.begin() + plan.first + k, v.shape.begin() + v.ndim);
    return shape;
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

// Copies one block per index point: src block at src_off[b] -> dst block at
// dst_off[b], converting dtype. In order, so with repeated indices in a
// scatter the last write wins.
void copy_blocks(const NDArray& src_block, std::span<const std::int64_t> src_off, const NDArray& dst_block,
                 std::span<const std::int64_t> dst_off) {
    const Loop loop = cast_kernel(src_block.dtype, dst_block.dtype);
    const std::int64_t zero[] = {0, 0};
    for (size_t b = 0; b < src_off.size(); ++b) {
        std::byte* s = src_block.data + src_off[b];
        std::byte* d = dst_block.data + dst_off[b];
        if (src_block.ndim == 0) {  // one element per index point: no iterator needed
            std::byte* args[] = {s, d};
            loop(args, 1, zero, nullptr);
            continue;
        }
        const IterOperand ops[] = {{s, src_block.shape_span(), src_block.strides_span()},
                                   {d, dst_block.shape_span(), dst_block.strides_span()}};
        NdIter(dst_block.shape_span(), ops).run(loop);
    }
}

nb::object gather(const Plan& plan) {
    const Offsets off = index_offsets(plan, nullptr);
    const int k = static_cast<int>(plan.indices.size());
    const int bd = static_cast<int>(off.shape.size());
    NDArray out = new_array(plan.view.dtype, result_shape(plan, off.shape));
    // The B axes of the fresh C-order result are contiguous together, so
    // index point b starts b * (stride of the last B axis) bytes in.
    std::vector<std::int64_t> dst_off(off.view.size());
    const std::int64_t step = out.strides[plan.first + bd - 1];
    for (size_t b = 0; b < dst_off.size(); ++b) dst_off[b] = static_cast<std::int64_t>(b) * step;
    copy_blocks(block_of(plan.view, plan.first, k), off.view, block_of(out, plan.first, bd), dst_off);
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

nb::object getitem(nb::handle x, nb::handle key) {
    Plan plan = make_plan(x, key);
    if (!plan.indices.empty()) return gather(plan);  // advanced: always a copy
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
    const std::vector<std::int64_t> shape = result_shape(plan, index_shape(plan));
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
    const Offsets offs = index_offsets(plan, w.strides.data());
    const int k = static_cast<int>(plan.indices.size());
    const int bd = static_cast<int>(offs.shape.size());
    copy_blocks(block_of(w, plan.first, bd), offs.other, block_of(plan.view, plan.first, k), offs.view);
}

}  // namespace

void setitem(nb::handle x, nb::handle key, nb::handle value) {
    if (!as_ndarray(x).writeable) throw nb::value_error("assignment destination is read-only");
    Plan plan = make_plan(x, key);
    if (!plan.indices.empty()) {
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
