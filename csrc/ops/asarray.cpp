// asarray: the single entry point that turns Python data into an ndarray.
//
// Three sources, in this order:
//   1. an ndarray            -> returned as is, or converted/copied
//   2. nested lists / tuples -> shape discovered, dtype inferred, values copied
//   3. the buffer protocol   -> zero-copy when the element type matches exactly
//                               (e.g. a numpy float64 array), else a converting copy

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "ops/copy.hpp"
#include "ops/creation.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace sd {

namespace {

const char* kNoZeroDim =
    "0-dimensional arrays are not supported: a scalar is already the result (stardust-numpy has no 0-d arrays)";

template <typename T>
T load(const std::byte* p) {
    if constexpr (std::is_same_v<T, bool>) return load<std::uint8_t>(p) != 0;  // any nonzero '?' byte is true
    T v;
    std::memcpy(&v, p, sizeof(T));  // memcpy: imported buffers need not be aligned
    return v;
}

template <typename T>
void store(std::byte* p, T v) {
    std::memcpy(p, &v, sizeof(T));
}

}  // namespace

// ------------------------------------------------------- nested sequences --

namespace {

bool is_seq(PyObject* p) { return PyList_Check(p) || PyTuple_Check(p); }
Py_ssize_t seq_len(PyObject* p) { return PyList_Check(p) ? PyList_GET_SIZE(p) : PyTuple_GET_SIZE(p); }
PyObject* seq_item(PyObject* p, Py_ssize_t i) { return PyList_Check(p) ? PyList_GET_ITEM(p, i) : PyTuple_GET_ITEM(p, i); }

[[noreturn]] void ragged(const std::vector<std::int64_t>& shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    s += shape.size() == 1 ? ",)" : ")";
    throw nb::value_error(("inhomogeneous nested sequence: expected every level to match shape " + s).c_str());
}

// Pass 1a: read the shape off the first element at every depth.
std::vector<std::int64_t> discover_shape(PyObject* o) {
    std::vector<std::int64_t> shape;
    for (;;) {
        if (is_ndarray(o)) {
            const NDArray& a = as_ndarray(o);
            shape.insert(shape.end(), a.shape.begin(), a.shape.begin() + a.ndim);
            break;
        }
        if (!is_seq(o)) break;
        const Py_ssize_t n = seq_len(o);
        shape.push_back(n);
        if (n == 0) break;
        o = seq_item(o, 0);
    }
    if (shape.size() > static_cast<size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    return shape;
}

// Pass 1b: check every element against that shape and promote the leaf kinds.
void validate(PyObject* o, size_t depth, const std::vector<std::int64_t>& shape, std::optional<DType>& kind) {
    auto merge = [&](DType d) { kind = kind ? promote(*kind, d) : d; };
    if (depth == shape.size()) {
        if (auto d = scalar_dtype(o)) return merge(*d);
        if (is_seq(o) || is_ndarray(o)) ragged(shape);
        throw nb::type_error(("cannot store an object of type '" + std::string(Py_TYPE(o)->tp_name) +
                              "' in an array (only bool, int, float and complex)").c_str());
    }
    if (is_ndarray(o)) {
        const NDArray& a = as_ndarray(o);
        if (static_cast<size_t>(a.ndim) != shape.size() - depth ||
            !std::equal(a.shape.begin(), a.shape.begin() + a.ndim, shape.begin() + depth))
            ragged(shape);
        return merge(a.dtype);
    }
    if (!is_seq(o) || seq_len(o) != shape[depth]) ragged(shape);
    for (Py_ssize_t i = 0; i < shape[depth]; ++i) validate(seq_item(o, i), depth + 1, shape, kind);
}

// Pass 2: write the values in row-major order.
template <typename T>
void fill_nested(PyObject* o, size_t depth, size_t ndim, T*& out) {
    if (depth == ndim) {
        *out++ = scalar_from_python<T>(o, *scalar_dtype(o));
        return;
    }
    if (is_ndarray(o)) {
        const NDArray& a = as_ndarray(o);
        visit_dtype(a.dtype, [&]<typename S>(std::type_identity<S>) {
            for_each_element<1>(a.shape_span(), {a.data}, {a.strides.data()},
                                [&](std::byte* p) { *out++ = cast_value<T>(load<S>(p)); });
        });
        return;
    }
    for (Py_ssize_t i = 0, n = seq_len(o); i < n; ++i) fill_nested(seq_item(o, i), depth + 1, ndim, out);
}

nb::object from_nested(nb::handle obj, std::optional<DType> dtype) {
    const std::vector<std::int64_t> shape = discover_shape(obj.ptr());
    std::optional<DType> kind;
    validate(obj.ptr(), 0, shape, kind);
    const DType target = dtype.value_or(kind.value_or(DType::Float64));  // [] -> float64
    if (kind) check_castable(*kind, target);
    NDArray a = new_array(target, shape);
    visit_dtype(target, [&]<typename T>(std::type_identity<T>) {
        T* out = reinterpret_cast<T*>(a.data);
        fill_nested<T>(obj.ptr(), 0, shape.size(), out);
    });
    return wrap(std::move(a));
}

// --------------------------------------------------------- buffer protocol --

// Element types we can read from a foreign buffer (ARCHITECTURE.md §6.5).
enum class Src { Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128 };

Src parse_format(const char* format, Py_ssize_t itemsize) {
    const std::string fmt = format ? format : "B";  // NULL format means unsigned bytes
    size_t i = 0;
    if (!fmt.empty() && (fmt[0] == '>' || fmt[0] == '!'))
        throw nb::type_error("buffers with non-native (big-endian) byte order are not supported");
    if (!fmt.empty() && (fmt[0] == '@' || fmt[0] == '=' || fmt[0] == '<')) i = 1;
    const std::string code = fmt.substr(i);
    auto unsupported = [&]() -> Src {
        throw nb::type_error(("unsupported buffer format '" + fmt + "'").c_str());
    };
    if (code == "?") return Src::Bool;
    if (code.size() == 1 && std::strchr("bhilqn", code[0])) {
        switch (itemsize) {
            case 1: return Src::I8;
            case 2: return Src::I16;
            case 4: return Src::I32;
            case 8: return Src::I64;
        }
    }
    if (code.size() == 1 && std::strchr("BHILQN", code[0])) {
        switch (itemsize) {
            case 1: return Src::U8;
            case 2: return Src::U16;
            case 4: return Src::U32;
            case 8: return Src::U64;
        }
    }
    if (code.size() == 1 && std::strchr("efd", code[0])) {
        switch (itemsize) {
            case 2: return Src::F16;
            case 4: return Src::F32;
            case 8: return Src::F64;
        }
    }
    if (code == "Zf" && itemsize == 8) return Src::C64;
    if (code == "Zd" && itemsize == 16) return Src::C128;
    return unsupported();
}

// The dtype a foreign element type widens to without losing information (R5).
DType natural_dtype(Src s) {
    switch (s) {
        case Src::Bool: return DType::Bool;
        case Src::I8: case Src::I16: case Src::I32: case Src::I64:
        case Src::U8: case Src::U16: case Src::U32: return DType::Int64;
        case Src::U64: throw nb::type_error("uint64 buffers cannot be converted to int64 without loss");
        case Src::F16: case Src::F32: case Src::F64: return DType::Float64;
        case Src::C64: case Src::C128: return DType::Complex128;
    }
    __builtin_unreachable();
}

bool is_storage_of(Src s, DType d) {
    return (s == Src::Bool && d == DType::Bool) || (s == Src::I64 && d == DType::Int64) ||
           (s == Src::F64 && d == DType::Float64) || (s == Src::C128 && d == DType::Complex128);
}

template <typename F>
void visit_src(Src s, F&& f) {
    switch (s) {
        case Src::Bool: return f(std::type_identity<bool>{});
        case Src::I8: return f(std::type_identity<std::int8_t>{});
        case Src::I16: return f(std::type_identity<std::int16_t>{});
        case Src::I32: return f(std::type_identity<std::int32_t>{});
        case Src::I64: return f(std::type_identity<std::int64_t>{});
        case Src::U8: return f(std::type_identity<std::uint8_t>{});
        case Src::U16: return f(std::type_identity<std::uint16_t>{});
        case Src::U32: return f(std::type_identity<std::uint32_t>{});
        case Src::F16: return f(std::type_identity<__fp16>{});
        case Src::F32: return f(std::type_identity<float>{});
        case Src::F64: return f(std::type_identity<double>{});
        case Src::C64: return f(std::type_identity<std::complex<float>>{});
        case Src::C128: return f(std::type_identity<std::complex<double>>{});
        case Src::U64: break;  // rejected by natural_dtype
    }
    __builtin_unreachable();
}

// __fp16 is a storage-only type; do arithmetic conversions from float.
template <typename S>
auto widen(S v) {
    if constexpr (std::is_same_v<S, __fp16>) return static_cast<float>(v);
    else return v;
}

struct ViewGuard {
    Py_buffer* view;
    bool armed = true;
    ~ViewGuard() {
        if (armed) PyBuffer_Release(view);
    }
};

nb::object from_buffer(nb::handle obj, std::optional<DType> dtype, std::optional<bool> copy) {
    Py_buffer view;
    if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_RECORDS_RO) != 0) throw nb::python_error();
    ViewGuard guard{&view};

    const Src src = parse_format(view.format, view.itemsize);
    const DType natural = natural_dtype(src);
    if (view.ndim == 0) throw nb::value_error(kNoZeroDim);
    if (view.ndim > kMaxNdim)
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    const DType target = dtype.value_or(natural);
    check_castable(natural, target);

    // Describe the foreign memory with our own shape/strides.
    NDArray foreign;
    foreign.dtype = target;
    foreign.ndim = view.ndim;
    std::int64_t stride = view.itemsize;
    for (int d = view.ndim - 1; d >= 0; --d) {
        foreign.shape[d] = view.shape[d];
        foreign.strides[d] = view.strides ? view.strides[d] : stride;
        stride *= view.shape[d];
    }

    // Our kernels assume naturally aligned elements (alignof(T)); a foreign
    // buffer that is not aligned is copied instead of shared.
    const auto align = static_cast<std::uintptr_t>(itemsize(target) == 16 ? 8 : itemsize(target));
    bool aligned = reinterpret_cast<std::uintptr_t>(view.buf) % align == 0;
    for (int d = 0; d < view.ndim; ++d) aligned = aligned && foreign.strides[d] % static_cast<std::int64_t>(align) == 0;

    // Our bool kernels add the bytes to count trues, so a shared bool buffer
    // must hold only 0 and 1; any other byte makes it a normalizing copy.
    // The OR of all bytes is at most 1 exactly then (a branch-free scan).
    bool valid = true;
    if (target == DType::Bool && is_storage_of(src, target) && aligned && copy != true) {
        std::uint8_t any = 0;
        foreign.update_flags();
        if (foreign.c_contiguous) {
            const auto* q = static_cast<const std::uint8_t*>(view.buf);
            for (std::int64_t i = 0, n = foreign.size(); i < n; ++i) any |= q[i];
        } else {
            for_each_element<1>(foreign.shape_span(), {static_cast<std::byte*>(view.buf)}, {foreign.strides.data()},
                                [&](std::byte* p) { any |= *reinterpret_cast<const std::uint8_t*>(p); });
        }
        valid = any <= 1;
    }

    if (is_storage_of(src, target) && aligned && valid && copy != true) {
        foreign.data = static_cast<std::byte*>(view.buf);
        foreign.writeable = !view.readonly;
        foreign.base = nb::borrow(view.obj ? view.obj : obj.ptr());
        foreign.update_flags();
        guard.armed = false;
        foreign.buffer = Buffer::adopt(view);
        return wrap(std::move(foreign));
    }
    if (copy == false)
        throw nb::value_error(valid ? "copy=False, but this buffer can only be converted by copying"
                                    : "copy=False, but this bool buffer holds bytes other than 0 and 1");

    NDArray dst = new_array_like(foreign, target);  // keeps the source's memory order
    visit_src(src, [&]<typename S>(std::type_identity<S>) {
        visit_dtype(target, [&]<typename T>(std::type_identity<T>) {
            for_each_element<2>(dst.shape_span(), {static_cast<std::byte*>(view.buf), dst.data},
                                {foreign.strides.data(), dst.strides.data()},
                                [](std::byte* ps, std::byte* pd) { store(pd, cast_value<T>(widen(load<S>(ps)))); });
        });
    });
    return wrap(std::move(dst));
}

}  // namespace

nb::object asarray(nb::handle obj, nb::handle dtype_obj, nb::handle copy_obj) {
    const std::optional<DType> dtype = parse_dtype(dtype_obj);
    const std::optional<bool> copy = parse_copy(copy_obj);

    if (is_ndarray(obj)) {
        const NDArray& a = as_ndarray(obj);
        const DType target = dtype.value_or(a.dtype);
        if (target == a.dtype && copy != true) return nb::borrow(obj);
        if (copy == false) throw nb::value_error("copy=False, but a dtype conversion requires a copy");
        return wrap(cast_copy(a, target));
    }
    if (scalar_dtype(obj)) throw nb::value_error(kNoZeroDim);
    if (is_seq(obj.ptr())) {
        if (copy == false) throw nb::value_error("copy=False, but a list or tuple can only be converted by copying");
        return from_nested(obj, dtype);
    }
    if (PyObject_CheckBuffer(obj.ptr())) return from_buffer(obj, dtype, copy);
    throw nb::type_error(("cannot convert an object of type '" + std::string(Py_TYPE(obj.ptr())->tp_name) +
                          "' to an array").c_str());
}

}  // namespace sd
