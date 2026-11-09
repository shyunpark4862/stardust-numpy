#include "ops/indexing.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "ops/copy.hpp"
#include "ops/creation.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace sd {

namespace {

enum class Kind { Int, Slice, Ellipsis, NewAxis };

struct Item {
    Kind kind;
    nb::handle obj;
    std::int64_t value = 0;  // Kind::Int only
};

[[noreturn]] void raise(PyObject* type, const std::string& msg) {
    PyErr_SetString(type, msg.c_str());
    throw nb::python_error();
}

Item classify(nb::handle o) {
    PyObject* p = o.ptr();
    if (p == Py_None) return {Kind::NewAxis, o};
    if (p == Py_Ellipsis) return {Kind::Ellipsis, o};
    if (PySlice_Check(p)) return {Kind::Slice, o};
    if (PyUnicode_Check(p) || PyBytes_Check(p)) {
        // Strings are sequences, but never indices.
    } else if (PyBool_Check(p) || is_ndarray(o) || PySequence_Check(p)) {
        // numpy reads a bool as a 0-d mask and a sequence as an integer array.
        raise(PyExc_NotImplementedError, "advanced indexing (integer arrays, boolean masks) arrives in M6");
    } else if (PyIndex_Check(p)) {  // int, or anything with __index__ (e.g. numpy.int64)
        nb::object i = nb::steal(PyNumber_Index(p));
        if (!i.is_valid()) throw nb::python_error();
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
        if (overflow != 0) raise(PyExc_IndexError, "index is out of bounds");
        return {Kind::Int, o, v};
    }
    raise(PyExc_IndexError,
          "only integers, slices (`:`), ellipsis (`...`), newaxis (`None`) and integer or boolean arrays are "
          "valid indices");
}

// The view x[key]. Its ndim may be 0 (every axis indexed by an int): that
// internal 0-d view never escapes to Python.
NDArray basic_view(nb::handle x, nb::handle key) {
    const NDArray& a = as_ndarray(x);
    std::vector<Item> items;
    if (PyTuple_Check(key.ptr())) {
        for (nb::handle h : nb::borrow<nb::tuple>(key)) items.push_back(classify(h));
    } else {
        items.push_back(classify(key));
    }

    int consumed = 0, ellipses = 0;
    for (const Item& it : items) {
        if (it.kind == Kind::Int || it.kind == Kind::Slice) ++consumed;
        if (it.kind == Kind::Ellipsis) ++ellipses;
    }
    if (ellipses > 1) raise(PyExc_IndexError, "an index can only have a single ellipsis ('...')");
    if (consumed > a.ndim)
        raise(PyExc_IndexError, "too many indices for array: array is " + std::to_string(a.ndim) +
                                    "-dimensional, but " + std::to_string(consumed) + " were indexed");

    NDArray v = view_of(x);
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
        }
    }
    while (src < a.ndim) keep_axis();  // unindexed trailing axes are kept whole
    v.data = data;
    v.update_flags();
    return v;
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
    NDArray v = basic_view(x, key);
    if (v.ndim == 0) return load_python(v.dtype, v.data);
    return wrap(std::move(v));
}

void setitem(nb::handle x, nb::handle key, nb::handle value) {
    if (!as_ndarray(x).writeable) throw nb::value_error("assignment destination is read-only");
    NDArray dst = basic_view(x, key);
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
