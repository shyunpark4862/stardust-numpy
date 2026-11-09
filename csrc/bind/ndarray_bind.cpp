// The Python `ndarray` type: attributes, tolist, repr hooks, and the buffer
// protocol export that lets numpy (or memoryview) read our memory directly.

#include "bind/ndarray_bind.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "ops/copy.hpp"

#include <cstring>

namespace sd {

namespace {

static_assert(sizeof(Py_ssize_t) == sizeof(std::int64_t), "shape/strides are handed out as Py_ssize_t*");

// PEP 3118 format codes of our dtypes ('q' = int64, 'Zd' = complex double).
const char* const kFormats[kNumDTypes] = {"?", "q", "d", "Zd"};

int getbuffer(PyObject* self, Py_buffer* view, int flags) {
    const NDArray& a = *nb::inst_ptr<NDArray>(self);
    auto fail = [&](const char* msg) {
        PyErr_SetString(PyExc_BufferError, msg);
        view->obj = nullptr;
        return -1;
    };
    if ((flags & PyBUF_WRITABLE) && !a.writeable) return fail("array is read-only");
    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !a.c_contiguous) return fail("array is not C-contiguous");
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !a.f_contiguous) return fail("array is not F-contiguous");
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS && !a.c_contiguous && !a.f_contiguous)
        return fail("array is not contiguous");
    const bool with_strides = (flags & PyBUF_STRIDES) == PyBUF_STRIDES;
    if (!with_strides && !a.c_contiguous) return fail("array is not C-contiguous; request strides");

    view->buf = a.data;
    view->obj = Py_NewRef(self);  // keeps the array (and so its Buffer) alive while exported
    view->len = a.nbytes();
    view->readonly = !a.writeable;
    view->itemsize = a.itemsize();
    view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(kFormats[static_cast<int>(a.dtype)]) : nullptr;
    view->ndim = a.ndim;
    view->shape = (flags & PyBUF_ND) == PyBUF_ND
                      ? reinterpret_cast<Py_ssize_t*>(const_cast<std::int64_t*>(a.shape.data()))
                      : nullptr;
    view->strides = with_strides ? reinterpret_cast<Py_ssize_t*>(const_cast<std::int64_t*>(a.strides.data())) : nullptr;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    return 0;
}

PyType_Slot ndarray_slots[] = {{Py_bf_getbuffer, reinterpret_cast<void*>(getbuffer)}, {0, nullptr}};

template <typename T>
T load(const std::byte* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

nb::object element(const NDArray& a, const std::byte* p) {
    return visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) { return to_python(load<T>(p)); });
}

nb::object tolist(const NDArray& a, int depth, const std::byte* p) {
    if (depth == a.ndim) return element(a, p);
    nb::list out;
    for (std::int64_t i = 0; i < a.shape[depth]; ++i) out.append(tolist(a, depth + 1, p + i * a.strides[depth]));
    return out;
}

// Like tolist, but along axes longer than 2*edge keeps only the first and
// last `edge` items with an Ellipsis between them. Lets repr show a huge
// array without materialising it as Python objects.
nb::object summary(const NDArray& a, int depth, const std::byte* p, std::int64_t edge) {
    if (depth == a.ndim) return element(a, p);
    nb::list out;
    const std::int64_t n = a.shape[depth];
    auto add = [&](std::int64_t i) { out.append(summary(a, depth + 1, p + i * a.strides[depth], edge)); };
    if (edge > 0 && n > 2 * edge) {
        for (std::int64_t i = 0; i < edge; ++i) add(i);
        out.append(nb::ellipsis());
        for (std::int64_t i = n - edge; i < n; ++i) add(i);
    } else {
        for (std::int64_t i = 0; i < n; ++i) add(i);
    }
    return out;
}

nb::tuple to_tuple(std::span<const std::int64_t> v) {
    nb::list l;
    for (std::int64_t x : v) l.append(x);
    return nb::tuple(l);
}

nb::object format_call(const char* fn, nb::handle self) {
    return nb::module_::import_("stardust_numpy._format").attr(fn)(self);
}

}  // namespace

void bind_ndarray(nb::module_& m) {
    auto cls = nb::class_<NDArray>(m, "ndarray", nb::type_slots(ndarray_slots))
        .def_prop_ro("dtype", [](const NDArray& a) { return nb::borrow(dtype_object(a.dtype)); })
        .def_prop_ro("shape", [](const NDArray& a) { return to_tuple(a.shape_span()); })
        .def_prop_ro("strides", [](const NDArray& a) { return to_tuple(a.strides_span()); })
        .def_prop_ro("ndim", [](const NDArray& a) { return a.ndim; })
        .def_prop_ro("size", [](const NDArray& a) { return a.size(); })
        .def_prop_ro("itemsize", [](const NDArray& a) { return a.itemsize(); })
        .def_prop_ro("nbytes", [](const NDArray& a) { return a.nbytes(); })
        .def_prop_ro("base", [](const NDArray& a) { return a.base.is_valid() ? a.base : nb::none(); })
        .def("tolist", [](const NDArray& a) { return tolist(a, 0, a.data); })
        .def("astype", [](nb::handle self, nb::handle dtype, nb::handle copy) { return astype(self, dtype, copy); },
             nb::arg(), nb::kw_only(), nb::arg("copy").none() = true)
        .def("copy", [](const NDArray& a) { return wrap(cast_copy(a, a.dtype)); })
        .def("_summary", [](const NDArray& a, std::int64_t edge) { return summary(a, 0, a.data, edge); })
        .def("_flags", [](const NDArray& a) {
            nb::dict d;
            d["c_contiguous"] = a.c_contiguous;
            d["f_contiguous"] = a.f_contiguous;
            d["writeable"] = a.writeable;
            return d;
        })
        .def("__repr__", [](nb::handle self) { return format_call("array_repr", self); })
        .def("__str__", [](nb::handle self) { return format_call("array_str", self); })
        // Q4: an array is never a scalar, whatever its size. All four hooks
        // must exist: without __int__/__float__, int()/float() fall back to
        // the buffer protocol and try to parse the raw bytes as text.
        .def("__bool__", [](const NDArray&) -> bool {
            throw nb::type_error("the truth value of an array is ambiguous; use any() or all()");
        })
        .def("__int__", [](const NDArray&) -> int { throw nb::type_error("an array cannot be converted to int"); })
        .def("__float__", [](const NDArray&) -> double { throw nb::type_error("an array cannot be converted to float"); })
        .def("__complex__", [](const NDArray&) -> double {
            throw nb::type_error("an array cannot be converted to complex");
        })
        .def("__index__", [](const NDArray&) -> int { throw nb::type_error("an array cannot be used as an index"); });
    // Arrays are mutable containers: unhashable, like lists.
    cls.attr("__hash__") = nb::none();
}

}  // namespace sd
