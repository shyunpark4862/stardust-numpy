#include "bind/pyutil.hpp"

#include <string>

namespace sd {

namespace {

// Borrowed: the module attributes (sdnp.bool, ...) own the four objects, and
// nothing in _core can run after the module itself is torn down.
PyObject* g_dtype_objects[kNumDTypes] = {};

}  // namespace

void register_dtypes(nb::module_& m) {
    nb::class_<PyDType>(m, "DType")
        .def("__repr__", [](const PyDType& d) { return dtype_name(d.value); })
        .def("__str__", [](const PyDType& d) { return dtype_name(d.value); });

    for (DType d : {DType::Bool, DType::Int64, DType::Float64, DType::Complex128}) {
        nb::object obj = nb::cast(PyDType{d});
        m.attr(dtype_name(d)) = obj;
        g_dtype_objects[static_cast<int>(d)] = obj.ptr();
    }
}

nb::handle dtype_object(DType d) { return g_dtype_objects[static_cast<int>(d)]; }

std::optional<DType> parse_dtype(nb::handle o) {
    if (o.is_none()) return std::nullopt;
    for (int i = 0; i < kNumDTypes; ++i)
        if (o.ptr() == g_dtype_objects[i]) return static_cast<DType>(i);
    throw nb::type_error("dtype must be one of sdnp.bool, sdnp.int64, sdnp.float64, sdnp.complex128");
}

std::optional<bool> parse_copy(nb::handle o) {
    if (o.is_none()) return std::nullopt;
    if (o.ptr() == Py_True) return true;
    if (o.ptr() == Py_False) return false;
    throw nb::type_error("copy must be True, False or None");
}

std::int64_t parse_int(nb::handle o, const char* what) {
    if (!PyLong_Check(o.ptr()) || PyBool_Check(o.ptr()))
        throw nb::type_error((std::string(what) + " must be an int").c_str());
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(o.ptr(), &overflow);
    if (overflow != 0) throw nb::value_error((std::string(what) + " is too large").c_str());
    return v;
}

std::vector<std::int64_t> parse_shape(nb::handle o) {
    if (PyLong_Check(o.ptr()) && !PyBool_Check(o.ptr())) return {parse_int(o, "shape")};
    if (!PyTuple_Check(o.ptr())) throw nb::type_error("shape must be an int or a tuple of ints");
    std::vector<std::int64_t> shape;
    for (nb::handle item : nb::borrow<nb::tuple>(o)) {
        if (!PyLong_Check(item.ptr()) || PyBool_Check(item.ptr()))
            throw nb::type_error("shape must be an int or a tuple of ints");
        shape.push_back(parse_int(item, "shape entry"));
    }
    return shape;
}

}  // namespace sd
