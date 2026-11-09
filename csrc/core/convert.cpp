#include "core/convert.hpp"

#include <stdexcept>
#include <string>

namespace sd {

void check_castable(DType from, DType to) {
    if (from == DType::Complex128 && (to == DType::Int64 || to == DType::Float64)) {
        std::string msg = std::string("cannot convert complex128 to ") + dtype_name(to) +
                          ": the imaginary part would be lost (take the real part explicitly)";
        throw nb::type_error(msg.c_str());
    }
}

std::optional<DType> scalar_dtype(nb::handle o) {
    PyObject* p = o.ptr();
    if (PyBool_Check(p)) return DType::Bool;  // before PyLong: bool is an int subclass
    if (PyLong_Check(p)) return DType::Int64;
    if (PyFloat_Check(p)) return DType::Float64;
    if (PyComplex_Check(p)) return DType::Complex128;
    return std::nullopt;
}

namespace {

std::int64_t py_to_int64(PyObject* p) {
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(p, &overflow);
    if (overflow != 0) throw std::overflow_error("Python int too large to convert to int64");
    if (v == -1 && PyErr_Occurred()) throw nb::python_error();
    return v;
}

double py_to_double(PyObject* p) {
    double v = PyFloat_AsDouble(p);  // works for int too; raises OverflowError for huge ints
    if (v == -1.0 && PyErr_Occurred()) throw nb::python_error();
    return v;
}

}  // namespace

template <typename T>
T scalar_from_python(nb::handle o, DType src) {
    PyObject* p = o.ptr();
    if constexpr (std::is_same_v<T, bool>) {
        int truth = PyObject_IsTrue(p);  // x != 0 for every numeric type (nan -> true)
        if (truth < 0) throw nb::python_error();
        return truth != 0;
    } else {
        switch (src) {
            case DType::Bool: return cast_value<T>(p == Py_True);
            case DType::Int64: {
                if constexpr (std::is_same_v<T, std::int64_t>) return py_to_int64(p);
                else return cast_value<T>(py_to_double(p));
            }
            case DType::Float64: return cast_value<T>(py_to_double(p));
            case DType::Complex128: {
                if constexpr (is_complex_v<T>) {
                    Py_complex c = PyComplex_AsCComplex(p);
                    if (c.real == -1.0 && PyErr_Occurred()) throw nb::python_error();
                    return T(c.real, c.imag);
                } else {
                    check_castable(DType::Complex128, dtype_of<T>);  // always throws
                    __builtin_unreachable();
                }
            }
        }
        __builtin_unreachable();
    }
}

template bool scalar_from_python<bool>(nb::handle, DType);
template std::int64_t scalar_from_python<std::int64_t>(nb::handle, DType);
template double scalar_from_python<double>(nb::handle, DType);
template complex128 scalar_from_python<complex128>(nb::handle, DType);

nb::object to_python(bool v) { return nb::bool_(v); }
nb::object to_python(std::int64_t v) { return nb::steal(PyLong_FromLongLong(v)); }
nb::object to_python(double v) { return nb::steal(PyFloat_FromDouble(v)); }
nb::object to_python(complex128 v) { return nb::steal(PyComplex_FromDoubles(v.real(), v.imag())); }

}  // namespace sd
