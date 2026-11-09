// The Python `ndarray` type: attributes, tolist, repr hooks, and the buffer
// protocol export that lets numpy (or memoryview) read our memory directly.

#include "bind/ndarray_bind.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "ops/copy.hpp"
#include "ops/indexing.hpp"
#include "ops/matmul.hpp"
#include "ops/reduction.hpp"
#include "ops/shape.hpp"
#include "ops/ufunc.hpp"

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

nb::object tolist(const NDArray& a, int depth, const std::byte* p) {
    if (depth == a.ndim) return load_python(a.dtype, p);
    nb::list out;
    for (std::int64_t i = 0; i < a.shape[depth]; ++i) out.append(tolist(a, depth + 1, p + i * a.strides[depth]));
    return out;
}

// Like tolist, but along axes longer than 2*edge keeps only the first and
// last `edge` items with an Ellipsis between them. Lets repr show a huge
// array without materialising it as Python objects.
nb::object summary(const NDArray& a, int depth, const std::byte* p, std::int64_t edge) {
    if (depth == a.ndim) return load_python(a.dtype, p);
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

// What iter(x) returns: walks axis 0, yielding x[0], x[1], ...
struct ArrayIterator {
    nb::object array;
    std::int64_t next = 0, stop = 0;
};

// ---- operators: thin wrappers over the elementwise functions --------------
// An operand we do not take (a list, a numpy array, None) gives
// NotImplemented, so Python can try the other side and then raise TypeError.

nb::object not_implemented() { return nb::borrow(Py_NotImplemented); }

template <typename Op>
nb::object op_forward(nb::handle self, nb::handle other) {
    return is_operand(other) ? binary<Op>(self, other, nb::none()) : not_implemented();
}

template <typename Op>
nb::object op_reflected(nb::handle self, nb::handle other) {  // other <op> self
    return is_operand(other) ? binary<Op>(other, self, nb::none()) : not_implemented();
}

// x op= y is op(x, y, out=x): same rules as out= (S6, overlap-safe), same object back.
template <typename Op>
nb::object op_inplace(nb::handle self, nb::handle other) {
    return is_operand(other) ? binary<Op>(self, other, self) : not_implemented();
}

template <typename Op>
nb::object op_unary(nb::handle self) {
    return unary<Op>(self, nb::none());
}

template <typename Op>
void def_arith(nb::class_<NDArray>& cls, const char* fwd, const char* refl, const char* inplace) {
    cls.def(fwd, &op_forward<Op>, nb::arg().none());
    cls.def(refl, &op_reflected<Op>, nb::arg().none());
    cls.def(inplace, &op_inplace<Op>, nb::arg().none());
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
        .def_prop_ro("T", &transpose_2d)
        .def_prop_ro("mT", &matrix_transpose)
        .def("tolist", [](const NDArray& a) { return tolist(a, 0, a.data); })
        .def("astype", [](nb::handle self, nb::handle dtype, nb::handle copy) { return astype(self, dtype, copy); },
             nb::arg(), nb::kw_only(), nb::arg("copy").none() = true)
        .def("copy", [](const NDArray& a) { return wrap(cast_copy(a, a.dtype)); })
        .def("reshape", nb::overload_cast<nb::handle, nb::handle, nb::handle>(&reshape), nb::arg(), nb::kw_only(),
             nb::arg("copy").none() = nb::none())
        // .none(): x[None] is newaxis, and nanobind rejects None by default.
        // Reductions as methods (A2): the same functions, x as self.
        .def("sum", &sum, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("dtype").none() = nb::none(),
             nb::arg("keepdims") = false)
        .def("prod", &prod, nb::kw_only(), nb::arg("axis").none() = nb::none(),
             nb::arg("dtype").none() = nb::none(), nb::arg("keepdims") = false)
        .def("min", &min, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("max", &max, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("mean", &mean, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("var", &var, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("correction") = 0,
             nb::arg("keepdims") = false)
        .def("std", &stddev, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("correction") = 0,
             nb::arg("keepdims") = false)
        .def("argmin", &argmin, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("argmax", &argmax, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("any", &any, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("all", &all, nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims") = false)
        .def("__getitem__", &getitem, nb::arg().none())
        .def("__setitem__", &setitem, nb::arg().none(), nb::arg().none())
        .def("__len__", [](const NDArray& a) { return a.shape[0]; })
        .def("__iter__", [](nb::handle self) { return ArrayIterator{nb::borrow(self), 0, as_ndarray(self).shape[0]}; })
        // Without this, `v in x` would quietly fall back to iterating x.
        .def("__contains__", [](const NDArray&, nb::handle) -> bool {
            throw nb::type_error("`in` is not supported for arrays; compare with == and use any()");
        })
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
    using namespace ops;
    def_arith<Add>(cls, "__add__", "__radd__", "__iadd__");
    def_arith<Subtract>(cls, "__sub__", "__rsub__", "__isub__");
    def_arith<Multiply>(cls, "__mul__", "__rmul__", "__imul__");
    def_arith<Divide>(cls, "__truediv__", "__rtruediv__", "__itruediv__");
    def_arith<FloorDivide>(cls, "__floordiv__", "__rfloordiv__", "__ifloordiv__");
    def_arith<Remainder>(cls, "__mod__", "__rmod__", "__imod__");
    def_arith<Pow>(cls, "__pow__", "__rpow__", "__ipow__");
    def_arith<BitwiseAnd>(cls, "__and__", "__rand__", "__iand__");
    def_arith<BitwiseOr>(cls, "__or__", "__ror__", "__ior__");
    def_arith<BitwiseXor>(cls, "__xor__", "__rxor__", "__ixor__");
    def_arith<LeftShift>(cls, "__lshift__", "__rlshift__", "__ilshift__");
    def_arith<RightShift>(cls, "__rshift__", "__rrshift__", "__irshift__");
    // Python reflects comparisons itself (x > y becomes y < x).
    cls.def("__eq__", &op_forward<Equal>, nb::arg().none());
    cls.def("__ne__", &op_forward<NotEqual>, nb::arg().none());
    cls.def("__lt__", &op_forward<Less>, nb::arg().none());
    cls.def("__le__", &op_forward<LessEqual>, nb::arg().none());
    cls.def("__gt__", &op_forward<Greater>, nb::arg().none());
    cls.def("__ge__", &op_forward<GreaterEqual>, nb::arg().none());
    // @ takes arrays only (no Python scalars). x @= y needs the result to
    // have x's shape, and S6 like every in-place operator.
    cls.def("__matmul__", [](nb::handle self, nb::handle other) {
        return is_ndarray(other) ? matmul(self, other) : not_implemented();
    }, nb::arg().none());
    cls.def("__rmatmul__", [](nb::handle self, nb::handle other) {
        return is_ndarray(other) ? matmul(other, self) : not_implemented();
    }, nb::arg().none());
    cls.def("__imatmul__", [](nb::handle self, nb::handle other) -> nb::object {
        if (!is_ndarray(other)) return not_implemented();
        nb::object r = matmul(self, other);
        NDArray& x = as_ndarray(self);
        if (!is_ndarray(r) || !std::ranges::equal(as_ndarray(r).shape_span(), x.shape_span()))
            throw nb::value_error("x @= y needs a result with the shape of x");
        assign_checked(x, as_ndarray(r));
        return nb::borrow(self);
    }, nb::arg().none());
    cls.def("__neg__", &op_unary<Negative>);
    cls.def("__pos__", &op_unary<Positive>);
    cls.def("__abs__", &op_unary<Abs>);
    cls.def("__invert__", &op_unary<BitwiseInvert>);

    // Arrays are mutable containers: unhashable, like lists. (Defining
    // __eq__ would also do this; being explicit documents it.)
    cls.attr("__hash__") = nb::none();
    // numpy, seeing `np_array + sdnp_array`, would convert our array through
    // the buffer protocol and return a numpy result. None tells numpy's
    // ufuncs to step aside, so mixing the two raises TypeError (M4 decision).
    cls.attr("__array_ufunc__") = nb::none();

    nb::class_<ArrayIterator>(m, "_ArrayIterator")
        .def("__iter__", [](nb::handle self) { return nb::borrow(self); })
        .def("__next__", [](ArrayIterator& it) {
            if (it.next >= it.stop) throw nb::stop_iteration();
            return item_along_axis0(it.array, it.next++);
        });
}

}  // namespace sd
