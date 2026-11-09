#include "ops/creation.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace sd {

namespace {

// New arrays are contiguous, so filling them is a flat loop over memory
// regardless of their axis order.
template <typename T>
void fill_flat(NDArray& a, T value) {
    T* p = reinterpret_cast<T*>(a.data);
    std::fill(p, p + a.size(), value);
}

void fill_one(NDArray& a) {
    visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) { fill_flat(a, cast_value<T>(true)); });
}

void fill_scalar(NDArray& a, nb::handle value, DType value_dtype) {
    visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) { fill_flat(a, scalar_from_python<T>(value, value_dtype)); });
}

DType require_scalar(nb::handle o, const char* what) {
    if (auto d = scalar_dtype(o)) return *d;
    throw nb::type_error((std::string(what) + " must be a Python bool, int, float or complex").c_str());
}

const NDArray& require_ndarray(nb::handle o) {
    if (!is_ndarray(o)) throw nb::type_error("x must be an sdnp.ndarray");
    return as_ndarray(o);
}

}  // namespace

nb::object zeros(nb::handle shape, nb::handle dtype) {
    return wrap(new_array(parse_dtype(dtype).value_or(DType::Float64), parse_shape(shape), /*zeroed=*/true));
}

nb::object ones(nb::handle shape, nb::handle dtype) {
    NDArray a = new_array(parse_dtype(dtype).value_or(DType::Float64), parse_shape(shape));
    fill_one(a);
    return wrap(std::move(a));
}

nb::object empty(nb::handle shape, nb::handle dtype) {
    return wrap(new_array(parse_dtype(dtype).value_or(DType::Float64), parse_shape(shape)));
}

nb::object full(nb::handle shape, nb::handle fill_value, nb::handle dtype) {
    const DType kind = require_scalar(fill_value, "fill_value");
    const DType target = parse_dtype(dtype).value_or(kind);
    check_castable(kind, target);
    NDArray a = new_array(target, parse_shape(shape));
    fill_scalar(a, fill_value, kind);
    return wrap(std::move(a));
}

nb::object zeros_like(nb::handle x, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    return wrap(new_array_like(p, parse_dtype(dtype).value_or(p.dtype), /*zeroed=*/true));
}

nb::object ones_like(nb::handle x, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    NDArray a = new_array_like(p, parse_dtype(dtype).value_or(p.dtype));
    fill_one(a);
    return wrap(std::move(a));
}

nb::object empty_like(nb::handle x, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    return wrap(new_array_like(p, parse_dtype(dtype).value_or(p.dtype)));
}

nb::object full_like(nb::handle x, nb::handle fill_value, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    const DType kind = require_scalar(fill_value, "fill_value");
    const DType target = parse_dtype(dtype).value_or(p.dtype);
    check_castable(kind, target);
    NDArray a = new_array_like(p, target);
    fill_scalar(a, fill_value, kind);
    return wrap(std::move(a));
}

// Values are start + i*step, computed in int64 when every argument is an int
// and in float64 otherwise, then converted to `dtype` by the explicit rules.
nb::object arange(nb::handle start_obj, nb::handle stop_obj, nb::handle step_obj, nb::handle dtype_obj) {
    const std::optional<DType> dtype = parse_dtype(dtype_obj);
    nb::object zero = nb::int_(0);
    nb::handle start = stop_obj.is_none() ? zero : start_obj;
    nb::handle stop = stop_obj.is_none() ? start_obj : stop_obj;

    DType kinds[3];
    nb::handle args[3] = {start, stop, step_obj};
    for (int i = 0; i < 3; ++i) {
        kinds[i] = require_scalar(args[i], "arange arguments");
        if (kinds[i] == DType::Complex128) throw nb::type_error("arange arguments must be real numbers");
    }
    const bool integral = std::max({kinds[0], kinds[1], kinds[2]}) <= DType::Int64;

    if (integral) {
        const std::int64_t a = scalar_from_python<std::int64_t>(start, kinds[0]);
        const std::int64_t b = scalar_from_python<std::int64_t>(stop, kinds[1]);
        const std::int64_t s = scalar_from_python<std::int64_t>(step_obj, kinds[2]);
        if (s == 0) throw nb::value_error("arange step must not be zero");
        const __int128 span = static_cast<__int128>(b) - a;
        std::int64_t n = 0;
        if (s > 0 && span > 0) n = static_cast<std::int64_t>((span + s - 1) / s);
        if (s < 0 && span < 0) n = static_cast<std::int64_t>((-span - s - 1) / -static_cast<__int128>(s));
        NDArray out = new_array(dtype.value_or(DType::Int64), std::vector<std::int64_t>{n});
        visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
            T* p = reinterpret_cast<T*>(out.data);
            for (std::int64_t i = 0; i < n; ++i) p[i] = cast_value<T>(static_cast<std::int64_t>(a + static_cast<__int128>(i) * s));
        });
        return wrap(std::move(out));
    }

    const double a = scalar_from_python<double>(start, kinds[0]);
    const double b = scalar_from_python<double>(stop, kinds[1]);
    const double s = scalar_from_python<double>(step_obj, kinds[2]);
    if (s == 0) throw nb::value_error("arange step must not be zero");
    const double len = std::ceil((b - a) / s);
    if (len >= 9.2e18) throw nb::value_error("arange would produce too many elements");
    const std::int64_t n = len > 0 ? static_cast<std::int64_t>(len) : 0;  // also catches nan
    NDArray out = new_array(dtype.value_or(DType::Float64), std::vector<std::int64_t>{n});
    visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
        T* p = reinterpret_cast<T*>(out.data);
        for (std::int64_t i = 0; i < n; ++i) p[i] = cast_value<T>(a + static_cast<double>(i) * s);
    });
    return wrap(std::move(out));
}

// Same arithmetic as numpy (y = i*step + start, last element pinned to stop)
// so that results agree bit for bit.
nb::object linspace(nb::handle start, nb::handle stop, nb::handle num_obj, nb::handle dtype_obj, nb::handle endpoint_obj) {
    const DType ks = require_scalar(start, "start");
    const DType ke = require_scalar(stop, "stop");
    const std::int64_t num = parse_int(num_obj, "num");
    if (num < 0) throw nb::value_error("num must be non-negative");
    if (!PyBool_Check(endpoint_obj.ptr())) throw nb::type_error("endpoint must be True or False");
    const bool endpoint = endpoint_obj.ptr() == Py_True;

    const DType inferred = promote(ks, ke) == DType::Complex128 ? DType::Complex128 : DType::Float64;
    const DType target = parse_dtype(dtype_obj).value_or(inferred);
    check_castable(inferred, target);
    NDArray out = new_array(target, std::vector<std::int64_t>{num});

    auto compute = [&]<typename C>(std::type_identity<C>) {
        const C s = scalar_from_python<C>(start, ks);
        const C e = scalar_from_python<C>(stop, ke);
        const C delta = e - s;
        const std::int64_t div = endpoint ? num - 1 : num;
        const C step = div > 0 ? delta / static_cast<double>(div) : C(0);
        visit_dtype(target, [&]<typename T>(std::type_identity<T>) {
            T* p = reinterpret_cast<T*>(out.data);
            for (std::int64_t i = 0; i < num; ++i) {
                const double fi = static_cast<double>(i);
                C y;
                if (div <= 0) y = fi * delta + s;
                else if (step == C(0)) y = (fi / static_cast<double>(div)) * delta + s;  // step underflowed
                else y = fi * step + s;
                if (endpoint && num > 1 && i == num - 1) y = e;
                p[i] = cast_value<T>(y);
            }
        });
    };
    if (inferred == DType::Complex128) compute(std::type_identity<complex128>{});
    else compute(std::type_identity<double>{});
    return wrap(std::move(out));
}

nb::object eye(nb::handle n_rows_obj, nb::handle n_cols_obj, nb::handle k_obj, nb::handle dtype_obj) {
    const std::int64_t n = parse_int(n_rows_obj, "n_rows");
    const std::int64_t m = n_cols_obj.is_none() ? n : parse_int(n_cols_obj, "n_cols");
    const std::int64_t k = parse_int(k_obj, "k");
    NDArray out = new_array(parse_dtype(dtype_obj).value_or(DType::Float64), std::vector<std::int64_t>{n, m}, true);
    visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
        T* p = reinterpret_cast<T*>(out.data);
        for (std::int64_t i = std::max<std::int64_t>(0, -k); i < n && i + k < m; ++i) p[i * m + i + k] = cast_value<T>(true);
    });
    return wrap(std::move(out));
}

void bind_creation(nb::module_& m) {
    using nb::arg;
    m.def("asarray", &asarray, arg(), nb::kw_only(), arg("dtype") = nb::none(), arg("copy") = nb::none());
    m.def("zeros", &zeros, arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("ones", &ones, arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("empty", &empty, arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("full", &full, arg(), arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("zeros_like", &zeros_like, arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("ones_like", &ones_like, arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("empty_like", &empty_like, arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("full_like", &full_like, arg(), arg(), nb::kw_only(), arg("dtype") = nb::none());
    m.def("arange", &arange, arg(), arg() = nb::none(), arg() = 1, nb::kw_only(), arg("dtype") = nb::none());
    m.def("linspace", &linspace, arg(), arg(), arg(), nb::kw_only(), arg("dtype") = nb::none(),
          arg("endpoint") = true);
    m.def("eye", &eye, arg(), arg() = nb::none(), nb::kw_only(), arg("k") = 0, arg("dtype") = nb::none());
}

}  // namespace sd
