#include "ops/reduction.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "kernels/buffered.hpp"
#include "kernels/cast.hpp"
#include "kernels/reduce.hpp"
#include "ops/copy.hpp"
#include "ops/lanes.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace sd {

namespace {

// ------------------------------------------------------------------ axes

struct Axes {
    std::array<bool, kMaxNdim> reduced{};
    int count = 0;
    std::int64_t per_output = 1;  // how many inputs fold into each output
};

Axes parse_reduce_axes(const NDArray& x, nb::handle axis) {
    Axes ax;
    if (axis.is_none()) {
        for (int d = 0; d < x.ndim; ++d) ax.reduced[d] = true;
    } else {
        for (int d : parse_axes(axis, x.ndim)) ax.reduced[d] = true;  // () reduces nothing
    }
    for (int d = 0; d < x.ndim; ++d)
        if (ax.reduced[d]) {
            ++ax.count;
            ax.per_output *= x.shape[d];
        }
    return ax;
}

std::int64_t output_size(const NDArray& x, const Axes& ax) {
    std::int64_t n = 1;
    for (int d = 0; d < x.ndim; ++d)
        if (!ax.reduced[d]) n *= x.shape[d];
    return n;
}

// The result with keepdims: x's shape with 1 on reduced axes, laid out in x's
// axis order so the iterator still walks x in memory order.
NDArray keepdims_result(const NDArray& x, const Axes& ax, DType dtype, bool zeroed = false) {
    NDArray proto = x;
    for (int d = 0; d < x.ndim; ++d)
        if (ax.reduced[d]) proto.shape[d] = 1;
    return new_array_like(proto, dtype, zeroed);
}

// keepdims=False: drop the reduced axes (a view of the fresh result), or hand
// back a Python scalar when nothing is left (Z1).
nb::object finish(NDArray&& res, const Axes& ax, bool keepdims) {
    if (keepdims) return wrap(std::move(res));
    if (ax.count == res.ndim) return load_python(res.dtype, res.data);
    int n = 0;
    for (int d = 0; d < res.ndim; ++d) {
        if (ax.reduced[d]) continue;
        res.shape[n] = res.shape[d];
        res.strides[n] = res.strides[d];
        ++n;
    }
    res.ndim = n;
    res.update_flags();
    return wrap(std::move(res));
}

template <typename T>
void fill(NDArray& a, T v) {
    T* p = reinterpret_cast<T*>(a.data);
    std::fill(p, p + a.size(), v);
}

// x folded into `out` (keepdims shape, pre-filled with the starting value) by
// `loop`, which works in out's dtype; an x of another dtype is converted
// chunk by chunk on the way (buffered casting, M7.6).
void run_reduce(const NDArray& x, NDArray& out, Loop loop) {
    const IterOperand ops[] = {operand(x), operand(out)};
    const NdIter it(x.shape_span(), ops);
    if (x.dtype == out.dtype) {
        it.run(loop, nullptr, Split::Reduction);
        return;
    }
    BufferedLoop buffered;
    buffered.inner = loop;
    buffered.nin = 1;
    buffered.nop = 2;
    buffered.add_cast(0, cast_kernel(x.dtype, out.dtype), out.itemsize());
    it.run(&buffered_loop, &buffered, Split::Reduction);
}

NDArray as_dtype(const NDArray& x, DType d) { return x.dtype == d ? x : cast_copy(x, d); }

[[noreturn]] void unsupported(const char* name, DType d) {
    throw nb::type_error((std::string(name) + "() is not supported for " + dtype_name(d)).c_str());
}

// sum/prod/cumulative dtype: bool accumulates in int64 (C13); dtype= is an
// explicit conversion (M1 rules) but never bool: add and multiply have no
// bool loop (M5 decision).
DType accumulate_dtype(const NDArray& x, nb::handle dtype_obj, const char* name) {
    const std::optional<DType> d = parse_dtype(dtype_obj);
    if (!d) return promote(x.dtype, DType::Int64);
    if (*d == DType::Bool)
        throw nb::type_error((std::string(name) + "() cannot accumulate in bool; use any() or all()").c_str());
    check_castable(x.dtype, *d);
    return *d;
}

// --------------------------------------------------------- sum, prod, any, all

template <typename Op>
nb::object fold(const NDArray& x, const Axes& ax, DType loop_dtype, bool keepdims, auto start) {
    NDArray out = keepdims_result(x, ax, loop_dtype);
    visit_dtype(loop_dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (Op::template ok<T>) {
            fill<T>(out, start(std::type_identity<T>{}));
            run_reduce(x, out, &reduce_loop<Op, T>);
        } else {
            unsupported(Op::name, loop_dtype);
        }
    });
    return finish(std::move(out), ax, keepdims);
}

// min/max have no identity: an output with nothing to fold is an error.
void require_nonempty(const NDArray& x, const Axes& ax, const char* name) {
    if (ax.per_output == 0 && output_size(x, ax) > 0)
        throw nb::value_error((std::string(name) + "() of an empty array (or along an empty axis) is undefined")
                                  .c_str());
}

// ---------------------------------------------------------- argmin / argmax

using ArgFn = std::int64_t (*)(const std::byte*, std::int64_t, std::int64_t);

void arg_lane(const std::byte* x, std::byte* o, const LaneCtx& c) {
    *reinterpret_cast<std::int64_t*>(o) = (*static_cast<const ArgFn*>(c.user))(x, c.n, c.x_stride);
}

// argmin/argmax of one long contiguous lane on several threads: each piece
// finds its own answer, then the pieces are compared in order -- the first
// piece with a NaN wins, otherwise the best value, earlier piece on a tie --
// so the answer is the serial one exactly.
std::int64_t arg_whole(const std::byte* p, std::int64_t n, std::int64_t item, DType dtype, ArgFn lane, bool max) {
    if (n < parallel::kThreshold || !parallel::available()) return lane(p, n, item);
    const std::int64_t pieces = (n + parallel::kTaskWork - 1) / parallel::kTaskWork;
    std::vector<std::int64_t> at(pieces);
    parallel::parallel_for(pieces, [&](std::int64_t t) {
        const std::int64_t begin = t * parallel::kTaskWork;
        at[t] = begin + lane(p + begin * item, std::min(parallel::kTaskWork, n - begin), item);
    });
    return visit_dtype(dtype, [&]<typename T>(std::type_identity<T>) -> std::int64_t {
        if constexpr (ops::is_cplx<T>) {
            return at[0];  // unreachable: complex is rejected before
        } else {
            std::int64_t best = at[0];
            for (std::int64_t t = 0; t < pieces; ++t) {
                const T v = load_as<T>(p + at[t] * item);
                if constexpr (ops::is_float<T>) {
                    if (std::isnan(v)) return at[t];  // the first NaN overall
                }
                if (t > 0 && (max ? v > load_as<T>(p + best * item) : v < load_as<T>(p + best * item))) best = at[t];
            }
            return best;
        }
    });
}

template <bool Max>
nb::object arg_extreme_impl(nb::handle x_obj, nb::handle axis_obj, bool keepdims) {
    const char* name = Max ? "argmax" : "argmin";
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) unsupported(name, x.dtype);  // S5
    // complex was rejected above; its branch only satisfies visit_dtype.
    ArgFn lane_fn = visit_dtype(x.dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_cplx<T>) return &arg_extreme<double, Max>;
        else return &arg_extreme<T, Max>;
    });
    auto empty_error = [&] {
        return nb::value_error((std::string(name) + "() of an empty sequence is undefined").c_str());
    };

    if (axis_obj.is_none()) {
        // Over everything: the index into the flattened (C order) array.
        if (x.size() == 0) throw empty_error();
        NDArray flat = x;
        if (!x.c_contiguous) {
            flat = new_array(x.dtype, x.shape_span());
            assign(flat, x);
        }
        const std::int64_t at = arg_whole(flat.data, x.size(), x.itemsize(), x.dtype, lane_fn, Max);
        if (!keepdims) return to_python(at);
        Axes all;
        for (int d = 0; d < x.ndim; ++d) all.reduced[d] = true;
        NDArray out = keepdims_result(x, all, DType::Int64);
        *reinterpret_cast<std::int64_t*>(out.data) = at;
        return wrap(std::move(out));
    }

    const int axis = normalize_axis(parse_int(axis_obj, "axis"), x.ndim);  // C5: int only
    Axes ax;
    ax.reduced[axis] = true;
    ax.count = 1;
    ax.per_output = x.shape[axis];
    if (ax.per_output == 0 && output_size(x, ax) > 0) throw empty_error();
    NDArray out = keepdims_result(x, ax, DType::Int64);
    LaneCtx ctx{x.shape[axis], x.strides[axis], 0, &arg_lane, &lane_fn};
    if (ctx.n > 0) run_lanes(x, out, axis, ctx);
    return finish(std::move(out), ax, keepdims);
}

// ------------------------------------------------------------ cumulative

template <typename Op, typename T>
void scan_fn(const std::byte* x, std::byte* o, const LaneCtx& c) {
    scan_lane<Op, T>(x, c.x_stride, o, c.o_stride, c.n);
}

template <typename Op, typename T>
void scan_with_initial_fn(const std::byte* x, std::byte* o, const LaneCtx& c) {
    // Array API: the initial value is the identity, written before the scan.
    *reinterpret_cast<T*>(o) = std::is_same_v<Op, ops::Add> ? T(0) : T(1);
    scan_lane<Op, T>(x, c.x_stride, o + c.o_stride, c.o_stride, c.n);
}

template <typename Op>
nb::object cumulative(nb::handle x_obj, nb::handle axis_obj, nb::handle dtype_obj, bool include_initial,
                      const char* name) {
    const NDArray& x = require_ndarray(x_obj);
    int axis;
    if (axis_obj.is_none()) {
        if (x.ndim != 1)
            throw nb::value_error((std::string(name) + "() needs axis= for arrays with more than one dimension")
                                      .c_str());  // C4
        axis = 0;
    } else {
        axis = normalize_axis(parse_int(axis_obj, "axis"), x.ndim);
    }
    const DType d = accumulate_dtype(x, dtype_obj, name);
    const NDArray xin = as_dtype(x, d);
    std::vector<std::int64_t> shape(x.shape_span().begin(), x.shape_span().end());
    if (include_initial) shape[axis] += 1;
    NDArray out = new_array(d, shape);
    LaneCtx ctx{x.shape[axis], xin.strides[axis], out.strides[axis], nullptr};
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (Op::template ok<T>)  // bool was rejected by accumulate_dtype
            ctx.fn = include_initial ? &scan_with_initial_fn<Op, T> : &scan_fn<Op, T>;
    });
    if (include_initial || ctx.n > 0) run_lanes(xin, out, axis, ctx);
    return wrap(std::move(out));
}

// ------------------------------------------------------------- mean, var

// The dtype mean/var compute in: float64, or complex128 for complex input.
DType inexact(DType d) { return promote(d, DType::Float64); }

// Sum over `ax` in the inexact dtype, divided by the count, keepdims shape.
NDArray mean_keepdims(const NDArray& x, const Axes& ax) {
    const DType d = inexact(x.dtype);
    NDArray out = keepdims_result(x, ax, d, /*zeroed=*/true);  // +0.0 (and 0+0j): the start of a sum
    const double n = static_cast<double>(ax.per_output);
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            run_reduce(x, out, &reduce_loop<ops::Add, T>);
            T* p = reinterpret_cast<T*>(out.data);
            for (std::int64_t i = 0; i < out.size(); ++i) p[i] = p[i] / n;  // empty: 0/0 = nan
        }
    });
    return out;
}

double parse_correction(nb::handle o) {
    if (PyBool_Check(o.ptr()) || !(PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr())))
        throw nb::type_error("correction must be an int or a float");
    const double c = PyFloat_AsDouble(o.ptr());
    if (c == -1.0 && PyErr_Occurred()) throw nb::python_error();
    return c;
}

// Two passes: the mean, then the sum of squared deviations from it, divided
// by N - correction. That is NaN whenever N - correction <= 0 (M5 decision,
// Array API), which also covers empty input.
NDArray var_keepdims(nb::handle x_obj, nb::handle axis_obj, nb::handle correction_obj, Axes& ax) {
    const NDArray& x = require_ndarray(x_obj);
    const double correction = parse_correction(correction_obj);
    ax = parse_reduce_axes(x, axis_obj);
    const NDArray m = mean_keepdims(x, ax);
    const NDArray xin = as_dtype(x, m.dtype);
    NDArray out = keepdims_result(x, ax, DType::Float64, /*zeroed=*/true);
    visit_dtype(m.dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            const IterOperand ops[] = {operand(xin), operand(m), operand(out)};
            NdIter(xin.shape_span(), ops).run(&sq_dev_loop<T>, nullptr, Split::Reduction);
        }
    });
    const double dof = static_cast<double>(ax.per_output) - correction;
    double* p = reinterpret_cast<double*>(out.data);
    for (std::int64_t i = 0; i < out.size(); ++i)
        p[i] = dof > 0 ? p[i] / dof : std::numeric_limits<double>::quiet_NaN();
    return out;
}

}  // namespace

nb::object sum(nb::handle x_obj, nb::handle axis, nb::handle dtype, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    const Axes ax = parse_reduce_axes(x, axis);
    // Starts from +0, so an empty sum is 0 (and sum([-0.0]) is +0.0, as in numpy).
    return fold<ops::Add>(x, ax, accumulate_dtype(x, dtype, "sum"), keepdims,
                          []<typename T>(std::type_identity<T>) { return T(0); });
}

nb::object prod(nb::handle x_obj, nb::handle axis, nb::handle dtype, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    const Axes ax = parse_reduce_axes(x, axis);
    return fold<ops::Multiply>(x, ax, accumulate_dtype(x, dtype, "prod"), keepdims,
                               []<typename T>(std::type_identity<T>) { return T(1); });
}

// max starts from the lowest value of the dtype, min from the highest: then
// the first real element always replaces it (and -inf/-0.0 stay exact).
nb::object max(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) unsupported("max", x.dtype);  // S5
    const Axes ax = parse_reduce_axes(x, axis);
    require_nonempty(x, ax, "max");
    return fold<ops::Maximum>(x, ax, x.dtype, keepdims, []<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T>) return -std::numeric_limits<double>::infinity();
        else return std::numeric_limits<T>::lowest();
    });
}

nb::object min(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) unsupported("min", x.dtype);
    const Axes ax = parse_reduce_axes(x, axis);
    require_nonempty(x, ax, "min");
    return fold<ops::Minimum>(x, ax, x.dtype, keepdims, []<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T>) return std::numeric_limits<double>::infinity();
        else return std::numeric_limits<T>::max();
    });
}

// any/all look at x != 0 (every dtype; nan counts as true).
nb::object any(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    return fold<ops::LogicalOr>(x, parse_reduce_axes(x, axis), DType::Bool, keepdims,
                                []<typename T>(std::type_identity<T>) { return T(false); });
}

nb::object all(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    return fold<ops::LogicalAnd>(x, parse_reduce_axes(x, axis), DType::Bool, keepdims,
                                 []<typename T>(std::type_identity<T>) { return T(true); });
}

nb::object mean(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    const Axes ax = parse_reduce_axes(x, axis);
    return finish(mean_keepdims(x, ax), ax, keepdims);
}

nb::object var(nb::handle x, nb::handle axis, nb::handle correction, bool keepdims) {
    Axes ax;
    NDArray v = var_keepdims(x, axis, correction, ax);
    return finish(std::move(v), ax, keepdims);
}

nb::object stddev(nb::handle x, nb::handle axis, nb::handle correction, bool keepdims) {
    Axes ax;
    NDArray v = var_keepdims(x, axis, correction, ax);
    double* p = reinterpret_cast<double*>(v.data);
    for (std::int64_t i = 0; i < v.size(); ++i) p[i] = std::sqrt(p[i]);
    return finish(std::move(v), ax, keepdims);
}

nb::object argmin(nb::handle x, nb::handle axis, bool keepdims) { return arg_extreme_impl<false>(x, axis, keepdims); }
nb::object argmax(nb::handle x, nb::handle axis, bool keepdims) { return arg_extreme_impl<true>(x, axis, keepdims); }

void bind_reductions(nb::module_& m) {
    using nb::arg;
    const auto axis = arg("axis").none() = nb::none();
    const auto dtype = arg("dtype").none() = nb::none();
    const auto keepdims = arg("keepdims") = false;
    m.def("sum", &sum, arg(), nb::kw_only(), axis, dtype, keepdims);
    m.def("prod", &prod, arg(), nb::kw_only(), axis, dtype, keepdims);
    m.def("min", &min, arg(), nb::kw_only(), axis, keepdims);
    m.def("max", &max, arg(), nb::kw_only(), axis, keepdims);
    m.def("mean", &mean, arg(), nb::kw_only(), axis, keepdims);
    m.def("var", &var, arg(), nb::kw_only(), axis, arg("correction") = 0, keepdims);
    m.def("std", &stddev, arg(), nb::kw_only(), axis, arg("correction") = 0, keepdims);
    m.def("argmin", &argmin, arg(), nb::kw_only(), axis, keepdims);
    m.def("argmax", &argmax, arg(), nb::kw_only(), axis, keepdims);
    m.def("any", &any, arg(), nb::kw_only(), axis, keepdims);
    m.def("all", &all, arg(), nb::kw_only(), axis, keepdims);
    m.def("cumulative_sum", [](nb::handle x, nb::handle axis, nb::handle dtype, bool include_initial) {
        return cumulative<ops::Add>(x, axis, dtype, include_initial, "cumulative_sum");
    }, arg(), nb::kw_only(), axis, dtype, arg("include_initial") = false);
    m.def("cumulative_prod", [](nb::handle x, nb::handle axis, nb::handle dtype, bool include_initial) {
        return cumulative<ops::Multiply>(x, axis, dtype, include_initial, "cumulative_prod");
    }, arg(), nb::kw_only(), axis, dtype, arg("include_initial") = false);
}

}  // namespace sd
