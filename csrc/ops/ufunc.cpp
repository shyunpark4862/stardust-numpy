#include "ops/ufunc.hpp"

#include "bind/pyutil.hpp"
#include "core/convert.hpp"
#include "kernels/buffered.hpp"
#include "kernels/cast.hpp"
#include "ops/copy.hpp"

#include <array>
#include <string>
#include <vector>

namespace sd {

namespace {

constexpr int kMaxInputs = kMaxOperands - 1;

DType operand_dtype(nb::handle o, const char* name) {
    if (is_ndarray(o)) return as_ndarray(o).dtype;
    if (std::optional<DType> kind = scalar_dtype(o)) return *kind;
    throw nb::type_error((std::string(name) + "() takes sdnp.ndarray or Python bool/int/float/complex operands, not '" +
                          Py_TYPE(o.ptr())->tp_name + "'")
                             .c_str());
}

[[noreturn]] void no_loop(const char* name, std::span<const DType> dtypes) {
    std::string msg = std::string(name) + "() is not supported for ";
    for (size_t i = 0; i < dtypes.size(); ++i) msg += (i ? ", " : "") + std::string(dtype_name(dtypes[i]));
    throw nb::type_error(msg.c_str());
}

// numpy's loop selection: the first dtype at or above `d` with a loop.
// Returns that dtype; the table entry says what it writes.
DType select(const LoopTable& table, DType d, const char* name, std::span<const DType> inputs) {
    for (int k = static_cast<int>(d); k < kNumDTypes; ++k)
        if (table[k].loop) return static_cast<DType>(k);
    no_loop(name, inputs);
}

// Operand `o` as an array. Arrays are used as they are -- a dtype that
// differs from the loop's is converted chunk by chunk while the loop runs
// (buffered casting, M7.6). A Python scalar becomes a 1-element array of the
// loop dtype `to` (broadcast by the iterator); conversion follows Z4, so an
// int that does not fit raises OverflowError only when the loop runs in int64.
NDArray prepare(nb::handle o, DType to) {
    if (is_ndarray(o)) return as_ndarray(o);
    const DType kind = *scalar_dtype(o);
    const std::int64_t one[] = {1};
    NDArray s = new_array(to, one);
    visit_dtype(to, [&]<typename T>(std::type_identity<T>) {
        *reinterpret_cast<T*>(s.data) = scalar_from_python<T>(o, kind);
    });
    return s;
}

// A new result array: laid out like the first array operand that already has
// the full shape (an F-ordered input gives an F-ordered result, keeping the
// iterator's coalescing), else C order.
NDArray new_result(std::span<const nb::handle> objs, std::span<const std::int64_t> shape, DType dtype) {
    for (nb::handle o : objs)
        if (is_ndarray(o) && std::ranges::equal(as_ndarray(o).shape_span(), shape))
            return new_array_like(as_ndarray(o), dtype);
    return new_array(dtype, shape);
}

// Runs `loop` over ins + out. Inputs whose dtype is not the loop's
// (`in_types`) are converted through small buffers on the way in.
void run_loop(std::span<const NDArray> ins, std::span<const DType> in_types, const NDArray& out,
              std::span<const std::int64_t> shape, Loop loop) {
    std::array<IterOperand, kMaxOperands> ops;
    BufferedLoop buffered;
    buffered.inner = loop;
    buffered.nin = static_cast<int>(ins.size());
    buffered.nop = buffered.nin + 1;
    for (size_t k = 0; k < ins.size(); ++k) {
        ops[k] = operand(ins[k]);
        if (ins[k].dtype != in_types[k])
            buffered.add_cast(static_cast<int>(k), cast_kernel(ins[k].dtype, in_types[k]), itemsize(in_types[k]));
    }
    ops[ins.size()] = operand(out);
    const NdIter it(shape, std::span(ops.data(), ins.size() + 1));
    if (buffered.needed()) it.run(&buffered_loop, &buffered);
    else it.run(loop);
}

// Steps 3 and 4 (see ufunc.hpp): convert, then run into out= or a new array.
nb::object launch(std::span<const nb::handle> objs, std::span<const DType> in_types, DType result, Loop loop,
                  nb::handle out_obj) {
    const size_t nin = objs.size();
    std::array<NDArray, kMaxInputs> ins;
    std::array<std::span<const std::int64_t>, kMaxInputs> shapes;  // fixed arrays: no allocation per call
    size_t narrays = 0;
    for (size_t k = 0; k < nin; ++k) {
        ins[k] = prepare(objs[k], in_types[k]);
        if (is_ndarray(objs[k])) shapes[narrays++] = as_ndarray(objs[k]).shape_span();
    }
    const bool scalars_only = narrays == 0;
    Dims dims;
    dims[0] = 1;
    const int ndim = scalars_only ? 1 : broadcast_into(std::span(shapes.data(), narrays), dims);
    const std::span<const std::int64_t> shape(dims.data(), static_cast<size_t>(ndim));
    const std::span<const NDArray> inputs(ins.data(), nin);

    if (!out_obj.is_none()) {
        NDArray& out = require_ndarray(out_obj, "out");
        if (!out.writeable) throw nb::value_error("output array is read-only");
        check_assignable(result, out.dtype);  // S6
        if (out.dtype == result) {
            // Write straight into out. An input that partly overlaps it is
            // copied first (R3); one that is the very same view is safe,
            // since each element is read just before the same place is written.
            for (size_t k = 0; k < nin; ++k)
                if (may_share_memory(ins[k], out) && !same_view(ins[k], out)) ins[k] = cast_copy(ins[k], ins[k].dtype);
            run_loop(inputs, in_types, out, out.shape_span(), loop);
        } else {
            // out is a wider dtype: compute, then convert into it.
            NDArray tmp = new_result(objs, shape, result);
            run_loop(inputs, in_types, tmp, shape, loop);
            assign(out, tmp);
        }
        return nb::borrow(out_obj);
    }

    NDArray res = new_result(objs, shape, result);
    run_loop(inputs, in_types, res, shape, loop);
    if (scalars_only) return load_python(result, res.data);
    return wrap(std::move(res));
}

// ------------------------------------------------------------------ where, clip

template <typename T> constexpr Loop where_entry = &where_loop<T>;
constexpr std::array<Loop, kNumDTypes> where_loops = {
    where_entry<bool>, where_entry<std::int64_t>, where_entry<double>, where_entry<complex128>};

nb::object where(nb::handle cond, nb::handle x1, nb::handle x2, nb::handle out) {
    const DType dc = operand_dtype(cond, "where");
    if (dc != DType::Bool)
        throw nb::type_error(("where() condition must be bool, not " + std::string(dtype_name(dc)) +
                              "; compare explicitly, e.g. x != 0")
                                 .c_str());
    const DType d = promote(operand_dtype(x1, "where"), operand_dtype(x2, "where"));
    const nb::handle objs[] = {cond, x1, x2};
    const DType types[] = {DType::Bool, d, d};
    return launch(objs, types, d, where_loops[static_cast<int>(d)], out);
}

// clip keeps x's dtype (M4 decision, Array API): min and max must convert to
// it implicitly (S6). No complex: it has no order (S5).
template <typename T> constexpr Loop clip_entry = &clip_loop<T>;
constexpr std::array<Loop, kNumDTypes> clip_loops = {
    clip_entry<bool>, clip_entry<std::int64_t>, clip_entry<double>, nullptr};

nb::object clip(nb::handle x, nb::handle lo, nb::handle hi, nb::handle out) {
    const DType d = operand_dtype(x, "clip");
    if (d == DType::Complex128) throw nb::type_error("clip() is not supported for complex128 (no ordering)");
    for (nb::handle bound : {lo, hi})
        if (!bound.is_none()) check_assignable(operand_dtype(bound, "clip"), d);
    const int k = static_cast<int>(d);
    if (lo.is_none() && hi.is_none()) {
        const nb::handle objs[] = {x};
        const DType types[] = {d};
        return launch(objs, types, d, cast_kernel(d, d), out);  // a plain copy
    }
    if (hi.is_none()) {
        const nb::handle objs[] = {x, lo};
        const DType types[] = {d, d};
        return launch(objs, types, d, binary_table<ops::Maximum>[k].loop, out);
    }
    if (lo.is_none()) {
        const nb::handle objs[] = {x, hi};
        const DType types[] = {d, d};
        return launch(objs, types, d, binary_table<ops::Minimum>[k].loop, out);
    }
    const nb::handle objs[] = {x, lo, hi};
    const DType types[] = {d, d, d};
    return launch(objs, types, d, clip_loops[k], out);
}

}  // namespace

bool is_operand(nb::handle o) { return is_ndarray(o) || scalar_dtype(o).has_value(); }

nb::object run_unary(const LoopTable& table, const char* name, nb::handle x, nb::handle out) {
    const DType d = operand_dtype(x, name);
    const DType loop = select(table, d, name, std::span(&d, 1));
    const nb::handle objs[] = {x};
    return launch(objs, std::span(&loop, 1), table[static_cast<int>(loop)].out, table[static_cast<int>(loop)].loop,
                  out);
}

nb::object run_binary(const LoopTable& table, const char* name, bool no_bool_inputs, nb::handle a, nb::handle b,
                      nb::handle out) {
    const DType dtypes[] = {operand_dtype(a, name), operand_dtype(b, name)};
    if (no_bool_inputs && (dtypes[0] == DType::Bool || dtypes[1] == DType::Bool)) no_loop(name, dtypes);
    const DType loop = select(table, promote(dtypes[0], dtypes[1]), name, dtypes);
    const nb::handle objs[] = {a, b};
    const DType types[] = {loop, loop};
    return launch(objs, types, table[static_cast<int>(loop)].out, table[static_cast<int>(loop)].loop, out);
}

namespace {

template <typename Op>
void def_unary(nb::module_& m) {
    m.def(Op::name, &unary<Op>, nb::arg(), nb::kw_only(), nb::arg("out").none() = nb::none());
}

template <typename Op>
void def_binary(nb::module_& m) {
    m.def(Op::name, &binary<Op>, nb::arg(), nb::arg(), nb::kw_only(), nb::arg("out").none() = nb::none());
}

template <typename... Ops>
void def_unaries(nb::module_& m) { (def_unary<Ops>(m), ...); }

template <typename... Ops>
void def_binaries(nb::module_& m) { (def_binary<Ops>(m), ...); }

}  // namespace

void bind_ufuncs(nb::module_& m) {
    using namespace ops;
    def_binaries<Add, Subtract, Multiply, Divide, FloorDivide, Remainder, Pow, Atan2, Maximum, Minimum,
                 Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, LogicalAnd, LogicalOr, LogicalXor,
                 BitwiseAnd, BitwiseOr, BitwiseXor, LeftShift, RightShift>(m);
    def_unaries<Negative, Positive, Abs, Square, Sqrt, Sign, Exp, Expm1, Log, Log1p, Log2, Log10, Sin, Cos, Tan,
                Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Floor, Ceil, Trunc, Round, IsNan, IsInf,
                IsFinite, LogicalNot, BitwiseInvert, Conj>(m);
    m.def("where", &where, nb::arg(), nb::arg(), nb::arg(), nb::kw_only(), nb::arg("out").none() = nb::none());
    m.def("clip", &clip, nb::arg(), nb::kw_only(), nb::arg("min").none() = nb::none(),
          nb::arg("max").none() = nb::none(), nb::arg("out").none() = nb::none());
}

}  // namespace sd
