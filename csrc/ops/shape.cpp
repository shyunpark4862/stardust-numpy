#include "ops/shape.hpp"

#include "bind/pyutil.hpp"
#include "core/iter.hpp"
#include "ops/copy.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace sd {

namespace {

// Row-major strides of a contiguous array of `shape`.
void c_strides(std::span<const std::int64_t> shape, std::int64_t itemsize, std::int64_t* out) {
    std::int64_t stride = itemsize;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        out[d] = stride;
        stride *= std::max<std::int64_t>(shape[d], 1);
    }
}

void set_shape(NDArray& v, std::span<const std::int64_t> shape, const std::int64_t* strides) {
    v.ndim = static_cast<int>(shape.size());
    std::copy(shape.begin(), shape.end(), v.shape.begin());
    std::copy(strides, strides + shape.size(), v.strides.begin());
    v.update_flags();
}

// Resolves a single -1 in the requested shape and checks the element count.
std::vector<std::int64_t> resolve_shape(std::vector<std::int64_t> shape, const NDArray& a) {
    int unknown = -1;
    std::int64_t known = 1;
    bool overflow = false;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (unknown >= 0) throw nb::value_error("can only specify one unknown dimension");
            unknown = static_cast<int>(i);
        } else if (shape[i] >= 0) {
            overflow |= __builtin_mul_overflow(known, shape[i], &known);
        }
    }
    const std::int64_t size = a.size();
    auto fail = [&] {
        return nb::value_error(("cannot reshape array of size " + std::to_string(size) + " into shape " +
                                shape_str(shape))
                                   .c_str());
    };
    if (unknown >= 0) {
        if (overflow || known == 0 || size % known != 0) throw fail();
        shape[unknown] = size / known;
    }
    if (checked_size(shape, a.dtype) != size) throw fail();  // also: no 0-d (Z2), no negative extents
    return shape;
}

// Can `a` be read in row-major order as `shape` without moving any data?
// Walks old and new axes in groups whose extents multiply to the same number
// (e.g. old (6, 4) vs new (2, 3, 4): {6} <-> {2, 3}, {4} <-> {4}). Inside a
// group the old axes must be contiguous with each other; the new strides are
// then carved out of the group's innermost stride. Extent-1 axes play no part.
// This is numpy's _attempt_nocopy_reshape. Requires a.size() > 0.
bool nocopy_strides(const NDArray& a, std::span<const std::int64_t> shape, std::int64_t* out) {
    std::int64_t od[kMaxNdim], os[kMaxNdim];
    int on = 0;
    for (int d = 0; d < a.ndim; ++d) {
        if (a.shape[d] == 1) continue;
        od[on] = a.shape[d];
        os[on] = a.strides[d];
        ++on;
    }
    const int nn = static_cast<int>(shape.size());
    int oi = 0, oj = 1, ni = 0, nj = 1;
    while (ni < nn && oi < on) {
        std::int64_t np = shape[ni], op = od[oi];
        while (np != op) {
            if (np < op) np *= shape[nj++];
            else op *= od[oj++];
        }
        for (int ok = oi; ok < oj - 1; ++ok)
            if (os[ok] != od[ok + 1] * os[ok + 1]) return false;
        out[nj - 1] = os[oj - 1];
        for (int nk = nj - 1; nk > ni; --nk) out[nk - 1] = out[nk] * shape[nk];
        ni = nj++;
        oi = oj++;
    }
    // Trailing extent-1 axes: any stride works; reuse the last one.
    const std::int64_t last = ni > 0 ? out[ni - 1] : a.itemsize();
    for (int nk = ni; nk < nn; ++nk) out[nk] = last;
    return true;
}

// A new C-contiguous copy of `a`, then given `shape`.
NDArray reshaped_copy(const NDArray& a, std::span<const std::int64_t> shape) {
    NDArray out = new_array(a.dtype, shape);
    NDArray as_a = out;  // the same memory, seen with a's shape
    std::int64_t st[kMaxNdim];
    c_strides(a.shape_span(), a.itemsize(), st);
    set_shape(as_a, a.shape_span(), st);
    assign(as_a, a);
    return out;
}

// --------------------------------------------------------------- transposes

nb::object permuted(nb::handle x, std::span<const int> order) {
    const NDArray& a = as_ndarray(x);
    NDArray v = view_of(x);
    for (int d = 0; d < a.ndim; ++d) {
        v.shape[d] = a.shape[order[d]];
        v.strides[d] = a.strides[order[d]];
    }
    v.update_flags();
    return wrap(std::move(v));
}

nb::object permute_dims(nb::handle x, nb::handle axes_obj) {
    const NDArray& a = require_ndarray(x);
    const std::vector<int> axes = parse_axes(axes_obj, a.ndim);
    if (static_cast<int>(axes.size()) != a.ndim)
        throw nb::value_error(("axes must list each of the " + std::to_string(a.ndim) + " axes exactly once").c_str());
    return permuted(x, axes);
}

nb::object moveaxis(nb::handle x, nb::handle source_obj, nb::handle destination_obj) {
    const NDArray& a = require_ndarray(x);
    const std::vector<int> source = parse_axes(source_obj, a.ndim);
    const std::vector<int> destination = parse_axes(destination_obj, a.ndim);
    if (source.size() != destination.size())
        throw nb::value_error("source and destination must have the same number of axes");
    // The axes that stay keep their relative order; the moved ones are
    // inserted at their destinations, smallest destination first.
    std::vector<int> order;
    for (int d = 0; d < a.ndim; ++d)
        if (std::find(source.begin(), source.end(), d) == source.end()) order.push_back(d);
    std::vector<std::pair<int, int>> moves;
    for (size_t i = 0; i < source.size(); ++i) moves.emplace_back(destination[i], source[i]);
    std::sort(moves.begin(), moves.end());
    for (auto [dst, src] : moves) order.insert(order.begin() + dst, src);
    return permuted(x, order);
}

// ------------------------------------------------- adding and removing axes

nb::object squeeze(nb::handle x, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    const std::vector<int> axes = parse_axes(axis_obj, a.ndim);
    for (int ax : axes)
        if (a.shape[ax] != 1)
            throw nb::value_error(("cannot squeeze axis " + std::to_string(ax) + ": its size is " +
                                   std::to_string(a.shape[ax]) + ", not 1")
                                      .c_str());
    if (axes.size() == static_cast<size_t>(a.ndim))
        throw nb::value_error("squeezing every axis would give a 0-dimensional array (stardust-numpy has none)");
    NDArray v = view_of(x);
    v.ndim = 0;
    for (int d = 0; d < a.ndim; ++d) {
        if (std::find(axes.begin(), axes.end(), d) != axes.end()) continue;
        v.shape[v.ndim] = a.shape[d];
        v.strides[v.ndim] = a.strides[d];
        ++v.ndim;
    }
    v.update_flags();
    return wrap(std::move(v));
}

nb::object expand_dims(nb::handle x, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    const int added = PyTuple_Check(axis_obj.ptr()) ? static_cast<int>(PyTuple_GET_SIZE(axis_obj.ptr())) : 1;
    const int out_ndim = a.ndim + added;
    if (out_ndim > kMaxNdim)
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    const std::vector<int> axes = parse_axes(axis_obj, out_ndim);  // positions in the *result*
    NDArray v = view_of(x);
    v.ndim = out_ndim;
    // Right to left, so a new axis can take the stride "one step of the axis
    // to its right" (numpy's choice; any stride is valid for extent 1).
    int src = a.ndim - 1;
    for (int d = out_ndim - 1; d >= 0; --d) {
        if (std::find(axes.begin(), axes.end(), d) != axes.end()) {
            v.shape[d] = 1;
            v.strides[d] = d + 1 < out_ndim ? v.strides[d + 1] * v.shape[d + 1] : a.itemsize();
        } else {
            v.shape[d] = a.shape[src];
            v.strides[d] = a.strides[src];
            --src;
        }
    }
    v.update_flags();
    return wrap(std::move(v));
}

// --------------------------------------------------------------- broadcasting

// Read-only view of x with `shape`: new leading axes and stretched extent-1
// axes get stride 0, so one stored element stands for a whole row.
NDArray broadcast_view(nb::handle x, std::span<const std::int64_t> shape) {
    const NDArray& a = as_ndarray(x);
    checked_size(shape, a.dtype);
    const int offset = static_cast<int>(shape.size()) - a.ndim;
    auto fail = [&] {
        return nb::value_error(("cannot broadcast an array of shape " + shape_str(a.shape_span()) + " to shape " +
                                shape_str(shape))
                                   .c_str());
    };
    if (offset < 0) throw fail();
    NDArray v = view_of(x);
    v.ndim = static_cast<int>(shape.size());
    for (int d = 0; d < v.ndim; ++d) {
        const int od = d - offset;
        v.shape[d] = shape[d];
        if (od < 0) v.strides[d] = 0;
        else if (a.shape[od] == shape[d]) v.strides[d] = a.strides[od];
        else if (a.shape[od] == 1) v.strides[d] = 0;
        else throw fail();
    }
    // Many indices now alias one element: writing through the view would
    // silently write the same memory several times.
    v.writeable = false;
    v.update_flags();
    return v;
}

nb::object broadcast_to(nb::handle x, nb::handle shape_obj) {
    require_ndarray(x);
    return wrap(broadcast_view(x, parse_shape(shape_obj)));
}

nb::tuple broadcast_arrays(nb::args arrays) {
    std::vector<std::span<const std::int64_t>> shapes;
    for (nb::handle h : arrays) shapes.push_back(require_ndarray(h, "every argument").shape_span());
    if (shapes.empty()) return nb::tuple();
    const std::vector<std::int64_t> shape = broadcast_shapes(shapes);
    nb::list out;
    for (nb::handle h : arrays) out.append(wrap(broadcast_view(h, shape)));
    return nb::tuple(out);
}

// --------------------------------------------------------------- joining

std::vector<nb::handle> array_list(nb::handle arrays) {
    if (!PyList_Check(arrays.ptr()) && !PyTuple_Check(arrays.ptr()))
        throw nb::type_error("arrays must be a list or tuple of sdnp.ndarray");
    std::vector<nb::handle> out;
    for (nb::handle h : arrays) {
        require_ndarray(h, "every element of arrays");
        out.push_back(h);
    }
    if (out.empty()) throw nb::value_error("need at least one array");
    return out;
}

DType promoted(const std::vector<nb::handle>& arrays) {
    DType d = DType::Bool;
    for (nb::handle h : arrays) d = promote(d, as_ndarray(h).dtype);
    return d;
}

// out[..., offset:offset+n, ...] along `axis` (a view into a fresh result).
NDArray slab(const NDArray& out, int axis, std::int64_t offset, std::int64_t n) {
    NDArray v = out;
    v.data += offset * out.strides[axis];
    v.shape[axis] = n;
    v.update_flags();
    return v;
}

nb::object concat(nb::handle arrays_obj, nb::handle axis_obj) {
    const std::vector<nb::handle> arrays = array_list(arrays_obj);
    const DType dtype = promoted(arrays);

    if (axis_obj.is_none()) {
        // Concatenate the flattened arrays: each input fills a contiguous run
        // of the 1-D result, which is viewed with the input's shape.
        std::int64_t total = 0;
        for (nb::handle h : arrays) total += as_ndarray(h).size();
        const std::int64_t shape[] = {total};
        NDArray out = new_array(dtype, shape);
        std::int64_t offset = 0;
        for (nb::handle h : arrays) {
            const NDArray& a = as_ndarray(h);
            NDArray dst = slab(out, 0, offset, a.size());
            std::int64_t st[kMaxNdim];
            c_strides(a.shape_span(), dst.itemsize(), st);
            set_shape(dst, a.shape_span(), st);
            assign(dst, a);
            offset += a.size();
        }
        return wrap(std::move(out));
    }

    const NDArray& first = as_ndarray(arrays[0]);
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), first.ndim);
    std::vector<std::int64_t> shape(first.shape_span().begin(), first.shape_span().end());
    shape[axis] = 0;
    for (nb::handle h : arrays) {
        const NDArray& a = as_ndarray(h);
        bool ok = a.ndim == first.ndim;
        for (int d = 0; ok && d < a.ndim; ++d) ok = d == axis || a.shape[d] == first.shape[d];
        if (!ok)
            throw nb::value_error(("all arrays must have the same shape except along axis " + std::to_string(axis) +
                                   "; got " + shape_str(first.shape_span()) + " and " + shape_str(a.shape_span()))
                                      .c_str());
        shape[axis] += a.shape[axis];
    }
    NDArray out = new_array(dtype, shape);
    std::int64_t offset = 0;
    for (nb::handle h : arrays) {
        const NDArray& a = as_ndarray(h);
        NDArray dst = slab(out, axis, offset, a.shape[axis]);
        assign(dst, a);
        offset += a.shape[axis];
    }
    return wrap(std::move(out));
}

nb::object stack(nb::handle arrays_obj, nb::handle axis_obj) {
    const std::vector<nb::handle> arrays = array_list(arrays_obj);
    const NDArray& first = as_ndarray(arrays[0]);
    for (nb::handle h : arrays) {
        const NDArray& a = as_ndarray(h);
        if (!std::ranges::equal(a.shape_span(), first.shape_span()))
            throw nb::value_error(("all arrays must have the same shape; got " + shape_str(first.shape_span()) +
                                   " and " + shape_str(a.shape_span()))
                                      .c_str());
    }
    if (first.ndim + 1 > kMaxNdim)
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), first.ndim + 1);
    std::vector<std::int64_t> shape(first.shape_span().begin(), first.shape_span().end());
    shape.insert(shape.begin() + axis, static_cast<std::int64_t>(arrays.size()));
    NDArray out = new_array(promoted(arrays), shape);
    // Input i is the result with index i fixed along `axis`.
    for (size_t i = 0; i < arrays.size(); ++i) {
        NDArray dst = out;
        dst.data += static_cast<std::int64_t>(i) * out.strides[axis];
        std::copy(out.shape.begin() + axis + 1, out.shape.begin() + out.ndim, dst.shape.begin() + axis);
        std::copy(out.strides.begin() + axis + 1, out.strides.begin() + out.ndim, dst.strides.begin() + axis);
        dst.ndim = out.ndim - 1;
        dst.update_flags();
        assign(dst, as_ndarray(arrays[i]));
    }
    return wrap(std::move(out));
}

// --------------------------------------------------------------- real / imag

// The out= tail shared by elementwise functions: no out -> the result itself;
// otherwise the result is written into out (S6, broadcast to out's shape).
nb::object finish(nb::object result, nb::handle out) {
    if (out.is_none()) return result;
    assign_checked(require_ndarray(out, "out"), as_ndarray(result));
    return nb::borrow(out);
}

// A complex128 element is two float64s side by side (real, then imaginary),
// so the real parts are a float64 view with the same strides, and the
// imaginary parts the same view shifted by 8 bytes.
nb::object complex_part(nb::handle x, std::int64_t byte_offset) {
    NDArray v = view_of(x);
    v.dtype = DType::Float64;
    v.data += byte_offset;
    v.update_flags();
    return wrap(std::move(v));
}

nb::object real(nb::handle x, nb::handle out) {
    const NDArray& a = require_ndarray(x);
    if (a.dtype == DType::Complex128) return finish(complex_part(x, 0), out);
    return finish(nb::borrow(x), out);  // a real array is its own real part
}

nb::object imag(nb::handle x, nb::handle out) {
    const NDArray& a = require_ndarray(x);
    if (a.dtype == DType::Complex128) return finish(complex_part(x, sizeof(double)), out);
    // A real array has no imaginary storage to view: a fresh array of zeros,
    // read-only like numpy's, since writing to it could not reach x.
    NDArray zeros = new_array_like(a, a.dtype, /*zeroed=*/true);
    zeros.writeable = false;
    return finish(wrap(std::move(zeros)), out);
}

}  // namespace

nb::object reshape(nb::handle x, std::span<const std::int64_t> requested, std::optional<bool> copy) {
    const NDArray& a = require_ndarray(x);
    const std::vector<std::int64_t> shape =
        resolve_shape(std::vector<std::int64_t>(requested.begin(), requested.end()), a);
    if (copy == true) return wrap(reshaped_copy(a, shape));
    std::int64_t st[kMaxNdim];
    bool viewable = true;
    if (a.size() == 0) c_strides(shape, a.itemsize(), st);  // nothing to read: any strides do
    else viewable = nocopy_strides(a, shape, st);
    if (viewable) {
        NDArray v = view_of(x);
        set_shape(v, shape, st);
        return wrap(std::move(v));
    }
    if (copy == false) throw nb::value_error("copy=False, but this reshape cannot be done without copying");
    return wrap(reshaped_copy(a, shape));
}

nb::object reshape(nb::handle x, nb::handle shape, nb::handle copy) {
    return reshape(x, parse_shape(shape), parse_copy(copy));
}

nb::object matrix_transpose(nb::handle x) {
    const NDArray& a = require_ndarray(x);
    if (a.ndim < 2) throw nb::value_error("matrix_transpose requires an array with at least 2 dimensions");
    std::vector<int> order(a.ndim);
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[a.ndim - 2], order[a.ndim - 1]);
    return permuted(x, order);
}

nb::object transpose_2d(nb::handle x) {
    const NDArray& a = as_ndarray(x);
    if (a.ndim != 2)
        throw nb::value_error(".T requires a 2-dimensional array; use permute_dims or matrix_transpose (.mT)");
    const int order[] = {1, 0};
    return permuted(x, order);
}

void bind_shape(nb::module_& m) {
    using nb::arg;
    m.def("reshape", nb::overload_cast<nb::handle, nb::handle, nb::handle>(&reshape), arg(), arg(), nb::kw_only(),
          arg("copy").none() = nb::none());
    m.def("ravel", [](nb::handle x) {
        const std::int64_t flat[] = {-1};
        return reshape(x, flat, std::nullopt);
    }, arg());
    m.def("permute_dims", &permute_dims, arg(), arg());
    m.def("matrix_transpose", &matrix_transpose, arg());
    m.def("moveaxis", &moveaxis, arg(), arg(), arg());
    m.def("squeeze", &squeeze, arg(), nb::kw_only(), arg("axis"));
    m.def("expand_dims", &expand_dims, arg(), nb::kw_only(), arg("axis"));
    m.def("broadcast_to", &broadcast_to, arg(), arg());
    m.def("broadcast_arrays", &broadcast_arrays);
    m.def("concat", &concat, arg(), nb::kw_only(), arg("axis").none() = 0);
    m.def("stack", &stack, arg(), nb::kw_only(), arg("axis") = 0);
    m.def("real", &real, arg(), nb::kw_only(), arg("out").none() = nb::none());
    m.def("imag", &imag, arg(), nb::kw_only(), arg("out").none() = nb::none());
}

}  // namespace sd
