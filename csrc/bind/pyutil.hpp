// Glue between Python objects and the C++ core: dtype objects, argument
// parsing shared by many functions, and wrapping NDArray as a Python object.
#pragma once

#include "core/ndarray.hpp"

#include <nanobind/nanobind.h>

#include <optional>
#include <vector>

namespace sd {

namespace nb = nanobind;

// Python-side dtype object. Exactly four instances exist (sdnp.bool, ...),
// so dtypes compare by identity.
struct PyDType {
    DType value;
};

void register_dtypes(nb::module_& m);
nb::handle dtype_object(DType d);

// `dtype=` argument: None -> nullopt; otherwise must be one of the four
// dtype objects (Python types like `float` are rejected: API.md §0).
std::optional<DType> parse_dtype(nb::handle o);

// `copy=` argument: None / True / False.
std::optional<bool> parse_copy(nb::handle o);

// `shape` argument: an int or a tuple of ints (C2).
std::vector<std::int64_t> parse_shape(nb::handle o);

// A plain Python int (bool rejected), e.g. `num` or `k`.
std::int64_t parse_int(nb::handle o, const char* what);

// Maps axis in [-ndim, ndim) to [0, ndim); ValueError otherwise.
int normalize_axis(std::int64_t axis, int ndim);

// `axis` argument: an int or a tuple of ints (C2), each normalized against
// ndim. Repeated axes are a ValueError.
std::vector<int> parse_axes(nb::handle o, int ndim);

inline bool is_ndarray(nb::handle o) { return nb::isinstance<NDArray>(o); }
inline NDArray& as_ndarray(nb::handle o) { return *nb::inst_ptr<NDArray>(o); }
// as_ndarray, with a TypeError naming the argument when `o` is not an array.
NDArray& require_ndarray(nb::handle o, const char* what = "x");
inline nb::object wrap(NDArray&& a) { return nb::cast(std::move(a)); }

// A new array object over the same memory as array `x`, ready to have its
// data/shape/strides edited (call update_flags() afterwards). Its `base` is
// the final owner of the memory: views of views never form chains.
NDArray view_of(nb::handle x);

}  // namespace sd
