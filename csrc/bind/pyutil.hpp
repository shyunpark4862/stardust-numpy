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

inline bool is_ndarray(nb::handle o) { return nb::isinstance<NDArray>(o); }
inline NDArray& as_ndarray(nb::handle o) { return *nb::inst_ptr<NDArray>(o); }
inline nb::object wrap(NDArray&& a) { return nb::cast(std::move(a)); }

}  // namespace sd
