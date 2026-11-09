// Reductions (API.md §6): sum, prod, min, max, mean, var, std, argmin,
// argmax, any, all, cumulative_sum, cumulative_prod.
//
// Results: a reduction over every axis gives a Python scalar (Z1) unless
// keepdims=True. Summation order is not numpy's (DESIGN.md §2.4) but is
// fixed: the same data in the same layout always gives the same bits.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

nb::object sum(nb::handle x, nb::handle axis, nb::handle dtype, bool keepdims);
nb::object prod(nb::handle x, nb::handle axis, nb::handle dtype, bool keepdims);
nb::object min(nb::handle x, nb::handle axis, bool keepdims);
nb::object max(nb::handle x, nb::handle axis, bool keepdims);
nb::object mean(nb::handle x, nb::handle axis, bool keepdims);
nb::object var(nb::handle x, nb::handle axis, nb::handle correction, bool keepdims);
nb::object stddev(nb::handle x, nb::handle axis, nb::handle correction, bool keepdims);  // Python: std
nb::object argmin(nb::handle x, nb::handle axis, bool keepdims);
nb::object argmax(nb::handle x, nb::handle axis, bool keepdims);
nb::object any(nb::handle x, nb::handle axis, bool keepdims);
nb::object all(nb::handle x, nb::handle axis, bool keepdims);

void bind_reductions(nb::module_& m);

}  // namespace sd
