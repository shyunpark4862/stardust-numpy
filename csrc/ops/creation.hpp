// Array creation (API.md §2). Python-facing signatures take raw handles so
// that argument rules (C1, C2, Q1, Z1-Z4) are enforced in one place.
#pragma once

#include "core/ndarray.hpp"

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

nb::object asarray(nb::handle obj, nb::handle dtype, nb::handle copy);

nb::object zeros(nb::handle shape, nb::handle dtype);
nb::object ones(nb::handle shape, nb::handle dtype);
nb::object empty(nb::handle shape, nb::handle dtype);
nb::object full(nb::handle shape, nb::handle fill_value, nb::handle dtype);

nb::object zeros_like(nb::handle x, nb::handle dtype);
nb::object ones_like(nb::handle x, nb::handle dtype);
nb::object empty_like(nb::handle x, nb::handle dtype);
nb::object full_like(nb::handle x, nb::handle fill_value, nb::handle dtype);

nb::object arange(nb::handle start, nb::handle stop, nb::handle step, nb::handle dtype);
nb::object linspace(nb::handle start, nb::handle stop, nb::handle num, nb::handle dtype, nb::handle endpoint);
nb::object eye(nb::handle n_rows, nb::handle n_cols, nb::handle k, nb::handle dtype);

void bind_creation(nb::module_& m);

}  // namespace sd
