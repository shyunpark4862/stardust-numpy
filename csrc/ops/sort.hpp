// sort / argsort and unique_values / unique_counts / unique_inverse /
// unique_all (API.md §7).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_sort(nb::module_& m);

}  // namespace sd
