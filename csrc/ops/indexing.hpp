// Basic indexing (API.md §4): int, slice, `...` and newaxis.
//
// A basic index never copies: it only moves the data pointer and rewrites
// shape/strides. x[2:8:3] is "start 2 elements in, take 2, stride x3".
// Advanced indexing (integer arrays, boolean masks) arrives in M6.
#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>

namespace sd {

namespace nb = nanobind;

// x[key]: a view, or a Python scalar when every axis is indexed by an int (Z1).
nb::object getitem(nb::handle x, nb::handle key);

// x[key] = value: value (array, nested list or Python scalar) is broadcast to
// the selected shape and converted upward only (S6).
void setitem(nb::handle x, nb::handle key, nb::handle value);

// x[i] for 0 <= i < len(x), without parsing a key: what iteration uses.
nb::object item_along_axis0(nb::handle x, std::int64_t i);

}  // namespace sd
