// Copying and converting whole arrays on top of the iterator: astype, copy,
// and the assignment engine that later serves setitem and `out=`.
#pragma once

#include "core/iter.hpp"
#include "core/ndarray.hpp"

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

// S6: an implicit conversion (assignment, out=, in-place) may only go up the
// promotion order bool < int64 < float64 < complex128.
void check_assignable(DType from, DType to);

// Conservative overlap test on the byte ranges the two arrays can touch.
bool may_share_memory(const NDArray& a, const NDArray& b);

// dst[...] = src: src is broadcast to dst's shape and converted to dst's
// dtype. If the two overlap (and are not the very same view), src is first
// copied to a temporary, so the result is as if every input element were read
// before any output element is written (R3). Policy checks (writeable, cast
// rule) belong to the caller.
void assign(NDArray& dst, const NDArray& src);

// assign() plus the policy checks shared by setitem, out= and in-place
// operators: dst must be writeable (ValueError) and src's dtype must convert
// to dst's implicitly (S6, TypeError).
void assign_checked(NDArray& dst, const NDArray& src);

// New array with src's values converted to `to`, laid out like src.
NDArray cast_copy(const NDArray& src, DType to, IterMode mode = IterMode::Full);

nb::object astype(nb::handle x, nb::handle dtype, nb::handle copy);

void bind_copy(nb::module_& m);

}  // namespace sd
