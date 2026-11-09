// Elementwise functions (API.md §5): the dispatcher between Python operands
// and the loop tables of kernels/elementwise.hpp.
//
// Every call goes through the same four steps:
//   1. operands   each is an sdnp.ndarray or a Python bool/int/float/complex
//   2. loop       promote the operand dtypes, then take the first dtype at or
//                 above it that the operation has a loop for (numpy's rule).
//                 C13 and "integers divide in float64" fall out of the tables:
//                 add has no bool loop, divide has no int64 loop.
//   3. convert    inputs are cast to the loop dtype (whole arrays for now;
//                 M7 measures chunked "buffered" casting against this)
//   4. run        NdIter over inputs + output; the output is `out=` (S6,
//                 overlap-safe) or a new array laid out like the inputs
// With no array operand at all the result is a Python scalar (Z1).
#pragma once

#include "kernels/elementwise.hpp"

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

// What elementwise functions accept as an operand (M4 decision: no lists or
// foreign arrays -- convert them with asarray first).
bool is_operand(nb::handle o);

nb::object run_unary(const LoopTable& table, const char* name, nb::handle x, nb::handle out);
nb::object run_binary(const LoopTable& table, const char* name, bool no_bool_inputs, nb::handle a, nb::handle b,
                      nb::handle out);

template <typename Op>
nb::object unary(nb::handle x, nb::handle out) {
    return run_unary(unary_table<Op>, Op::name, x, out);
}

template <typename Op>
nb::object binary(nb::handle a, nb::handle b, nb::handle out) {
    constexpr bool no_bool = requires { Op::no_bool_inputs; };
    return run_binary(binary_table<Op>, Op::name, no_bool, a, b, out);
}

void bind_ufuncs(nb::module_& m);

}  // namespace sd
