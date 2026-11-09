// The n-dimensional iterator: a scaled-down numpy `nditer` (ARCHITECTURE.md §4.2).
//
// Every elementwise operation (copy, cast, add, ...) has the same shape:
// walk N arrays of one common shape in lockstep and apply a 1-D "inner loop"
// kernel. The iterator turns the N-d problem into as few, as long, as
// memory-friendly 1-D runs as possible, in four steps:
//
//   1. broadcast   every operand gets the common shape; broadcast axes get stride 0
//   2. drop        axes of extent 1 (never stepped over)
//   3. reorder     axes sorted by |stride| so the innermost loop walks memory
//                  sequentially (a transposed array is read in memory order)
//   4. coalesce    neighbouring axes that are contiguous for *every* operand are
//                  merged; a contiguous (3, 4, 5) array becomes one run of 60
//
// What remains is driven by an odometer over the outer axes, calling the
// kernel once per innermost run.
#pragma once

#include "core/ndarray.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace sd {

inline constexpr int kMaxOperands = 4;  // e.g. where(cond, x1, x2) -> out

// The signature every kernel ("inner loop") has: process n elements; operand
// k starts at args[k] and advances strides[k] bytes per element. As in numpy,
// inputs come first and outputs last.
using Loop = void (*)(std::byte* const* args, std::int64_t n, const std::int64_t* strides, void* ctx);

struct IterOperand {
    std::byte* data;
    std::span<const std::int64_t> shape;
    std::span<const std::int64_t> strides;
};

inline IterOperand operand(const NDArray& a) { return {a.data, a.shape_span(), a.strides_span()}; }

// Naive and Reorder exist only so the effect of each step can be measured
// (bench/m2_iterator.py); production code always uses Full.
enum class IterMode { Naive, Reorder, Full };

// The broadcast shape of several shapes (numpy rules: align right; extents
// must match or be 1). Raises ValueError when they are incompatible.
std::vector<std::int64_t> broadcast_shapes(std::span<const std::span<const std::int64_t>> shapes);

class NdIter {
public:
    // Plans the traversal of `ops`, each broadcast to `shape`.
    NdIter(std::span<const std::int64_t> shape, std::span<const IterOperand> ops, IterMode mode = IterMode::Full);

    int nop() const { return nop_; }
    int ndim() const { return ndim_; }
    bool empty() const { return empty_; }
    std::int64_t extent(int d) const { return shape_[d]; }
    std::int64_t stride(int op, int d) const { return strides_[op][d]; }

    // Calls kernel(args, n, strides, ctx) once per innermost run.
    void run(Loop kernel, void* ctx = nullptr) const;

private:
    int nop_ = 0;
    int ndim_ = 0;
    bool empty_ = false;
    Dims shape_{};
    std::array<Dims, kMaxOperands> strides_{};
    std::array<std::byte*, kMaxOperands> data_{};
};

}  // namespace sd
