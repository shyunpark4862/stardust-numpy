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

// The shortest piece of the innermost axis a parallel run hands out (M8).
inline constexpr std::int64_t kMinInnerPiece = 1024;

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

// How run() may split the work across threads (M8). Results never depend on
// it -- only which splits are safe does:
//   Elementwise  every element is computed on its own: any axis may be cut
//   Reduction    the last operand is the output, broadcast (stride 0) along
//                the reduced axes; only axes where it has a real stride may
//                be cut, so each output element still folds its inputs in
//                exactly the serial order, on one thread
//   None         never split
enum class Split { None, Elementwise, Reduction };

// The broadcast shape of several shapes (numpy rules: align right; extents
// must match or be 1). Raises ValueError when they are incompatible.
std::vector<std::int64_t> broadcast_shapes(std::span<const std::span<const std::int64_t>> shapes);
// The same into a fixed array, without allocating; returns the ndim.
int broadcast_into(std::span<const std::span<const std::int64_t>> shapes, Dims& out);

class NdIter {
public:
    // Plans the traversal of `ops`, each broadcast to `shape`.
    NdIter(std::span<const std::int64_t> shape, std::span<const IterOperand> ops, IterMode mode = IterMode::Full);

    int nop() const { return nop_; }
    int ndim() const { return ndim_; }
    bool empty() const { return empty_; }
    std::int64_t extent(int d) const { return shape_[d]; }
    std::int64_t stride(int op, int d) const { return strides_[op][d]; }

    // Calls kernel(args, n, strides, ctx) once per innermost run. With enough
    // work, the runs are spread over the thread pool (see Split); the kernel
    // and ctx must then be safe to call from several threads at once.
    // `weight` is the work per element, for kernels that do more than one
    // element's worth per call position (a lane kernel processes a whole lane).
    void run(Loop kernel, void* ctx = nullptr, Split split = Split::Elementwise, std::int64_t weight = 1) const;

private:
    void run_serial(Loop kernel, void* ctx) const;
    int split_axis(Split split) const;  // -1: no axis may be split

    int nop_ = 0;
    int ndim_ = 0;
    bool empty_ = false;
    // Left uninitialized: only the first ndim_ entries (and nop_ operands)
    // are ever written or read. Zero-filling 1.3 KB per iterator was a
    // visible cost on tiny arrays (M7.7).
    Dims shape_;
    std::array<Dims, kMaxOperands> strides_;
    std::array<std::byte*, kMaxOperands> data_{};
};

}  // namespace sd
