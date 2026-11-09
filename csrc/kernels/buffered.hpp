// Buffered casting (ARCHITECTURE.md §4.2 step 6; M7.6).
//
// When an input's dtype differs from the loop's (int64 + float64 runs the
// float64 loop), M4 converted the whole input into a temporary array first:
// one full pass over memory to write it, another to read it back. Here the
// conversion happens in chunks of kBufferElements instead: cast a chunk into
// a small buffer that stays in L1, run the kernel on it, move on. Memory
// traffic is that of the inputs and the output only.
//
// BufferedLoop wraps any kernel with the M2 signature (inputs first, outputs
// last) and is itself a kernel with that signature, so the iterator does not
// know it is there.
#pragma once

#include "core/iter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

namespace sd {

// 4096 elements: three 32 KB buffers fit the 128 KB L1 of a P-core
// (ARCHITECTURE.md §4.2; the other sizes measured in M7.6 are in the explainer).
inline constexpr std::int64_t kBufferElements = 4096;

struct BufferedLoop {
    Loop inner = nullptr;
    int nin = 0;  // inputs; operands after them are outputs, never buffered
    int nop = 0;  // all operands
    std::array<Loop, kMaxOperands> cast{};          // input k -> loop dtype, or null: used in place
    std::array<std::int64_t, kMaxOperands> item{};  // loop-dtype itemsize of input k

    void add_cast(int k, Loop c, std::int64_t itemsize) {
        cast[k] = c;
        item[k] = itemsize;
    }
    bool needed() const {
        return std::any_of(cast.begin(), cast.begin() + nin, [](Loop c) { return c != nullptr; });
    }
};

// The kernel; ctx is the BufferedLoop.
// Per-thread scratch: the ctx is shared by every thread running the loop (M8),
// so the buffers cannot live in it. Room for every input at the widest dtype.
inline std::byte* scratch(int k) {
    static constexpr std::int64_t kSlot = kBufferElements * 16;  // complex128
    thread_local std::unique_ptr<std::byte[]> mem =
        std::make_unique_for_overwrite<std::byte[]>(static_cast<size_t>(kSlot * kMaxOperands));
    return mem.get() + k * kSlot;
}

inline void buffered_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const BufferedLoop& b = *static_cast<const BufferedLoop*>(ctx);
    std::array<std::byte*, kMaxOperands> p;
    std::array<std::int64_t, kMaxOperands> st;
    for (std::int64_t off = 0; off < n; off += kBufferElements) {
        const std::int64_t m = std::min(kBufferElements, n - off);
        for (int k = 0; k < b.nop; ++k) {
            if (k < b.nin && b.cast[k]) {
                std::byte* buf = scratch(k);
                std::byte* io[] = {args[k] + off * s[k], buf};
                const std::int64_t cs[] = {s[k], b.item[k]};
                b.cast[k](io, m, cs, nullptr);  // chunk -> buffer, in the loop dtype
                p[k] = buf;
                st[k] = b.item[k];
            } else {
                p[k] = args[k] + off * s[k];
                st[k] = s[k];
            }
        }
        b.inner(p.data(), m, st.data(), nullptr);
    }
}

}  // namespace sd
