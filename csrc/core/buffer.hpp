// A reference-counted block of memory that array data lives in
// (ARCHITECTURE.md §3.2). Several arrays (views) may share one Buffer.
#pragma once

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace sd {

class Buffer {
public:
    // One cache line on Apple M4. Aligning the start of every allocation to a
    // cache line means a SIMD loop never straddles two lines at its first load.
    static constexpr std::size_t kAlignment = 128;

    // Fresh memory owned by this Buffer. `zeroed` uses calloc: for large
    // sizes the OS hands out pages that are zero-filled lazily on first
    // touch, so zeros((10000, 10000)) costs almost nothing until written.
    static std::shared_ptr<Buffer> allocate(std::int64_t nbytes, bool zeroed);

    // Memory owned by another Python object, obtained through the buffer
    // protocol. Takes ownership of `view` and releases it on destruction.
    static std::shared_ptr<Buffer> adopt(const Py_buffer& view);

    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::byte* data() const { return data_; }

    // Arrays of up to kInline bytes keep their data inside the Buffer object
    // itself, which make_shared allocates together with the reference count:
    // one allocation for a small array instead of three (M7.7).
    static constexpr std::size_t kInline = 256;

    struct Key {  // lets make_shared call the constructor; nobody else can
        explicit Key() = default;
    };
    explicit Buffer(Key) {}

private:
    void* raw_ = nullptr;  // pointer returned by malloc/calloc (own memory only)
    std::byte* data_ = nullptr;
    bool external_ = false;
    Py_buffer view_{};
    alignas(kAlignment) std::byte inline_[kInline];
};

}  // namespace sd
