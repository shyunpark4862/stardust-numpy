#include "core/buffer.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace sd {

std::shared_ptr<Buffer> Buffer::allocate(std::int64_t nbytes, bool zeroed) {
    auto buf = std::make_shared<Buffer>(Key{});  // the Buffer and its reference count, together
    if (static_cast<std::size_t>(nbytes) <= kInline) {
        buf->data_ = buf->inline_;
        if (zeroed) std::memset(buf->inline_, 0, static_cast<std::size_t>(nbytes));
        return buf;
    }
    // Over-allocate by one alignment unit and round the start up, so the
    // calloc fast path above still applies (aligned_alloc has no calloc twin).
    const std::size_t request = static_cast<std::size_t>(nbytes) + kAlignment;
    void* raw = zeroed ? std::calloc(1, request) : std::malloc(request);
    if (raw == nullptr) throw std::bad_alloc();
    auto addr = reinterpret_cast<std::uintptr_t>(raw);
    auto aligned = (addr + kAlignment - 1) & ~(std::uintptr_t{kAlignment} - 1);
    buf->raw_ = raw;
    buf->data_ = reinterpret_cast<std::byte*>(aligned);
    return buf;
}

std::shared_ptr<Buffer> Buffer::adopt(const Py_buffer& view) {
    auto buf = std::make_shared<Buffer>(Key{});
    buf->external_ = true;
    buf->view_ = view;
    buf->data_ = static_cast<std::byte*>(view.buf);
    return buf;
}

Buffer::~Buffer() {
    if (external_) {
        // The last owner may be dropped where the GIL is not held (later,
        // worker threads); releasing a Py_buffer touches Python objects.
        PyGILState_STATE gil = PyGILState_Ensure();
        PyBuffer_Release(&view_);
        PyGILState_Release(gil);
    } else {
        std::free(raw_);  // null for inline data: a no-op
    }
}

}  // namespace sd
