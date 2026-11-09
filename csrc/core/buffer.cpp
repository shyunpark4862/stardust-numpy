#include "core/buffer.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>

namespace sd {

std::shared_ptr<Buffer> Buffer::allocate(std::int64_t nbytes, bool zeroed) {
    // Over-allocate by one alignment unit and round the start up, so the
    // calloc fast path above still applies (aligned_alloc has no calloc twin).
    const std::size_t request = static_cast<std::size_t>(nbytes) + kAlignment;
    void* raw = zeroed ? std::calloc(1, request) : std::malloc(request);
    if (raw == nullptr) throw std::bad_alloc();

    auto addr = reinterpret_cast<std::uintptr_t>(raw);
    auto aligned = (addr + kAlignment - 1) & ~(std::uintptr_t{kAlignment} - 1);

    std::shared_ptr<Buffer> buf(new Buffer());
    buf->raw_ = raw;
    buf->data_ = reinterpret_cast<std::byte*>(aligned);
    return buf;
}

std::shared_ptr<Buffer> Buffer::adopt(const Py_buffer& view) {
    std::shared_ptr<Buffer> buf(new Buffer());
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
        std::free(raw_);
    }
}

}  // namespace sd
