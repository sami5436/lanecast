#include "core/scratch.hpp"

#include <new>

#include "lanecast/frame.hpp"

namespace lanecast::detail {
namespace {

struct AlignedBuffer {
  std::uint8_t* data = nullptr;
  std::size_t capacity = 0;
  ~AlignedBuffer() { release(); }
  void release() noexcept {
    if (data != nullptr) ::operator delete[](data, std::align_val_t{kFrameAlignment});
    data = nullptr;
    capacity = 0;
  }
};

thread_local AlignedBuffer t_slots[kScratchSlots];

}  // namespace

std::uint8_t* thread_scratch(int slot, std::size_t bytes) noexcept {
  if (slot < 0 || slot >= kScratchSlots) return nullptr;
  AlignedBuffer& b = t_slots[slot];
  const std::size_t need = bytes + kRowPadding;
  if (b.capacity < need) {
    b.release();
    const std::size_t cap = (need + need / 2 + kFrameAlignment - 1) & ~(kFrameAlignment - 1);
    b.data = static_cast<std::uint8_t*>(
        ::operator new[](cap, std::align_val_t{kFrameAlignment}, std::nothrow));
    if (b.data == nullptr) return nullptr;
    b.capacity = cap;
  }
  return b.data;
}

}  // namespace lanecast::detail
