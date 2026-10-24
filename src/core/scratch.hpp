// Thread-local, grow-only scratch buffers so ops do not allocate per frame.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lanecast::detail {

inline constexpr int kScratchSlots = 4;

// Returns a 64-byte aligned buffer of at least `bytes` + kRowPadding bytes,
// owned by the calling thread and valid until the next call with the same
// slot on this thread. Contents are unspecified. Allocates only when the
// slot must grow. Returns nullptr on allocation failure.
//
// Slot convention, so nested users never collide:
//   0: scale intermediate rows      1: fused strip buffers (luma)
//   2: fused strip buffers (chroma) 3: filters (blur rows)
std::uint8_t* thread_scratch(int slot, std::size_t bytes) noexcept;

}  // namespace lanecast::detail
