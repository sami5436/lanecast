// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
//
// Rows are consumed in 16 byte blocks. The last partial block loads past the
// visible end (row padding guarantee, kernels.hpp) and masks the extra lanes.
#include <arm_neon.h>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {
namespace {

// Lanes below `rem` (0..15) set, the rest clear.
inline uint8x16_t tail_mask(int rem) {
  static const std::uint8_t kIota[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return vcltq_u8(vld1q_u8(kIota), vdupq_n_u8(static_cast<std::uint8_t>(rem)));
}

// Squared |a - b| of 16 bytes, pairwise-accumulated into four u32 lanes. A
// lane gains at most 4 * 65025 per call, so callers flush every 8192 calls.
inline uint32x4_t sse_block(uint32x4_t acc, uint8x16_t d) {
  acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
  return vpadalq_u16(acc, vmull_high_u8(d, d));
}

std::uint64_t sse_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t total = 0;
  while (count >= 16) {
    const int n = count / 16 < 8192 ? count / 16 : 8192;
    uint32x4_t acc = vdupq_n_u32(0);
    for (int i = 0; i < n; ++i, a += 16, b += 16) {
      acc = sse_block(acc, vabdq_u8(vld1q_u8(a), vld1q_u8(b)));
    }
    total += vaddlvq_u32(acc);
    count -= n * 16;
  }
  if (count > 0) {
    const uint8x16_t d = vandq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), tail_mask(count));
    total += vaddlvq_u32(sse_block(vdupq_n_u32(0), d));
  }
  return total;
}

}  // namespace

void fill_neon(Table& t) { t.sse = &sse_row; }

}  // namespace lanecast::kernels::metrics
