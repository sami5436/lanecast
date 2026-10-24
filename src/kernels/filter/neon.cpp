// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
//
// Every kernel works in blocks of 16 elements and relies on the row padding
// guarantee (kernels.hpp): the last block may read and write past the visible
// width instead of running a scalar tail.
#include <arm_neon.h>

#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {
namespace {

// (x + 128 + ((x + 128) >> 8)) >> 8 per u16 lane, narrowed to u8.
inline uint8x8_t div255(uint16x8_t x) { return vraddhn_u16(x, vrshrq_n_u16(x, 8)); }

inline uint8x16_t div255_pair(uint16x8_t lo, uint16x8_t hi) {
  return vcombine_u8(div255(lo), div255(hi));
}

void blend_row(const std::uint8_t* over, std::uint8_t* base, int pixels) {
  for (int i = 0; i < pixels; i += 16, over += 64, base += 64) {
    const uint8x16x4_t o = vld4q_u8(over);
    uint8x16x4_t b = vld4q_u8(base);
    const uint8x16_t a = o.val[3];
    const uint8x16_t na = vmvnq_u8(a);
    for (int c = 0; c < 3; ++c) {
      const uint16x8_t lo = vmlal_u8(vmull_u8(vget_low_u8(o.val[c]), vget_low_u8(a)),
                                     vget_low_u8(b.val[c]), vget_low_u8(na));
      const uint16x8_t hi = vmlal_high_u8(vmull_high_u8(o.val[c], a), b.val[c], na);
      b.val[c] = div255_pair(lo, hi);
    }
    const uint16x8_t lo = vmull_u8(vget_low_u8(b.val[3]), vget_low_u8(na));
    const uint16x8_t hi = vmull_high_u8(b.val[3], na);
    b.val[3] = vaddq_u8(a, div255_pair(lo, hi));
    vst4q_u8(base, b);
  }
}

}  // namespace

void fill_neon(Table& t) {
  t.blend = &blend_row;
}

}  // namespace lanecast::kernels::filter
