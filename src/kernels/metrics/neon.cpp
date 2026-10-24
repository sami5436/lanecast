// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
//
// Rows are consumed in 16 byte blocks. The last partial block loads past the
// visible end (row padding guarantee, kernels.hpp) and masks the extra lanes.
#include <arm_neon.h>

#include <cstddef>

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

std::uint64_t sad_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t total = 0;
  while (count >= 16) {
    // A u16 lane gains at most 510 per block, so 128 blocks cannot wrap.
    const int n = count / 16 < 128 ? count / 16 : 128;
    uint16x8_t acc = vdupq_n_u16(0);
    for (int i = 0; i < n; ++i, a += 16, b += 16) {
      acc = vpadalq_u8(acc, vabdq_u8(vld1q_u8(a), vld1q_u8(b)));
    }
    total += vaddlvq_u16(acc);
    count -= n * 16;
  }
  if (count > 0) {
    const uint8x16_t d = vandq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), tail_mask(count));
    total += vaddlvq_u8(d);
  }
  return total;
}

// Sums for 4 adjacent 4x4 blocks (16 columns): lane k of val[0..3] holds
// {sum a, sum b, sum a^2 + b^2, sum a*b} of block k.
inline uint32x4x4_t ssim_group(const std::uint8_t* a, std::ptrdiff_t a_stride,
                               const std::uint8_t* b, std::ptrdiff_t b_stride) {
  uint16x8_t s1 = vdupq_n_u16(0), s2 = vdupq_n_u16(0);
  uint32x4_t ss_lo = vdupq_n_u32(0), ss_hi = vdupq_n_u32(0);
  uint32x4_t ab_lo = vdupq_n_u32(0), ab_hi = vdupq_n_u32(0);
  for (int y = 0; y < 4; ++y, a += a_stride, b += b_stride) {
    const uint8x16_t va = vld1q_u8(a);
    const uint8x16_t vb = vld1q_u8(b);
    s1 = vpadalq_u8(s1, va);
    s2 = vpadalq_u8(s2, vb);
    ss_lo = vpadalq_u16(vpadalq_u16(ss_lo, vmull_u8(vget_low_u8(va), vget_low_u8(va))),
                        vmull_u8(vget_low_u8(vb), vget_low_u8(vb)));
    ss_hi = vpadalq_u16(vpadalq_u16(ss_hi, vmull_high_u8(va, va)), vmull_high_u8(vb, vb));
    ab_lo = vpadalq_u16(ab_lo, vmull_u8(vget_low_u8(va), vget_low_u8(vb)));
    ab_hi = vpadalq_u16(ab_hi, vmull_high_u8(va, vb));
  }
  return {{vpaddlq_u16(s1), vpaddlq_u16(s2), vpaddq_u32(ss_lo, ss_hi), vpaddq_u32(ab_lo, ab_hi)}};
}

void ssim_4x4_row(const std::uint8_t* a, std::ptrdiff_t a_stride, const std::uint8_t* b,
                  std::ptrdiff_t b_stride, int blocks, std::int32_t (*sums)[4]) {
  int i = 0;
  for (; i + 4 <= blocks; i += 4) {
    vst4q_u32(reinterpret_cast<std::uint32_t*>(sums[i]),
              ssim_group(a + 4 * i, a_stride, b + 4 * i, b_stride));
  }
  if (i < blocks) {
    // sums holds exactly `blocks` entries: stage the last group.
    std::uint32_t tmp[16];
    vst4q_u32(tmp, ssim_group(a + 4 * i, a_stride, b + 4 * i, b_stride));
    for (int k = 0; i + k < blocks; ++k) {
      for (int j = 0; j < 4; ++j) sums[i + k][j] = static_cast<std::int32_t>(tmp[4 * k + j]);
    }
  }
}

}  // namespace

void fill_neon(Table& t) {
  t.sse = &sse_row;
  t.sad = &sad_row;
  t.ssim_4x4 = &ssim_4x4_row;
}

}  // namespace lanecast::kernels::metrics
