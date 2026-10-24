// NEON with FEAT_DotProd, compiled with -march=armv8.2-a+dotprod. Runs on top of the neon table.
// Store only the kernels this tier specializes; others are inherited.
//
// Same block layout and tail masking as neon.cpp. UDOT folds four byte
// products into each u32 lane in one instruction.
#include <arm_neon.h>

#include <cstddef>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {
namespace {

inline uint8x16_t tail_mask(int rem) {
  static const std::uint8_t kIota[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return vcltq_u8(vld1q_u8(kIota), vdupq_n_u8(static_cast<std::uint8_t>(rem)));
}

// A u32 lane gains at most 4 * 65025 per block, so 8192 blocks cannot wrap.
std::uint64_t sse_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t total = 0;
  while (count >= 16) {
    const int n = count / 16 < 8192 ? count / 16 : 8192;
    uint32x4_t acc0 = vdupq_n_u32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int i = 0;
    for (; i + 4 <= n; i += 4, a += 64, b += 64) {
      const uint8x16_t d0 = vabdq_u8(vld1q_u8(a), vld1q_u8(b));
      const uint8x16_t d1 = vabdq_u8(vld1q_u8(a + 16), vld1q_u8(b + 16));
      const uint8x16_t d2 = vabdq_u8(vld1q_u8(a + 32), vld1q_u8(b + 32));
      const uint8x16_t d3 = vabdq_u8(vld1q_u8(a + 48), vld1q_u8(b + 48));
      acc0 = vdotq_u32(acc0, d0, d0);
      acc1 = vdotq_u32(acc1, d1, d1);
      acc2 = vdotq_u32(acc2, d2, d2);
      acc3 = vdotq_u32(acc3, d3, d3);
    }
    for (; i < n; ++i, a += 16, b += 16) {
      const uint8x16_t d = vabdq_u8(vld1q_u8(a), vld1q_u8(b));
      acc0 = vdotq_u32(acc0, d, d);
    }
    total += vaddlvq_u32(vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3)));
    count -= n * 16;
  }
  if (count > 0) {
    const uint8x16_t d = vandq_u8(vabdq_u8(vld1q_u8(a), vld1q_u8(b)), tail_mask(count));
    total += vaddlvq_u32(vdotq_u32(vdupq_n_u32(0), d, d));
  }
  return total;
}

// Sums for 4 adjacent 4x4 blocks (16 columns): lane k of val[0..3] holds
// {sum a, sum b, sum a^2 + b^2, sum a*b} of block k. Each UDOT reduces one
// row of a block to a lane.
inline uint32x4x4_t ssim_group(const std::uint8_t* a, std::ptrdiff_t a_stride,
                               const std::uint8_t* b, std::ptrdiff_t b_stride) {
  const uint8x16_t ones = vdupq_n_u8(1);
  uint32x4_t s1 = vdupq_n_u32(0), s2 = s1, ss = s1, ab = s1;
  for (int y = 0; y < 4; ++y, a += a_stride, b += b_stride) {
    const uint8x16_t va = vld1q_u8(a);
    const uint8x16_t vb = vld1q_u8(b);
    s1 = vdotq_u32(s1, va, ones);
    s2 = vdotq_u32(s2, vb, ones);
    ss = vdotq_u32(vdotq_u32(ss, va, va), vb, vb);
    ab = vdotq_u32(ab, va, vb);
  }
  return {{s1, s2, ss, ab}};
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

void fill_neon_dotprod(Table& t) {
  t.sse = &sse_row;
  t.ssim_4x4 = &ssim_4x4_row;
}

}  // namespace lanecast::kernels::metrics
