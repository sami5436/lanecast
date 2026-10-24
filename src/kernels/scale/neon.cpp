// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
//
// Horizontal pass: gather-based kernels for the common tap counts (2-tap
// bilinear and 2x area for 1, 3 and 4 channels), a widening multiply-add
// over the taps for everything else. Vertical pass: 16 samples per step,
// two-tap case unrolled. All of them use whole-vector loads and stores into
// the 64 byte row padding instead of scalar tails, and match the scalar
// reference bit for bit (int32 accumulation, one rounding per pass).
#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

// ---- horizontal ----------------------------------------------------------

// Eight outputs of a 1 channel, 2 tap row. Each lane gathers the two adjacent
// source bytes of one output as a u16, then bytes are split into tap 0 / tap 1.
inline void h1_t2_block(const std::uint8_t* src, const std::int32_t* st, const std::int16_t* w,
                        std::int16_t* mid) {
  uint16x8_t pair = vdupq_n_u16(0);
#define LC_PAIR(j) \
  pair = vld1q_lane_u16(reinterpret_cast<const std::uint16_t*>(src + st[j]), pair, j)
  LC_PAIR(0);
  LC_PAIR(1);
  LC_PAIR(2);
  LC_PAIR(3);
  LC_PAIR(4);
  LC_PAIR(5);
  LC_PAIR(6);
  LC_PAIR(7);
#undef LC_PAIR
  const int16x8x2_t wt = vld2q_s16(w);
  const int16x8_t a = vreinterpretq_s16_u16(vmovl_u8(vmovn_u16(pair)));
  const int16x8_t b = vreinterpretq_s16_u16(vmovl_u8(vshrn_n_u16(pair, 8)));
  int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(wt.val[0]));
  lo = vmlal_s16(lo, vget_low_s16(b), vget_low_s16(wt.val[1]));
  int32x4_t hi = vmull_high_s16(a, wt.val[0]);
  hi = vmlal_high_s16(hi, b, wt.val[1]);
  vst1q_s16(mid, vcombine_s16(vrshrn_n_s32(lo, 7), vrshrn_n_s32(hi, 7)));
}

void h1_t2(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  int x = 0;
  for (; x + 8 <= n; x += 8) h1_t2_block(src, f.start + x, f.weights + 2 * x, mid + x);
  if (x < n) {
    std::int32_t st[8];
    for (int j = 0; j < 8; ++j) st[j] = f.start[x + j < n ? x + j : n - 1];
    h1_t2_block(src, st, f.weights + 2 * x, mid + x);
  }
}

// 1 channel, taps >= 3: widening multiply-add along the taps, 8 then 4 at a
// time (weight rows are zero padded to a multiple of 4).
void h1_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  const int stride = f.stride;
  for (int x = 0; x < n; ++x) {
    const std::uint8_t* s = src + f.start[x];
    const std::int16_t* w = f.weights + static_cast<std::ptrdiff_t>(x) * stride;
    int32x4_t acc = vdupq_n_s32(0);
    int k = 0;
    for (; k + 8 <= stride; k += 8) {
      const int16x8_t v = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s + k)));
      const int16x8_t wk = vld1q_s16(w + k);
      acc = vmlal_s16(acc, vget_low_s16(v), vget_low_s16(wk));
      acc = vmlal_high_s16(acc, v, wk);
    }
    if (k < stride) {
      const int16x4_t v = vreinterpret_s16_u16(
          vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(
              reinterpret_cast<const std::uint32_t*>(s + k))))));
      acc = vmlal_s16(acc, v, vld1_s16(w + k));
    }
    mid[x] = static_cast<std::int16_t>((vaddvq_s32(acc) + 64) >> 7);
  }
}

// Interleaved channels (C = 2..4), any taps: each tap loads one pixel as 4
// bytes and accumulates it per channel lane. Extra lanes are overwritten by
// the next output or land in the row padding.
template <int C>
void hc_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  for (int x = 0; x < n; ++x) {
    const std::uint8_t* s = src + static_cast<std::ptrdiff_t>(f.start[x]) * C;
    const std::int16_t* w = f.weights + static_cast<std::ptrdiff_t>(x) * f.stride;
    int32x4_t acc = vdupq_n_s32(0);
    for (int k = 0; k < f.taps; ++k) {
      const int16x4_t v = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(
          vreinterpret_u8_u32(vld1_dup_u32(reinterpret_cast<const std::uint32_t*>(s + k * C))))));
      acc = vmlal_n_s16(acc, v, w[k]);
    }
    vst1_s16(mid + x * C, vrshrn_n_s32(acc, 7));
  }
}

// 4 channels, 2 taps: both pixels are one 8 byte load.
void h4_t2(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  for (int x = 0; x < n; ++x) {
    const int16x8_t v =
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + static_cast<std::ptrdiff_t>(f.start[x]) * 4)));
    int32x4_t acc = vmull_n_s16(vget_low_s16(v), f.weights[2 * x]);
    acc = vmlal_n_s16(acc, vget_high_s16(v), f.weights[2 * x + 1]);
    vst1_s16(mid + x * 4, vrshrn_n_s32(acc, 7));
  }
}

// 3 channels, 2 taps: six bytes of one 8 byte load; tap 1 is lanes 3..5.
void h3_t2(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  for (int x = 0; x < n; ++x) {
    const int16x8_t v =
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + static_cast<std::ptrdiff_t>(f.start[x]) * 3)));
    const int16x4_t lo = vget_low_s16(v);
    const int16x4_t b = vext_s16(lo, vget_high_s16(v), 3);
    int32x4_t acc = vmull_n_s16(lo, f.weights[2 * x]);
    acc = vmlal_n_s16(acc, b, f.weights[2 * x + 1]);
    vst1_s16(mid + x * 3, vrshrn_n_s32(acc, 7));
  }
}

void hscale1(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 2) return h1_t2(src, mid, n, f);
  if (f.taps == 1) {  // 1 pixel wide source: mid = src * 128
    for (int x = 0; x < n; ++x) mid[x] = static_cast<std::int16_t>(src[f.start[x]] << 7);
    return;
  }
  h1_taps(src, mid, n, f);
}

void hscale2(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  hc_taps<2>(src, mid, n, f);
}
void hscale3(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 2) return h3_t2(src, mid, n, f);
  hc_taps<3>(src, mid, n, f);
}
void hscale4(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 2) return h4_t2(src, mid, n, f);
  hc_taps<4>(src, mid, n, f);
}

// ---- vertical ------------------------------------------------------------

// (acc + 2^20) >> 21 for four lanes, narrowed with saturation to u16.
inline uint16x4_t round_narrow(int32x4_t acc) { return vqmovun_s32(vrshrq_n_s32(acc, 21)); }

inline uint8x16_t pack16(int32x4_t a0, int32x4_t a1, int32x4_t a2, int32x4_t a3) {
  return vcombine_u8(vqmovn_u16(vcombine_u16(round_narrow(a0), round_narrow(a1))),
                     vqmovn_u16(vcombine_u16(round_narrow(a2), round_narrow(a3))));
}

void vscale(const std::int16_t* const* rows, const std::int16_t* w, int taps, std::uint8_t* dst,
            int count) {
  if (taps == 2) {
    const std::int16_t* r0 = rows[0];
    const std::int16_t* r1 = rows[1];
    const std::int16_t w0 = w[0];
    const std::int16_t w1 = w[1];
    for (int i = 0; i < count; i += 16) {
      const int16x8_t a0 = vld1q_s16(r0 + i), a1 = vld1q_s16(r0 + i + 8);
      const int16x8_t b0 = vld1q_s16(r1 + i), b1 = vld1q_s16(r1 + i + 8);
      int32x4_t c0 = vmull_n_s16(vget_low_s16(a0), w0);
      int32x4_t c1 = vmull_high_n_s16(a0, w0);
      int32x4_t c2 = vmull_n_s16(vget_low_s16(a1), w0);
      int32x4_t c3 = vmull_high_n_s16(a1, w0);
      c0 = vmlal_n_s16(c0, vget_low_s16(b0), w1);
      c1 = vmlal_high_n_s16(c1, b0, w1);
      c2 = vmlal_n_s16(c2, vget_low_s16(b1), w1);
      c3 = vmlal_high_n_s16(c3, b1, w1);
      vst1q_u8(dst + i, pack16(c0, c1, c2, c3));
    }
    return;
  }
  for (int i = 0; i < count; i += 16) {
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < taps; ++k) {
      const int16x8_t a = vld1q_s16(rows[k] + i), b = vld1q_s16(rows[k] + i + 8);
      const std::int16_t wk = w[k];
      c0 = vmlal_n_s16(c0, vget_low_s16(a), wk);
      c1 = vmlal_high_n_s16(c1, a, wk);
      c2 = vmlal_n_s16(c2, vget_low_s16(b), wk);
      c3 = vmlal_high_n_s16(c3, b, wk);
    }
    vst1q_u8(dst + i, pack16(c0, c1, c2, c3));
  }
}

}  // namespace

void fill_neon(Table& t) {
  t.hscale[1] = &hscale1;
  t.hscale[2] = &hscale2;
  t.hscale[3] = &hscale3;
  t.hscale[4] = &hscale4;
  t.vscale = &vscale;
}

}  // namespace lanecast::kernels::scale
