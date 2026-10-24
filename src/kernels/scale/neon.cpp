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

// 1 channel, taps >= 3, four outputs per step. Per group of four taps, a
// u32 lane load gathers each output's four source bytes; widened and
// multiplied against that output's weights they accumulate into one s32x4 per
// output, so the groups of any tap count add up without leaving the vector
// domain. Two levels of pairwise adds then reduce the four accumulators to one
// sum per output. G > 0 fixes the group count at compile time (taps <= 4G).
// Rows are padded to a multiple of four weights; the last block may run past
// n by reusing the last output, its extra lanes land in the row padding.
template <int G>
void h1_groups(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  const int groups = G ? G : f.stride / 4;
  for (int x = 0; x < n; x += 4) {
    const std::uint8_t* s[4];
    const std::int16_t* w[4];
    for (int j = 0; j < 4; ++j) {
      const int xj = x + j < n ? x + j : n - 1;
      s[j] = src + f.start[xj];
      w[j] = f.weights + static_cast<std::ptrdiff_t>(xj) * f.stride;
    }
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (int g = 0; g < groups; ++g) {
      uint32x4_t b = vdupq_n_u32(0);
      b = vld1q_lane_u32(reinterpret_cast<const std::uint32_t*>(s[0] + 4 * g), b, 0);
      b = vld1q_lane_u32(reinterpret_cast<const std::uint32_t*>(s[1] + 4 * g), b, 1);
      b = vld1q_lane_u32(reinterpret_cast<const std::uint32_t*>(s[2] + 4 * g), b, 2);
      b = vld1q_lane_u32(reinterpret_cast<const std::uint32_t*>(s[3] + 4 * g), b, 3);
      const uint8x16_t bytes = vreinterpretq_u8_u32(b);
      const int16x8_t v01 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes)));
      const int16x8_t v23 = vreinterpretq_s16_u16(vmovl_high_u8(bytes));
      const int16x8_t w01 = vcombine_s16(vld1_s16(w[0] + 4 * g), vld1_s16(w[1] + 4 * g));
      const int16x8_t w23 = vcombine_s16(vld1_s16(w[2] + 4 * g), vld1_s16(w[3] + 4 * g));
      acc0 = vmlal_s16(acc0, vget_low_s16(v01), vget_low_s16(w01));
      acc1 = vmlal_high_s16(acc1, v01, w01);
      acc2 = vmlal_s16(acc2, vget_low_s16(v23), vget_low_s16(w23));
      acc3 = vmlal_high_s16(acc3, v23, w23);
    }
    const int32x4_t sums = vpaddq_s32(vpaddq_s32(acc0, acc1), vpaddq_s32(acc2, acc3));
    vst1_s16(mid + x, vrshrn_n_s32(sums, 7));
  }
}

void h1_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.stride == 4) return h1_groups<1>(src, mid, n, f);
  if (f.stride == 8) return h1_groups<2>(src, mid, n, f);
  h1_groups<0>(src, mid, n, f);
}

// 1 channel, 2..6 taps, eight outputs per block from the FilterView block
// tables: two 16 byte loads cover the block's source window, one table lookup
// yields two taps of all eight outputs, and tap-major weights make each
// accumulate a plain vector multiply-add. Outputs past the table blocks go to
// the gather kernels.
template <int T>
void h1_tbl(const std::uint8_t* src, std::int16_t* mid, const FilterView& f) {
  const std::uint8_t* ti = f.tbl;
  const std::int16_t* wt = f.wt;
  for (int b = 0; b < f.tbl_blocks; ++b) {
    const std::uint8_t* s = src + f.start[b * 8];
    const uint8x16x2_t window = {{vld1q_u8(s), vld1q_u8(s + 16)}};
    int32x4_t lo = vdupq_n_s32(0), hi = lo;
    for (int p = 0; p < (T + 1) / 2; ++p) {
      const uint8x16_t g = vqtbl2q_u8(window, vld1q_u8(ti));
      ti += 16;
      const int16x8_t v0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g)));
      const int16x8_t w0 = vld1q_s16(wt);
      wt += 8;
      lo = vmlal_s16(lo, vget_low_s16(v0), vget_low_s16(w0));
      hi = vmlal_high_s16(hi, v0, w0);
      if (2 * p + 1 < T) {
        const int16x8_t v1 = vreinterpretq_s16_u16(vmovl_high_u8(g));
        const int16x8_t w1 = vld1q_s16(wt);
        wt += 8;
        lo = vmlal_s16(lo, vget_low_s16(v1), vget_low_s16(w1));
        hi = vmlal_high_s16(hi, v1, w1);
      }
    }
    vst1q_s16(mid + b * 8, vcombine_s16(vrshrn_n_s32(lo, 7), vrshrn_n_s32(hi, 7)));
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
  if (f.tbl_blocks > 0) {
    switch (f.taps) {
      case 2: h1_tbl<2>(src, mid, f); break;
      case 3: h1_tbl<3>(src, mid, f); break;
      case 4: h1_tbl<4>(src, mid, f); break;
      case 5: h1_tbl<5>(src, mid, f); break;
      default: h1_tbl<6>(src, mid, f); break;
    }
    const int done = f.tbl_blocks * 8;
    if (done == n) return;
    const FilterView rest{f.taps, f.stride, f.start + done,
                          f.weights + static_cast<std::ptrdiff_t>(done) * f.stride};
    return hscale1(src, mid + done, n - done, rest);
  }
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

// Fixed tap count: row pointers and weights stay in registers and the tap loop
// unrolls, 16 samples per step.
template <int T>
void vscale_t(const std::int16_t* const* rows, const std::int16_t* w, std::uint8_t* dst,
              int count) {
  const std::int16_t* r[T];
  std::int16_t wk[T];
  for (int k = 0; k < T; ++k) {
    r[k] = rows[k];
    wk[k] = w[k];
  }
  for (int i = 0; i < count; i += 16) {
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < T; ++k) {
      const int16x8_t a = vld1q_s16(r[k] + i), b = vld1q_s16(r[k] + i + 8);
      c0 = vmlal_n_s16(c0, vget_low_s16(a), wk[k]);
      c1 = vmlal_high_n_s16(c1, a, wk[k]);
      c2 = vmlal_n_s16(c2, vget_low_s16(b), wk[k]);
      c3 = vmlal_high_n_s16(c3, b, wk[k]);
    }
    vst1q_u8(dst + i, pack16(c0, c1, c2, c3));
  }
}

void vscale(const std::int16_t* const* rows, const std::int16_t* w, int taps, std::uint8_t* dst,
            int count) {
  switch (taps) {
    case 3: return vscale_t<3>(rows, w, dst, count);
    case 4: return vscale_t<4>(rows, w, dst, count);
    case 5: return vscale_t<5>(rows, w, dst, count);
    case 6: return vscale_t<6>(rows, w, dst, count);
    default: break;
  }
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
