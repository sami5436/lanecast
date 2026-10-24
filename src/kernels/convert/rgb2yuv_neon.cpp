// NEON packed RGB -> yuv420p / nv12 / gray row kernels. Arithmetic follows the
// Q15 contract in convert.hpp: int32 accumulation, one rounding shift, and
// saturation to 8 bits (vqrshrun / vqmovun).
#include <arm_neon.h>

#include <cstring>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/neon_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// ky lanes: yr, yg, yb. kc lanes: ur, ug, ub, vr, vg, vb.
struct Coef {
  int16x4_t ky;
  int16x8_t kc;
  std::int32_t y_bias;  // y_offset << 15
};

inline Coef make_coef(const RgbToYuvCoeffs& k) {
  const std::int16_t y[4] = {k.yr, k.yg, k.yb, 0};
  const std::int16_t c[8] = {k.ur, k.ug, k.ub, k.vr, k.vg, k.vb, 0, 0};
  return {vld1_s16(y), vld1q_s16(c), static_cast<std::int32_t>(k.y_offset) << 15};
}

// Y for 8 pixels: (yr*R + yg*G + yb*B + bias + 16384) >> 15, clamped to u16.
inline uint16x8_t luma8(int16x8_t r, int16x8_t g, int16x8_t b, const Coef& c) {
  const int32x4_t bias = vdupq_n_s32(c.y_bias);
  int32x4_t lo = vmlal_lane_s16(bias, vget_low_s16(r), c.ky, 0);
  int32x4_t hi = vmlal_high_lane_s16(bias, r, c.ky, 0);
  lo = vmlal_lane_s16(lo, vget_low_s16(g), c.ky, 1);
  hi = vmlal_high_lane_s16(hi, g, c.ky, 1);
  lo = vmlal_lane_s16(lo, vget_low_s16(b), c.ky, 2);
  hi = vmlal_high_lane_s16(hi, b, c.ky, 2);
  return vqrshrun_high_n_s32(vqrshrun_n_s32(lo, 15), hi, 15);
}

inline int16x8_t widen_lo(uint8x16_t v) { return vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))); }
inline int16x8_t widen_hi(uint8x16_t v) { return vreinterpretq_s16_u16(vmovl_high_u8(v)); }

inline uint8x16_t luma16(const Planar16& p, const Coef& c) {
  return vqmovn_high_u16(vqmovn_u16(luma8(widen_lo(p.r), widen_lo(p.g), widen_lo(p.b), c)),
                         luma8(widen_hi(p.r), widen_hi(p.g), widen_hi(p.b), c));
}

// One chroma component for 8 samples from the sums. Lanes of kc start at
// Lane0 (0 for U, 3 for V). (k.r*SR + k.g*SG + k.b*SB + (128 << Shift) +
// (1 << (Shift - 1))) >> Shift.
template <int Lane0, int Shift>
inline uint8x8_t chroma8(int16x8_t sr, int16x8_t sg, int16x8_t sb, const int16x8_t kc) {
  const int32x4_t bias = vdupq_n_s32(128 << Shift);
  int32x4_t lo = vmlal_laneq_s16(bias, vget_low_s16(sr), kc, Lane0);
  int32x4_t hi = vmlal_high_laneq_s16(bias, sr, kc, Lane0);
  lo = vmlal_laneq_s16(lo, vget_low_s16(sg), kc, Lane0 + 1);
  hi = vmlal_high_laneq_s16(hi, sg, kc, Lane0 + 1);
  lo = vmlal_laneq_s16(lo, vget_low_s16(sb), kc, Lane0 + 2);
  hi = vmlal_high_laneq_s16(hi, sb, kc, Lane0 + 2);
  // vqrshrun_n_s32 only shifts by up to 16, so round with vrshr first.
  return vqmovn_u16(vcombine_u16(vqmovun_s32(vrshrq_n_s32(lo, Shift)),
                                 vqmovun_s32(vrshrq_n_s32(hi, Shift))));
}

// Sum of each horizontal pixel pair over both rows, as s16.
inline int16x8_t sum2x2(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_s16_u16(vpadalq_u8(vpaddlq_u8(a), b));
}

// Previous block's channel vectors (two rows), for the [1 2 1] left filter.
struct Prev {
  uint8x16_t r0, g0, b0, r1, g1, b1;
};

// [1 2 1] taps over both rows, centered on the even columns, as the pair sums
// (x[2i-1] + x[2i]) + (x[2i] + x[2i+1]). pa, pb (the previous block) supply
// x[-1] across blocks; for the first block they repeat x[0] (edge clamp).
inline int16x8_t sum121(uint8x16_t a, uint8x16_t b, uint8x16_t pa, uint8x16_t pb) {
  uint16x8_t acc = vpadalq_u8(vpaddlq_u8(a), b);
  acc = vpadalq_u8(acc, vextq_u8(pa, a, 15));
  acc = vpadalq_u8(acc, vextq_u8(pb, b, 15));
  return vreinterpretq_s16_u16(acc);
}

template <int P, bool Nv12, bool Left>
inline void block(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                  std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v, const Coef& c, Prev& prev,
                  bool first) {
  const Planar16 a = load16<P>(s0), b = load16<P>(s1);
  vst1q_u8(y0, luma16(a, c));
  vst1q_u8(y1, luma16(b, c));
  uint8x8_t cu, cv;
  if constexpr (Left) {
    if (first) {
      prev = {vdupq_laneq_u8(a.r, 0), vdupq_laneq_u8(a.g, 0), vdupq_laneq_u8(a.b, 0),
              vdupq_laneq_u8(b.r, 0), vdupq_laneq_u8(b.g, 0), vdupq_laneq_u8(b.b, 0)};
    }
    const int16x8_t sr = sum121(a.r, b.r, prev.r0, prev.r1);
    const int16x8_t sg = sum121(a.g, b.g, prev.g0, prev.g1);
    const int16x8_t sb = sum121(a.b, b.b, prev.b0, prev.b1);
    prev = {a.r, a.g, a.b, b.r, b.g, b.b};
    cu = chroma8<0, 18>(sr, sg, sb, c.kc);
    cv = chroma8<3, 18>(sr, sg, sb, c.kc);
  } else {
    const int16x8_t sr = sum2x2(a.r, b.r), sg = sum2x2(a.g, b.g), sb = sum2x2(a.b, b.b);
    cu = chroma8<0, 17>(sr, sg, sb, c.kc);
    cv = chroma8<3, 17>(sr, sg, sb, c.kc);
  }
  if constexpr (Nv12) {
    uint8x8x2_t uv;
    uv.val[0] = cu;
    uv.val[1] = cv;
    vst2_u8(u, uv);
  } else {
    vst1_u8(u, cu);
    vst1_u8(v, cv);
  }
}

// Shared row driver. For odd widths the last chroma sample pairs the last
// pixel with itself, so the final block runs on a copy whose tail repeats the
// last pixel instead of reading padding.
template <int P, bool Nv12, bool Left>
void packed_to_yuv_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                       std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v, int width,
                       const RgbToYuvCoeffs& k) {
  constexpr int bpp = Layout<P>::bpp;
  const Coef c = make_coef(k);
  const int step = Nv12 ? 2 : 1;  // chroma bytes per sample in u
  Prev prev{};
  int x = 0;
  const int body = (width & 1) ? width - 16 : width;
  for (; x < body; x += 16) {
    block<P, Nv12, Left>(s0 + x * bpp, s1 + x * bpp, y0 + x, y1 + x, u + (x / 2) * step,
                         v + (x / 2) * step, c, prev, x == 0);
  }
  if (x < width) {
    alignas(16) std::uint8_t t0[16 * bpp], t1[16 * bpp];
    const int rem = width - x;
    std::memcpy(t0, s0 + x * bpp, static_cast<std::size_t>(rem * bpp));
    std::memcpy(t1, s1 + x * bpp, static_cast<std::size_t>(rem * bpp));
    for (int i = rem; i < 16; ++i) {
      std::memcpy(t0 + i * bpp, t0 + (rem - 1) * bpp, bpp);
      std::memcpy(t1 + i * bpp, t1 + (rem - 1) * bpp, bpp);
    }
    block<P, Nv12, Left>(t0, t1, y0 + x, y1 + x, u + (x / 2) * step, v + (x / 2) * step, c, prev,
                         x == 0);
  }
}

template <int P>
void packed_to_yuv420_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                          std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v, int width,
                          const RgbToYuvCoeffs& k, ChromaLocation loc) {
  if (loc == ChromaLocation::center) {
    packed_to_yuv_row<P, false, false>(s0, s1, y0, y1, u, v, width, k);
  } else {
    packed_to_yuv_row<P, false, true>(s0, s1, y0, y1, u, v, width, k);
  }
}

template <int P>
void packed_to_nv12_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                        std::uint8_t* y1, std::uint8_t* uv, int width, const RgbToYuvCoeffs& k,
                        ChromaLocation loc) {
  if (loc == ChromaLocation::center) {
    packed_to_yuv_row<P, true, false>(s0, s1, y0, y1, uv, uv, width, k);
  } else {
    packed_to_yuv_row<P, true, true>(s0, s1, y0, y1, uv, uv, width, k);
  }
}

template <int P>
void packed_to_gray_row(const std::uint8_t* src, std::uint8_t* gray, int width,
                        const RgbToYuvCoeffs& k) {
  const Coef c = make_coef(k);
  for (int x = 0; x < width; x += 16) {
    vst1q_u8(gray + x, luma16(load16<P>(src + x * Layout<P>::bpp), c));
  }
}

}  // namespace

void fill_neon_rgb2yuv(Table& t) {
  t.packed_to_yuv420[kRgb24] = &packed_to_yuv420_row<kRgb24>;
  t.packed_to_yuv420[kRgba] = &packed_to_yuv420_row<kRgba>;
  t.packed_to_yuv420[kBgra] = &packed_to_yuv420_row<kBgra>;
  t.packed_to_nv12[kRgb24] = &packed_to_nv12_row<kRgb24>;
  t.packed_to_nv12[kRgba] = &packed_to_nv12_row<kRgba>;
  t.packed_to_nv12[kBgra] = &packed_to_nv12_row<kBgra>;
  t.packed_to_gray[kRgb24] = &packed_to_gray_row<kRgb24>;
  t.packed_to_gray[kRgba] = &packed_to_gray_row<kRgba>;
  t.packed_to_gray[kBgra] = &packed_to_gray_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
