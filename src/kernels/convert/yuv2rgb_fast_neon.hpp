// Pieces shared by the NEON fast-precision yuv2rgb translation units (plain
// NEON and the RDM build). Each TU supplies a block policy B:
//   template <int P> static void B::block32(uint8x16_t ye, uint8x16_t yo,
//       uint8x16_t u, uint8x16_t v, std::uint8_t* dst, const Coef& c);
// converting 32 pixels: ye holds the even pixels (0, 2, ..., 30), yo the odd
// ones, u and v the 16 chroma samples they share. Luma is loaded
// de-interleaved so chroma sample k lines up with lane k of both and is never
// duplicated; the narrowed channels are interleaved into pixel order at the
// end. Arithmetic follows the 16-bit contract in convert.hpp.
#pragma once

#include <arm_neon.h>

#include <cstring>

#include "kernels/convert/neon_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// Rounding high-half multiply, same value as the scalar rmul.
inline int rmul(int a, int k) { return (a * k + 16384) >> 15; }

struct Coef {
  int16x8_t k;  // lanes: cy, crv, -cgu, -cgv, cbu
  int16x8_t br, bg, bb;
};

inline Coef make_coef(const YuvToRgbCoeffs& c) {
  const int mid = 128 << 7, ky = rmul(c.y_offset << 7, c.cy);
  const std::int16_t k[8] = {c.cy, c.crv, static_cast<std::int16_t>(-c.cgu),
                             static_cast<std::int16_t>(-c.cgv), c.cbu, 0, 0, 0};
  return {vld1q_s16(k),
          vdupq_n_s16(static_cast<std::int16_t>(16 - ky - rmul(mid, c.crv))),
          vdupq_n_s16(static_cast<std::int16_t>(16 - ky + rmul(mid, c.cgu) + rmul(mid, c.cgv))),
          vdupq_n_s16(static_cast<std::int16_t>(16 - ky - rmul(mid, c.cbu)))};
}

inline int16x8_t scaled(uint8x8_t v) { return vreinterpretq_s16_u16(vshll_n_u8(v, 7)); }
inline int16x8_t scaled_high(uint8x16_t v) {
  return vreinterpretq_s16_u16(vshll_high_n_u8(v, 7));
}

// Narrows two int16 vectors of sums with 5 fraction bits to 16 clamped bytes.
inline uint8x16_t narrow(int16x8_t lo, int16x8_t hi) {
  return vqshrun_high_n_s16(vqshrun_n_s16(lo, 5), hi, 5);
}

// Walks a row in blocks of 32. load(x, out, c) converts the block starting
// at pixel x. The last partial block goes through a bounce buffer: 32 pixels
// at 4 bytes would write more past the row than the padding guarantee allows.
template <int P, class Load>
inline void nearest_row(std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf, Load load) {
  const Coef c = make_coef(cf);
  int x = 0;
  for (; x + 32 <= width; x += 32) load(x, dst + x * Layout<P>::bpp, c);
  if (x < width) {
    alignas(16) std::uint8_t tail[32 * 4];
    load(x, tail, c);
    std::memcpy(dst + x * Layout<P>::bpp, tail,
                static_cast<std::size_t>(width - x) * Layout<P>::bpp);
  }
}

template <class B, int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  nearest_row<P>(dst, width, cf, [&](int x, std::uint8_t* out, const Coef& c) {
    const uint8x16x2_t yy = vld2q_u8(y + x);
    B::template block32<P>(yy.val[0], yy.val[1], vld1q_u8(u + x / 2), vld1q_u8(v + x / 2), out,
                           c);
  });
}

template <class B, int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& cf) {
  nearest_row<P>(dst, width, cf, [&](int x, std::uint8_t* out, const Coef& c) {
    const uint8x16x2_t yy = vld2q_u8(y + x), cc = vld2q_u8(uv + x);
    B::template block32<P>(yy.val[0], yy.val[1], cc.val[0], cc.val[1], out, c);
  });
}

template <class B>
void fill_fast(Table& t) {
  t.yuv420_fast_to_packed[kRgb24] = &yuv420_nearest_row<B, kRgb24>;
  t.yuv420_fast_to_packed[kRgba] = &yuv420_nearest_row<B, kRgba>;
  t.yuv420_fast_to_packed[kBgra] = &yuv420_nearest_row<B, kBgra>;
  t.nv12_fast_to_packed[kRgb24] = &nv12_nearest_row<B, kRgb24>;
  t.nv12_fast_to_packed[kRgba] = &nv12_nearest_row<B, kRgba>;
  t.nv12_fast_to_packed[kBgra] = &nv12_nearest_row<B, kBgra>;
}

}  // namespace
}  // namespace lanecast::kernels::convert
