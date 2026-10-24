// NEON yuv420p / nv12 -> packed RGB nearest kernels with fast precision.
// Arithmetic follows the 16-bit contract in convert.hpp exactly: vqrdmulh is
// the rounding high-half multiply, and no sum leaves the int16 range.
#include <arm_neon.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/neon_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// Rounding high-half multiply, same value as the scalar rmul.
inline int rmul(int a, int k) { return (a * k + 16384) >> 15; }

struct Coef {
  std::int16_t cy, crv, ncgu, ncgv, cbu;
  int16x8_t br, bg, bb;
};

inline Coef make_coef(const YuvToRgbCoeffs& c) {
  const int mid = 128 << 7, ky = rmul(c.y_offset << 7, c.cy);
  return {c.cy,
          c.crv,
          static_cast<std::int16_t>(-c.cgu),
          static_cast<std::int16_t>(-c.cgv),
          c.cbu,
          vdupq_n_s16(static_cast<std::int16_t>(16 - ky - rmul(mid, c.crv))),
          vdupq_n_s16(static_cast<std::int16_t>(16 - ky + rmul(mid, c.cgu) + rmul(mid, c.cgv))),
          vdupq_n_s16(static_cast<std::int16_t>(16 - ky - rmul(mid, c.cbu)))};
}

inline int16x8_t scaled(uint8x8_t v) { return vreinterpretq_s16_u16(vshll_n_u8(v, 7)); }

// Narrows two int16 vectors of sums with 5 fraction bits to 16 clamped bytes.
inline uint8x16_t narrow(int16x8_t lo, int16x8_t hi) {
  return vqshrun_high_n_s16(vqshrun_n_s16(lo, 5), hi, 5);
}

// 16 pixels: y is 16 luma bytes, u8 and v8 the 8 chroma bytes they share.
inline Planar16 yuv16(uint8x16_t y, uint8x8_t u8, uint8x8_t v8, const Coef& c) {
  const int16x8_t ylo = vqrdmulhq_n_s16(scaled(vget_low_u8(y)), c.cy);
  const int16x8_t yhi = vqrdmulhq_n_s16(vreinterpretq_s16_u16(vshll_high_n_u8(y, 7)), c.cy);
  const int16x8_t u = scaled(u8), v = scaled(v8);
  const int16x8_t rc = vaddq_s16(vqrdmulhq_n_s16(v, c.crv), c.br);
  const int16x8_t gc = vaddq_s16(
      vaddq_s16(vqrdmulhq_n_s16(u, c.ncgu), vqrdmulhq_n_s16(v, c.ncgv)), c.bg);
  const int16x8_t bc = vaddq_s16(vqrdmulhq_n_s16(u, c.cbu), c.bb);
  return {narrow(vaddq_s16(ylo, vzip1q_s16(rc, rc)), vaddq_s16(yhi, vzip2q_s16(rc, rc))),
          narrow(vaddq_s16(ylo, vzip1q_s16(gc, gc)), vaddq_s16(yhi, vzip2q_s16(gc, gc))),
          narrow(vaddq_s16(ylo, vzip1q_s16(bc, bc)), vaddq_s16(yhi, vzip2q_s16(bc, bc))),
          vdupq_n_u8(255)};
}

template <int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  const Coef c = make_coef(cf);
  for (int x = 0; x < width; x += 16) {
    store16<P>(dst + x * Layout<P>::bpp,
               yuv16(vld1q_u8(y + x), vld1_u8(u + x / 2), vld1_u8(v + x / 2), c));
  }
}

template <int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& cf) {
  const Coef c = make_coef(cf);
  for (int x = 0; x < width; x += 16) {
    const uint8x8x2_t p = vld2_u8(uv + x);
    store16<P>(dst + x * Layout<P>::bpp, yuv16(vld1q_u8(y + x), p.val[0], p.val[1], c));
  }
}

}  // namespace

void fill_neon_fast(Table& t) {
  t.yuv420_fast_to_packed[kRgb24] = &yuv420_nearest_row<kRgb24>;
  t.yuv420_fast_to_packed[kRgba] = &yuv420_nearest_row<kRgba>;
  t.yuv420_fast_to_packed[kBgra] = &yuv420_nearest_row<kBgra>;
  t.nv12_fast_to_packed[kRgb24] = &nv12_nearest_row<kRgb24>;
  t.nv12_fast_to_packed[kRgba] = &nv12_nearest_row<kRgba>;
  t.nv12_fast_to_packed[kBgra] = &nv12_nearest_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
