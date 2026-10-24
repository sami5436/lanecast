// NEON yuv420p / nv12 -> packed RGB nearest kernels with fast precision.
// Arithmetic follows the 16-bit contract in convert.hpp exactly: vqrdmulh is
// the rounding high-half multiply, and no sum leaves the int16 range.
//
// Blocks are 32 pixels. Luma is loaded de-interleaved (even and odd pixels),
// so chroma sample k lines up with lane k of both and is never duplicated;
// the narrowed channels are interleaved back into pixel order at the end.
#include <arm_neon.h>

#include <cstring>

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

// Chroma terms (bias included) of 8 chroma samples.
struct Terms {
  int16x8_t r, g, b;
};

inline Terms chroma_terms(uint8x8_t u8, uint8x8_t v8, const Coef& c) {
  const int16x8_t u = scaled(u8), v = scaled(v8);
  return {vaddq_s16(vqrdmulhq_n_s16(v, c.crv), c.br),
          vaddq_s16(vaddq_s16(vqrdmulhq_n_s16(u, c.ncgu), vqrdmulhq_n_s16(v, c.ncgv)), c.bg),
          vaddq_s16(vqrdmulhq_n_s16(u, c.cbu), c.bb)};
}

// Luma term of 16 pixels as two int16 vectors.
struct Luma {
  int16x8_t lo, hi;
};

inline Luma luma_term(uint8x16_t y, const Coef& c) {
  return {vqrdmulhq_n_s16(scaled(vget_low_u8(y)), c.cy),
          vqrdmulhq_n_s16(vreinterpretq_s16_u16(vshll_high_n_u8(y, 7)), c.cy)};
}

// 32 pixels at dst. ye holds the even pixels (0, 2, ..., 30), yo the odd
// ones; u and v are the 16 chroma samples they share.
template <int P>
inline void block32(uint8x16_t ye, uint8x16_t yo, uint8x16_t u, uint8x16_t v, std::uint8_t* dst,
                    const Coef& c) {
  const Terms t0 = chroma_terms(vget_low_u8(u), vget_low_u8(v), c);
  const Terms t1 = chroma_terms(vget_high_u8(u), vget_high_u8(v), c);
  const Luma le = luma_term(ye, c), lo = luma_term(yo, c);
  Planar16 first, second;
  const auto chan = [&](int16x8_t a0, int16x8_t a1, uint8x16_t& f, uint8x16_t& s) {
    const uint8x16_t e = narrow(vaddq_s16(le.lo, a0), vaddq_s16(le.hi, a1));
    const uint8x16_t o = narrow(vaddq_s16(lo.lo, a0), vaddq_s16(lo.hi, a1));
    f = vzip1q_u8(e, o);
    s = vzip2q_u8(e, o);
  };
  chan(t0.r, t1.r, first.r, second.r);
  chan(t0.g, t1.g, first.g, second.g);
  chan(t0.b, t1.b, first.b, second.b);
  first.a = second.a = vdupq_n_u8(255);
  store16<P>(dst, first);
  store16<P>(dst + 16 * Layout<P>::bpp, second);
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

template <int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  nearest_row<P>(dst, width, cf, [&](int x, std::uint8_t* out, const Coef& c) {
    const uint8x16x2_t yy = vld2q_u8(y + x);
    block32<P>(yy.val[0], yy.val[1], vld1q_u8(u + x / 2), vld1q_u8(v + x / 2), out, c);
  });
}

template <int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& cf) {
  nearest_row<P>(dst, width, cf, [&](int x, std::uint8_t* out, const Coef& c) {
    const uint8x16x2_t yy = vld2q_u8(y + x), cc = vld2q_u8(uv + x);
    block32<P>(yy.val[0], yy.val[1], cc.val[0], cc.val[1], out, c);
  });
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
