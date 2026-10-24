// NEON yuv420p / nv12 -> packed RGB nearest kernels with fast precision, for
// baseline AArch64. vqrdmulh is the rounding high-half multiply of the
// contract in convert.hpp; the sums are separate vector adds.
#include "kernels/convert/yuv2rgb_fast_neon.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Plain {
  // Chroma terms (bias included) of 8 chroma samples.
  struct Terms {
    int16x8_t r, g, b;
  };

  static Terms chroma_terms(uint8x8_t u8, uint8x8_t v8, const Coef& c) {
    const int16x8_t u = scaled(u8), v = scaled(v8);
    return {vaddq_s16(vqrdmulhq_laneq_s16(v, c.k, 1), c.br),
            vaddq_s16(vaddq_s16(vqrdmulhq_laneq_s16(u, c.k, 2), vqrdmulhq_laneq_s16(v, c.k, 3)),
                      c.bg),
            vaddq_s16(vqrdmulhq_laneq_s16(u, c.k, 4), c.bb)};
  }

  // Luma term of 16 pixels as two int16 vectors.
  struct Luma {
    int16x8_t lo, hi;
  };

  static Luma luma_term(uint8x16_t y, const Coef& c) {
    return {vqrdmulhq_laneq_s16(scaled(vget_low_u8(y)), c.k, 0),
            vqrdmulhq_laneq_s16(scaled_high(y), c.k, 0)};
  }

  template <int P>
  static void block32(uint8x16_t ye, uint8x16_t yo, uint8x16_t u, uint8x16_t v, std::uint8_t* dst,
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
};

}  // namespace

void fill_neon_fast(Table& t) { fill_fast<Plain>(t); }

}  // namespace lanecast::kernels::convert
