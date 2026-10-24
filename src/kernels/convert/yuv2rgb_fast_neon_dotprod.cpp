// yuv420p / nv12 -> packed RGB nearest kernels with fast precision, for CPUs
// with FEAT_RDM (implied by the armv8.2 dotprod tier). vqrdmlah adds the
// rounding high-half product to an accumulator, which is exactly rmul(a, k)
// followed by an add in the contract (convert.hpp), so the output equals the
// plain NEON kernels with fewer operations.
#include "kernels/convert/yuv2rgb_fast_neon.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Rdm {
  template <int P>
  static void block32(uint8x16_t ye, uint8x16_t yo, uint8x16_t u, uint8x16_t v, std::uint8_t* dst,
                      const Coef& c) {
    // Chroma terms (bias included) of the low (0) and high (1) 8 samples.
    int16x8_t r[2], g[2], b[2];
    const uint8x8_t us[2] = {vget_low_u8(u), vget_high_u8(u)};
    const uint8x8_t vs[2] = {vget_low_u8(v), vget_high_u8(v)};
    for (int h = 0; h < 2; ++h) {
      const int16x8_t us16 = scaled(us[h]), vs16 = scaled(vs[h]);
      r[h] = vqrdmlahq_laneq_s16(c.br, vs16, c.k, 1);
      g[h] = vqrdmlahq_laneq_s16(vqrdmlahq_laneq_s16(c.bg, us16, c.k, 2), vs16, c.k, 3);
      b[h] = vqrdmlahq_laneq_s16(c.bb, us16, c.k, 4);
    }
    const int16x8_t y[4] = {scaled(vget_low_u8(ye)), scaled_high(ye), scaled(vget_low_u8(yo)),
                            scaled_high(yo)};
    Planar16 first, second;
    const auto chan = [&](const int16x8_t (&t)[2], uint8x16_t& f, uint8x16_t& s) {
      const uint8x16_t e = narrow(vqrdmlahq_laneq_s16(t[0], y[0], c.k, 0),
                                  vqrdmlahq_laneq_s16(t[1], y[1], c.k, 0));
      const uint8x16_t o = narrow(vqrdmlahq_laneq_s16(t[0], y[2], c.k, 0),
                                  vqrdmlahq_laneq_s16(t[1], y[3], c.k, 0));
      f = vzip1q_u8(e, o);
      s = vzip2q_u8(e, o);
    };
    chan(r, first.r, second.r);
    chan(g, first.g, second.g);
    chan(b, first.b, second.b);
    first.a = second.a = vdupq_n_u8(255);
    store16<P>(dst, first);
    store16<P>(dst + 16 * Layout<P>::bpp, second);
  }
};

}  // namespace

void fill_neon_dotprod_yuv2rgb(Table& t) { fill_fast<Rdm>(t); }

}  // namespace lanecast::kernels::convert
