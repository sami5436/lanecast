// NEON 4:4:4 Q7 yuv -> packed RGB row kernels. Arithmetic follows the formula
// in kernels.hpp exactly: int32 accumulation, one rounding shift by 20 and
// saturation to 8 bits.
#include <arm_neon.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/neon_common.hpp"

namespace lanecast::kernels::convert {
namespace {

inline uint8x8_t narrow(int32x4_t lo, int32x4_t hi) {
  return vqmovn_u16(vcombine_u16(vqmovun_s32(vrshrq_n_s32(lo, 20)),
                                 vqmovun_s32(vrshrq_n_s32(hi, 20))));
}

// 8 pixels from offset int16 inputs. k lanes: cy, crv, cgu, cgv, cbu.
inline void core8(int16x8_t y, int16x8_t u, int16x8_t v, int16x8_t k, uint8x8_t& r, uint8x8_t& g,
                  uint8x8_t& b) {
  const int32x4_t ylo = vmull_laneq_s16(vget_low_s16(y), k, 0);
  const int32x4_t yhi = vmull_high_laneq_s16(y, k, 0);
  const int32x4_t rl = vmlal_laneq_s16(ylo, vget_low_s16(v), k, 1);
  const int32x4_t rh = vmlal_high_laneq_s16(yhi, v, k, 1);
  int32x4_t gl = vmlsl_laneq_s16(ylo, vget_low_s16(u), k, 2);
  int32x4_t gh = vmlsl_high_laneq_s16(yhi, u, k, 2);
  gl = vmlsl_laneq_s16(gl, vget_low_s16(v), k, 3);
  gh = vmlsl_high_laneq_s16(gh, v, k, 3);
  const int32x4_t bl = vmlal_laneq_s16(ylo, vget_low_s16(u), k, 4);
  const int32x4_t bh = vmlal_high_laneq_s16(yhi, u, k, 4);
  r = narrow(rl, rh);
  g = narrow(gl, gh);
  b = narrow(bl, bh);
}

template <int P>
void yuv444_row(const std::int16_t* y, const std::int16_t* u, const std::int16_t* v,
                std::uint8_t* dst, int width, const YuvToRgbCoeffs& c) {
  const std::int16_t a[8] = {c.cy, c.crv, c.cgu, c.cgv, c.cbu, 0, 0, 0};
  const int16x8_t k = vld1q_s16(a);
  const int16x8_t y_off = vdupq_n_s16(static_cast<std::int16_t>(c.y_offset << 7));
  const int16x8_t c128 = vdupq_n_s16(128 << 7);
  // The rounding bias (1 << 19) is part of vrshrq_n_s32.
  for (int x = 0; x < width; x += 16) {
    uint8x8_t r0, g0, b0, r1, g1, b1;
    core8(vsubq_s16(vld1q_s16(y + x), y_off), vsubq_s16(vld1q_s16(u + x), c128),
          vsubq_s16(vld1q_s16(v + x), c128), k, r0, g0, b0);
    core8(vsubq_s16(vld1q_s16(y + x + 8), y_off), vsubq_s16(vld1q_s16(u + x + 8), c128),
          vsubq_s16(vld1q_s16(v + x + 8), c128), k, r1, g1, b1);
    store16<P>(dst + x * Layout<P>::bpp, {vcombine_u8(r0, r1), vcombine_u8(g0, g1),
                                          vcombine_u8(b0, b1), vdupq_n_u8(255)});
  }
}

}  // namespace

void fill_neon_yuv444(Table& t) {
  t.yuv444_q7_to_packed[kRgb24] = &yuv444_row<kRgb24>;
  t.yuv444_q7_to_packed[kRgba] = &yuv444_row<kRgba>;
  t.yuv444_q7_to_packed[kBgra] = &yuv444_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
