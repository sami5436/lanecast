// NEON Q7 vertical pass (formula in kernels.hpp), 16 samples per step.
#include <arm_neon.h>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

void vscale_q7_row(const std::int16_t* const* rows, const std::int16_t* w, int taps,
                   std::int16_t* dst, int count) {
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
    // Rounding shift by 14 is (x + 2^13) >> 14; the result fits int16.
    vst1q_s16(dst + i, vcombine_s16(vrshrn_n_s32(c0, 14), vrshrn_n_s32(c1, 14)));
    vst1q_s16(dst + i + 8, vcombine_s16(vrshrn_n_s32(c2, 14), vrshrn_n_s32(c3, 14)));
  }
}

}  // namespace

void fill_neon_q7(Table& t) { t.vscale_q7 = &vscale_q7_row; }

}  // namespace lanecast::kernels::scale
