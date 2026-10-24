// Scalar reference for the Q7 vertical pass (formula in kernels.hpp).
#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

void vscale_q7_row(const std::int16_t* const* rows, const std::int16_t* w, int taps,
                   std::int16_t* dst, int count) {
  for (int i = 0; i < count; ++i) {
    std::int32_t sum = 0;
    for (int k = 0; k < taps; ++k) sum += w[k] * rows[k][i];
    dst[i] = static_cast<std::int16_t>((sum + (1 << 13)) >> 14);
  }
}

}  // namespace

void fill_scalar_q7(Table& t) { t.vscale_q7 = &vscale_q7_row; }

}  // namespace lanecast::kernels::scale
