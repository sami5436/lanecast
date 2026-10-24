// Scalar reference kernels for scale. Defines the behavior every SIMD tier
// must match bit for bit (formulas in include/lanecast/scale.hpp).
#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

template <int C>
void hscale_row(const std::uint8_t* src, std::int16_t* mid, int dst_width, const FilterView& f) {
  for (int x = 0; x < dst_width; ++x) {
    const std::int16_t* w = f.weights + static_cast<std::ptrdiff_t>(x) * f.stride;
    const std::uint8_t* s = src + static_cast<std::ptrdiff_t>(f.start[x]) * C;
    for (int c = 0; c < C; ++c) {
      std::int32_t sum = 0;
      for (int k = 0; k < f.taps; ++k) sum += w[k] * s[k * C + c];
      mid[x * C + c] = static_cast<std::int16_t>((sum + 64) >> 7);
    }
  }
}

void vscale_row(const std::int16_t* const* rows, const std::int16_t* w, int taps,
                std::uint8_t* dst, int count) {
  for (int i = 0; i < count; ++i) {
    std::int32_t sum = 0;
    for (int k = 0; k < taps; ++k) sum += w[k] * rows[k][i];
    const std::int32_t v = (sum + (1 << 20)) >> 21;
    dst[i] = static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
  }
}

}  // namespace

void fill_scalar(Table& t) {
  t.hscale[1] = &hscale_row<1>;
  t.hscale[2] = &hscale_row<2>;
  t.hscale[3] = &hscale_row<3>;
  t.hscale[4] = &hscale_row<4>;
  t.vscale = &vscale_row;
  fill_scalar_q7(t);
}

}  // namespace lanecast::kernels::scale
