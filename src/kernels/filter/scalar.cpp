// Scalar reference kernels for filter. Defines the behavior every SIMD tier
// must match bit for bit (formulas in include/lanecast/filter.hpp).
#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {
namespace {

inline int div255(int x) {
  x += 128;
  return (x + (x >> 8)) >> 8;
}

void blend_row(const std::uint8_t* over, std::uint8_t* base, int pixels) {
  for (int i = 0; i < pixels; ++i, over += 4, base += 4) {
    const int a = over[3];
    const int na = 255 - a;
    for (int c = 0; c < 3; ++c) {
      base[c] = static_cast<std::uint8_t>(div255(over[c] * a + base[c] * na));
    }
    base[3] = static_cast<std::uint8_t>(a + div255(base[3] * na));
  }
}

}  // namespace

void fill_scalar(Table& t) { t.blend = &blend_row; }

}  // namespace lanecast::kernels::filter
