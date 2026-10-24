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

void lut_row(const std::uint8_t* src, std::uint8_t* dst, int count, const std::uint8_t* lut) {
  for (int i = 0; i < count; ++i) dst[i] = lut[src[i]];
}

void lut_skip_alpha_row(const std::uint8_t* src, std::uint8_t* dst, int pixels,
                        const std::uint8_t* lut) {
  for (int i = 0; i < pixels; ++i, src += 4, dst += 4) {
    dst[0] = lut[src[0]];
    dst[1] = lut[src[1]];
    dst[2] = lut[src[2]];
    dst[3] = src[3];
  }
}

void blur_h_row(const std::uint8_t* src, std::uint16_t* dst, int count, int channels,
                int radius) {
  const int c = channels;
  if (radius == 1) {
    for (int i = 0; i < count; ++i) {
      dst[i] = static_cast<std::uint16_t>(src[i] + 2 * src[i + c] + src[i + 2 * c]);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      dst[i] = static_cast<std::uint16_t>(src[i] + 4 * src[i + c] + 6 * src[i + 2 * c] +
                                          4 * src[i + 3 * c] + src[i + 4 * c]);
    }
  }
}

void blur_v_row(const std::uint16_t* const* rows, std::uint8_t* dst, int count, int radius) {
  if (radius == 1) {
    for (int i = 0; i < count; ++i) {
      dst[i] = static_cast<std::uint8_t>((rows[0][i] + 2 * rows[1][i] + rows[2][i] + 8) >> 4);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      const int v = rows[0][i] + 4 * rows[1][i] + 6 * rows[2][i] + 4 * rows[3][i] + rows[4][i];
      dst[i] = static_cast<std::uint8_t>((v + 128) >> 8);
    }
  }
}

}  // namespace

void fill_scalar(Table& t) {
  t.blend = &blend_row;
  t.blur_h = &blur_h_row;
  t.blur_v = &blur_v_row;
  t.lut = &lut_row;
  t.lut_skip_alpha = &lut_skip_alpha_row;
}

}  // namespace lanecast::kernels::filter
