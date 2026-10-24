// Helpers shared by the NEON convert translation units. Everything is in an
// anonymous namespace so each TU gets its own copy (see the SIMD TU rules in
// kernels.hpp).
#pragma once

#include <arm_neon.h>

#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {
namespace {

// Channel positions inside a pixel, and the pixel size in bytes.
template <int P>
struct Layout;
template <>
struct Layout<kRgb24> {
  static constexpr int r = 0, g = 1, b = 2, a = -1, bpp = 3;
};
template <>
struct Layout<kRgba> {
  static constexpr int r = 0, g = 1, b = 2, a = 3, bpp = 4;
};
template <>
struct Layout<kBgra> {
  static constexpr int r = 2, g = 1, b = 0, a = 3, bpp = 4;
};

// 16 pixels split into channels.
struct Planar16 {
  uint8x16_t r, g, b, a;
};

// Loads 16 pixels (vld3 / vld4). Alpha is 255 when the layout has none.
template <int P>
inline Planar16 load16(const std::uint8_t* p) {
  using L = Layout<P>;
  if constexpr (L::a < 0) {
    const uint8x16x3_t v = vld3q_u8(p);
    return {v.val[L::r], v.val[L::g], v.val[L::b], vdupq_n_u8(255)};
  } else {
    const uint8x16x4_t v = vld4q_u8(p);
    return {v.val[L::r], v.val[L::g], v.val[L::b], v.val[L::a]};
  }
}

// Stores 16 pixels (vst3 / vst4). Alpha is dropped when the layout has none.
template <int P>
inline void store16(std::uint8_t* p, const Planar16& c) {
  using L = Layout<P>;
  if constexpr (L::a < 0) {
    uint8x16x3_t v;
    v.val[L::r] = c.r;
    v.val[L::g] = c.g;
    v.val[L::b] = c.b;
    vst3q_u8(p, v);
  } else {
    uint8x16x4_t v;
    v.val[L::r] = c.r;
    v.val[L::g] = c.g;
    v.val[L::b] = c.b;
    v.val[L::a] = c.a;
    vst4q_u8(p, v);
  }
}

}  // namespace
}  // namespace lanecast::kernels::convert
