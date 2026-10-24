// Scalar reference for the 4:4:4 Q7 yuv -> packed RGB row kernels. Defines the
// behavior every SIMD tier must match bit for bit (formula in kernels.hpp).
#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {
namespace {

inline std::uint8_t clamp8(int v) {
  return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// Byte offsets of R, G, B, A inside a pixel, and the pixel size.
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

template <int P>
void yuv444_row(const std::int16_t* y, const std::int16_t* u, const std::int16_t* v,
                std::uint8_t* dst, int width, const YuvToRgbCoeffs& c) {
  using L = Layout<P>;
  for (int x = 0; x < width; ++x) {
    std::uint8_t* d = dst + x * L::bpp;
    const int cu = u[x] - (128 << 7), cv = v[x] - (128 << 7);
    const int ly = c.cy * (y[x] - (c.y_offset << 7)) + (1 << 19);
    d[L::r] = clamp8((ly + c.crv * cv) >> 20);
    d[L::g] = clamp8((ly - c.cgu * cu - c.cgv * cv) >> 20);
    d[L::b] = clamp8((ly + c.cbu * cu) >> 20);
    if constexpr (L::a >= 0) d[L::a] = 255;
  }
}

}  // namespace

void fill_scalar_yuv444(Table& t) {
  t.yuv444_q7_to_packed[kRgb24] = &yuv444_row<kRgb24>;
  t.yuv444_q7_to_packed[kRgba] = &yuv444_row<kRgba>;
  t.yuv444_q7_to_packed[kBgra] = &yuv444_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
