// Scalar reference kernels for convert. This file defines the behavior every
// SIMD tier must match bit for bit (formulas in include/lanecast/convert.hpp).
#include <algorithm>

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
inline void store_rgb(std::uint8_t* d, int yv, int u, int v, const YuvToRgbCoeffs& c) {
  using L = Layout<P>;
  const int ly = c.cy * (yv - c.y_offset) + 4096;
  d[L::r] = clamp8((ly + c.crv * v) >> 13);
  d[L::g] = clamp8((ly - c.cgu * u - c.cgv * v) >> 13);
  d[L::b] = clamp8((ly + c.cbu * u) >> 13);
  if constexpr (L::a >= 0) d[L::a] = 255;
}

// Bilinear chroma for pixel x (formula in convert.hpp). cx, cxn are already
// clamped column indices; a is the near row, b the vertical neighbor row.
inline int bilinear(const std::uint8_t* a, const std::uint8_t* b, int cx, int cxn, bool odd) {
  const int v0 = 3 * a[cx] + b[cx];
  const int h = odd ? v0 + 3 * a[cxn] + b[cxn] : 2 * v0;
  return (h + 4) >> 3;
}

template <int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& c) {
  for (int x = 0; x < width; ++x) {
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], u[x >> 1], v[x >> 1], c);
  }
}

template <int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& c) {
  for (int x = 0; x < width; ++x) {
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], uv[(x >> 1) * 2], uv[(x >> 1) * 2 + 1], c);
  }
}

template <int P>
void yuv420_bilinear_row(const std::uint8_t* y, const std::uint8_t* u0, const std::uint8_t* u1,
                         const std::uint8_t* v0, const std::uint8_t* v1, std::uint8_t* dst,
                         int width, const YuvToRgbCoeffs& c) {
  const int cw = (width + 1) >> 1;
  for (int x = 0; x < width; ++x) {
    const int cx = x >> 1, cxn = std::min(cx + 1, cw - 1);
    const bool odd = (x & 1) != 0;
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], bilinear(u0, u1, cx, cxn, odd),
                 bilinear(v0, v1, cx, cxn, odd), c);
  }
}

template <int P>
void nv12_bilinear_row(const std::uint8_t* y, const std::uint8_t* uv0, const std::uint8_t* uv1,
                       std::uint8_t* dst, int width, const YuvToRgbCoeffs& c) {
  const int cw = (width + 1) >> 1;
  for (int x = 0; x < width; ++x) {
    const int cx = x >> 1, cxn = std::min(cx + 1, cw - 1);
    const bool odd = (x & 1) != 0;
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], bilinear(uv0, uv1, cx * 2, cxn * 2, odd),
                 bilinear(uv0 + 1, uv1 + 1, cx * 2, cxn * 2, odd), c);
  }
}

}  // namespace

void fill_scalar(Table& t) {
  t.yuv420_to_packed[kRgb24] = &yuv420_nearest_row<kRgb24>;
  t.yuv420_to_packed[kRgba] = &yuv420_nearest_row<kRgba>;
  t.yuv420_to_packed[kBgra] = &yuv420_nearest_row<kBgra>;
  t.nv12_to_packed[kRgb24] = &nv12_nearest_row<kRgb24>;
  t.nv12_to_packed[kRgba] = &nv12_nearest_row<kRgba>;
  t.nv12_to_packed[kBgra] = &nv12_nearest_row<kBgra>;
  t.yuv420_bilinear_to_packed[kRgb24] = &yuv420_bilinear_row<kRgb24>;
  t.yuv420_bilinear_to_packed[kRgba] = &yuv420_bilinear_row<kRgba>;
  t.yuv420_bilinear_to_packed[kBgra] = &yuv420_bilinear_row<kBgra>;
  t.nv12_bilinear_to_packed[kRgb24] = &nv12_bilinear_row<kRgb24>;
  t.nv12_bilinear_to_packed[kRgba] = &nv12_bilinear_row<kRgba>;
  t.nv12_bilinear_to_packed[kBgra] = &nv12_bilinear_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
