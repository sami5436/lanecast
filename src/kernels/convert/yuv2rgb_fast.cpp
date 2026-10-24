// Scalar reference for the fast (16-bit) yuv420p / nv12 -> packed RGB nearest
// kernels. Defines the behavior every SIMD tier must match bit for bit
// (formulas under "Fast precision" in include/lanecast/convert.hpp).
#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {
namespace {

inline std::uint8_t clamp8(int v) {
  return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// Rounding high-half multiply: what vqrdmulh and pmulhrsw compute.
inline int rmul(int a, int k) { return (a * k + 16384) >> 15; }

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

// The three per-channel constants of the contract (bias_r, bias_g, bias_b).
struct Bias {
  int r, g, b;
};

inline Bias make_bias(const YuvToRgbCoeffs& c) {
  const int mid = 128 << 7, ky = rmul(c.y_offset << 7, c.cy);
  return {16 - ky - rmul(mid, c.crv), 16 - ky + rmul(mid, c.cgu) + rmul(mid, c.cgv),
          16 - ky - rmul(mid, c.cbu)};
}

template <int P>
inline void store_rgb(std::uint8_t* d, int yv, int u, int v, const YuvToRgbCoeffs& c,
                      const Bias& k) {
  using L = Layout<P>;
  const int ly = rmul(yv << 7, c.cy);
  d[L::r] = clamp8((ly + rmul(v << 7, c.crv) + k.r) >> 5);
  d[L::g] = clamp8((ly + rmul(u << 7, -c.cgu) + rmul(v << 7, -c.cgv) + k.g) >> 5);
  d[L::b] = clamp8((ly + rmul(u << 7, c.cbu) + k.b) >> 5);
  if constexpr (L::a >= 0) d[L::a] = 255;
}

template <int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& c) {
  const Bias k = make_bias(c);
  for (int x = 0; x < width; ++x) {
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], u[x >> 1], v[x >> 1], c, k);
  }
}

template <int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& c) {
  const Bias k = make_bias(c);
  for (int x = 0; x < width; ++x) {
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], uv[(x >> 1) * 2], uv[(x >> 1) * 2 + 1], c, k);
  }
}

}  // namespace

void fill_scalar_fast(Table& t) {
  t.yuv420_fast_to_packed[kRgb24] = &yuv420_nearest_row<kRgb24>;
  t.yuv420_fast_to_packed[kRgba] = &yuv420_nearest_row<kRgba>;
  t.yuv420_fast_to_packed[kBgra] = &yuv420_nearest_row<kBgra>;
  t.nv12_fast_to_packed[kRgb24] = &nv12_nearest_row<kRgb24>;
  t.nv12_fast_to_packed[kRgba] = &nv12_nearest_row<kRgba>;
  t.nv12_fast_to_packed[kBgra] = &nv12_nearest_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
