// Drivers for the fast-precision yuv420p / nv12 -> packed RGB nearest kernels,
// shared by the SSE4.1 and AVX2 translation units. Each TU supplies a core K:
//   typename K::Coef; static Coef K::make(const YuvToRgbCoeffs&);
//   static Planar16 K::yuv16(__m128i y, __m128i u8, __m128i v8, const Coef&);
// converting 16 pixels: y is 16 luma bytes, u8 and v8 hold the 8 chroma
// samples they share in their low 8 bytes. The core computes the chroma terms
// once per sample (pmulhrsw is the rounding high-half multiply of the contract
// in convert.hpp) and duplicates them to pixel pairs.
#pragma once

#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// Rounding high-half multiply, same value as the scalar rmul.
inline int rmul(int a, int k) { return (a * k + 16384) >> 15; }

// The three per-channel constants of the contract (bias_r, bias_g, bias_b).
struct Bias {
  int r, g, b;
};

inline Bias make_bias(const YuvToRgbCoeffs& c) {
  const int mid = 128 << 7, ky = rmul(c.y_offset << 7, c.cy);
  return {16 - ky - rmul(mid, c.crv), 16 - ky + rmul(mid, c.cgu) + rmul(mid, c.cgv),
          16 - ky - rmul(mid, c.cbu)};
}

// nv12 chroma: u bytes into the low 8 bytes, v bytes into the high 8.
constexpr Shuf kUvSplit = {{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}};

template <class K, int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  const typename K::Coef c = K::make(cf);
  for (int x = 0; x < width; x += 16) {
    const __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(u + x / 2));
    const __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v + x / 2));
    store16<P>(dst + x * Layout<P>::bpp, K::yuv16(loadu(y + x), u8, v8, c));
  }
}

template <class K, int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& cf) {
  const typename K::Coef c = K::make(cf);
  for (int x = 0; x < width; x += 16) {
    const __m128i p = pshufb(loadu(uv + x), kUvSplit);
    store16<P>(dst + x * Layout<P>::bpp, K::yuv16(loadu(y + x), p, _mm_srli_si128(p, 8), c));
  }
}

template <class K>
void fill_fast(Table& t) {
  t.yuv420_fast_to_packed[kRgb24] = &yuv420_nearest_row<K, kRgb24>;
  t.yuv420_fast_to_packed[kRgba] = &yuv420_nearest_row<K, kRgba>;
  t.yuv420_fast_to_packed[kBgra] = &yuv420_nearest_row<K, kBgra>;
  t.nv12_fast_to_packed[kRgb24] = &nv12_nearest_row<K, kRgb24>;
  t.nv12_fast_to_packed[kRgba] = &nv12_nearest_row<K, kRgba>;
  t.nv12_fast_to_packed[kBgra] = &nv12_nearest_row<K, kBgra>;
}

}  // namespace
}  // namespace lanecast::kernels::convert
