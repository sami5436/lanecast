// SSE4.1 yuv420p / nv12 -> packed RGB row kernels (nearest and bilinear
// chroma). The shared drivers are in yuv2rgb_x86.hpp; this file is the
// 16-pixel Q13 core.
#include <smmintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/yuv2rgb_x86.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Sse {
  // pmaddwd operands. ky pairs with (Y, 1) and folds in the rounding bias;
  // kr, kg, kb pair with (U, V).
  struct Coef {
    __m128i y_off, c128, one, ky, kr, kg, kb;
  };

  static Coef make(const YuvToRgbCoeffs& c) {
    return {_mm_set1_epi16(c.y_offset),
            _mm_set1_epi16(128),
            _mm_set1_epi16(1),
            _mm_set1_epi32(pair16(c.cy, 4096)),
            _mm_set1_epi32(pair16(0, c.crv)),
            _mm_set1_epi32(pair16(-c.cgu, -c.cgv)),
            _mm_set1_epi32(pair16(c.cbu, 0))};
  }

  // 8 pixels: y, u, v are already offset int16 lanes. r, g, b come back as
  // int16 (arithmetic shift of the int32 sums), not yet clamped to 8 bits.
  static void core8(__m128i y, __m128i u, __m128i v, const Coef& c, __m128i& r, __m128i& g,
                    __m128i& b) {
    const __m128i ylo = _mm_madd_epi16(_mm_unpacklo_epi16(y, c.one), c.ky);
    const __m128i yhi = _mm_madd_epi16(_mm_unpackhi_epi16(y, c.one), c.ky);
    const __m128i uvlo = _mm_unpacklo_epi16(u, v), uvhi = _mm_unpackhi_epi16(u, v);
    const auto plane = [&](const __m128i& k) {
      const __m128i lo = _mm_add_epi32(ylo, _mm_madd_epi16(uvlo, k));
      const __m128i hi = _mm_add_epi32(yhi, _mm_madd_epi16(uvhi, k));
      return _mm_packs_epi32(_mm_srai_epi32(lo, 13), _mm_srai_epi32(hi, 13));
    };
    r = plane(c.kr);
    g = plane(c.kg);
    b = plane(c.kb);
  }

  static Planar16 yuv16(__m128i y, __m128i u, __m128i v, const Coef& c) {
    const auto lo = [&](__m128i x, __m128i off) { return _mm_sub_epi16(_mm_cvtepu8_epi16(x), off); };
    const auto hi = [&](__m128i x, __m128i off) {
      return _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(x, 8)), off);
    };
    __m128i r0, g0, b0, r1, g1, b1;
    core8(lo(y, c.y_off), lo(u, c.c128), lo(v, c.c128), c, r0, g0, b0);
    core8(hi(y, c.y_off), hi(u, c.c128), hi(v, c.c128), c, r1, g1, b1);
    return {_mm_packus_epi16(r0, r1), _mm_packus_epi16(g0, g1), _mm_packus_epi16(b0, b1),
            _mm_set1_epi8(-1)};
  }
};

}  // namespace

void fill_sse41_yuv2rgb(Table& t) { fill_yuv2rgb<Sse>(t); }

}  // namespace lanecast::kernels::convert
