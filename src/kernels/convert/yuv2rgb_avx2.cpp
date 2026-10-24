// AVX2 yuv420p / nv12 -> packed RGB row kernels. Chroma upsampling and the
// drivers are shared with SSE4.1 (yuv2rgb_x86.hpp); the Q13 core handles 16
// pixels in one 256-bit register. Per-lane unpack and per-lane packs cancel,
// so the int16 results stay in pixel order and only the final 8-bit pack
// joins the two lanes.
#include <immintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/yuv2rgb_x86.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Avx2 {
  struct Coef {
    __m256i y_off, c128, one, ky, kr, kg, kb;
  };

  static Coef make(const YuvToRgbCoeffs& c) {
    return {_mm256_set1_epi16(c.y_offset),
            _mm256_set1_epi16(128),
            _mm256_set1_epi16(1),
            _mm256_set1_epi32(pair16(c.cy, 4096)),
            _mm256_set1_epi32(pair16(0, c.crv)),
            _mm256_set1_epi32(pair16(-c.cgu, -c.cgv)),
            _mm256_set1_epi32(pair16(c.cbu, 0))};
  }

  static __m128i to_u8(__m256i x) {
    return _mm_packus_epi16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  }

  static Planar16 yuv16(__m128i y, __m128i u, __m128i v, const Coef& c) {
    const __m256i y16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(y), c.y_off);
    const __m256i u16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u), c.c128);
    const __m256i v16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v), c.c128);
    const __m256i ylo = _mm256_madd_epi16(_mm256_unpacklo_epi16(y16, c.one), c.ky);
    const __m256i yhi = _mm256_madd_epi16(_mm256_unpackhi_epi16(y16, c.one), c.ky);
    const __m256i uvlo = _mm256_unpacklo_epi16(u16, v16), uvhi = _mm256_unpackhi_epi16(u16, v16);
    const auto plane = [&](const __m256i& k) {
      const __m256i lo = _mm256_add_epi32(ylo, _mm256_madd_epi16(uvlo, k));
      const __m256i hi = _mm256_add_epi32(yhi, _mm256_madd_epi16(uvhi, k));
      return to_u8(_mm256_packs_epi32(_mm256_srai_epi32(lo, 13), _mm256_srai_epi32(hi, 13)));
    };
    return {plane(c.kr), plane(c.kg), plane(c.kb), _mm_set1_epi8(-1)};
  }
};

}  // namespace

void fill_avx2_yuv2rgb(Table& t) { fill_yuv2rgb<Avx2>(t); }

}  // namespace lanecast::kernels::convert
