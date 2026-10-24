// SSE4.1 packed RGB -> yuv420p / nv12 / gray row kernels. The shared drivers
// are in rgb2yuv_x86.hpp; this file is the 16-pixel Q15 luma core.
#include <smmintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/rgb2yuv_x86.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Sse {
  struct Coef {
    __m128i krg, kb, bias;
  };

  static Coef make(const RgbToYuvCoeffs& k) {
    return {_mm_set1_epi32(pair16(k.yr, k.yg)), _mm_set1_epi32(pair16(k.yb, 0)),
            _mm_set1_epi32((static_cast<int>(k.y_offset) << 15) + 16384)};
  }

  // Y for 8 pixels as int16 (not yet clamped to 8 bits).
  static __m128i luma8(__m128i r, __m128i g, __m128i b, const Coef& c) {
    const __m128i z = _mm_setzero_si128();
    const __m128i lo = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(r, g), c.krg),
                                                   _mm_madd_epi16(_mm_unpacklo_epi16(b, z), c.kb)),
                                     c.bias);
    const __m128i hi = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(r, g), c.krg),
                                                   _mm_madd_epi16(_mm_unpackhi_epi16(b, z), c.kb)),
                                     c.bias);
    return _mm_packs_epi32(_mm_srai_epi32(lo, 15), _mm_srai_epi32(hi, 15));
  }

  static __m128i luma16(const Planar16& p, const Coef& c) {
    const __m128i lo = luma8(_mm_cvtepu8_epi16(p.r), _mm_cvtepu8_epi16(p.g),
                             _mm_cvtepu8_epi16(p.b), c);
    const __m128i hi = luma8(_mm_cvtepu8_epi16(_mm_srli_si128(p.r, 8)),
                             _mm_cvtepu8_epi16(_mm_srli_si128(p.g, 8)),
                             _mm_cvtepu8_epi16(_mm_srli_si128(p.b, 8)), c);
    return _mm_packus_epi16(lo, hi);
  }
};

}  // namespace

void fill_sse41_rgb2yuv(Table& t) { fill_rgb2yuv<Sse>(t); }

}  // namespace lanecast::kernels::convert
