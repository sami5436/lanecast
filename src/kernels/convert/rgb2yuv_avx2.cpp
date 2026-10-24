// AVX2 packed RGB -> yuv420p / nv12 / gray row kernels. Chroma and the drivers
// are shared with SSE4.1 (rgb2yuv_x86.hpp); the Q15 luma core handles 16
// pixels in one 256-bit register (per-lane unpack and packs keep pixel order).
#include <immintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/rgb2yuv_x86.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Avx2 {
  struct Coef {
    __m256i krg, kb, bias;
  };

  static Coef make(const RgbToYuvCoeffs& k) {
    return {_mm256_set1_epi32(pair16(k.yr, k.yg)), _mm256_set1_epi32(pair16(k.yb, 0)),
            _mm256_set1_epi32((static_cast<int>(k.y_offset) << 15) + 16384)};
  }

  static __m128i luma16(const Planar16& p, const Coef& c) {
    const __m256i r = _mm256_cvtepu8_epi16(p.r), g = _mm256_cvtepu8_epi16(p.g);
    const __m256i b = _mm256_cvtepu8_epi16(p.b);
    const __m256i z = _mm256_setzero_si256();
    const __m256i lo = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(r, g), c.krg),
                         _mm256_madd_epi16(_mm256_unpacklo_epi16(b, z), c.kb)),
        c.bias);
    const __m256i hi = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(r, g), c.krg),
                         _mm256_madd_epi16(_mm256_unpackhi_epi16(b, z), c.kb)),
        c.bias);
    const __m256i y = _mm256_packs_epi32(_mm256_srai_epi32(lo, 15), _mm256_srai_epi32(hi, 15));
    return _mm_packus_epi16(_mm256_castsi256_si128(y), _mm256_extracti128_si256(y, 1));
  }
};

}  // namespace

void fill_avx2_rgb2yuv(Table& t) { fill_rgb2yuv<Avx2>(t); }

}  // namespace lanecast::kernels::convert
