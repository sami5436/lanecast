// AVX2 yuv420p / nv12 -> packed RGB nearest kernels with fast precision
// (16-bit contract in convert.hpp). 16 pixels fill one 256-bit register; the
// chroma bytes are duplicated before widening so the terms land in pixel
// order. The drivers are in yuv2rgb_fast_x86.hpp.
#include <immintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/yuv2rgb_fast_x86.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Avx2Fast {
  struct Coef {
    __m256i cy, crv, ncgu, ncgv, cbu, br, bg, bb;
  };

  static Coef make(const YuvToRgbCoeffs& c) {
    const Bias b = make_bias(c);
    return {_mm256_set1_epi16(c.cy),    _mm256_set1_epi16(c.crv),
            _mm256_set1_epi16(static_cast<short>(-c.cgu)),
            _mm256_set1_epi16(static_cast<short>(-c.cgv)),
            _mm256_set1_epi16(c.cbu),   _mm256_set1_epi16(static_cast<short>(b.r)),
            _mm256_set1_epi16(static_cast<short>(b.g)), _mm256_set1_epi16(static_cast<short>(b.b))};
  }

  static __m256i scaled(__m128i x) { return _mm256_slli_epi16(_mm256_cvtepu8_epi16(x), 7); }

  // 16 int16 lanes (5 fraction bits) to 16 clamped bytes.
  static __m128i to_u8(__m256i x) {
    const __m256i s = _mm256_srai_epi16(x, 5);
    return _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  }

  static Planar16 yuv16(__m128i y, __m128i u8, __m128i v8, const Coef& c) {
    const __m256i ly = _mm256_mulhrs_epi16(scaled(y), c.cy);
    const __m256i u = scaled(_mm_unpacklo_epi8(u8, u8)), v = scaled(_mm_unpacklo_epi8(v8, v8));
    const __m256i rc = _mm256_add_epi16(_mm256_mulhrs_epi16(v, c.crv), c.br);
    const __m256i gc = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_mulhrs_epi16(u, c.ncgu), _mm256_mulhrs_epi16(v, c.ncgv)), c.bg);
    const __m256i bc = _mm256_add_epi16(_mm256_mulhrs_epi16(u, c.cbu), c.bb);
    return {to_u8(_mm256_add_epi16(ly, rc)), to_u8(_mm256_add_epi16(ly, gc)),
            to_u8(_mm256_add_epi16(ly, bc)), _mm_set1_epi8(-1)};
  }
};

}  // namespace

void fill_avx2_fast(Table& t) { fill_fast<Avx2Fast>(t); }

}  // namespace lanecast::kernels::convert
