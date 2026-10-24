// SSE4.1 yuv420p / nv12 -> packed RGB nearest kernels with fast precision
// (16-bit contract in convert.hpp). The drivers are in yuv2rgb_fast_x86.hpp.
#include <smmintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/yuv2rgb_fast_x86.hpp"

namespace lanecast::kernels::convert {
namespace {

struct SseFast {
  struct Coef {
    __m128i cy, crv, ncgu, ncgv, cbu, br, bg, bb;
  };

  static Coef make(const YuvToRgbCoeffs& c) {
    const Bias b = make_bias(c);
    return {_mm_set1_epi16(c.cy),    _mm_set1_epi16(c.crv),
            _mm_set1_epi16(static_cast<short>(-c.cgu)),
            _mm_set1_epi16(static_cast<short>(-c.cgv)),
            _mm_set1_epi16(c.cbu),   _mm_set1_epi16(static_cast<short>(b.r)),
            _mm_set1_epi16(static_cast<short>(b.g)), _mm_set1_epi16(static_cast<short>(b.b))};
  }

  // The low 8 bytes of x as int16 lanes shifted left by 7.
  static __m128i scaled(__m128i x) { return _mm_slli_epi16(_mm_cvtepu8_epi16(x), 7); }

  static Planar16 yuv16(__m128i y, __m128i u8, __m128i v8, const Coef& c) {
    const __m128i ylo = _mm_mulhrs_epi16(scaled(y), c.cy);
    const __m128i yhi = _mm_mulhrs_epi16(scaled(_mm_srli_si128(y, 8)), c.cy);
    const __m128i u = scaled(u8), v = scaled(v8);
    const __m128i rc = _mm_add_epi16(_mm_mulhrs_epi16(v, c.crv), c.br);
    const __m128i gc = _mm_add_epi16(
        _mm_add_epi16(_mm_mulhrs_epi16(u, c.ncgu), _mm_mulhrs_epi16(v, c.ncgv)), c.bg);
    const __m128i bc = _mm_add_epi16(_mm_mulhrs_epi16(u, c.cbu), c.bb);
    const auto plane = [&](__m128i t) {
      const __m128i lo = _mm_srai_epi16(_mm_add_epi16(ylo, _mm_unpacklo_epi16(t, t)), 5);
      const __m128i hi = _mm_srai_epi16(_mm_add_epi16(yhi, _mm_unpackhi_epi16(t, t)), 5);
      return _mm_packus_epi16(lo, hi);
    };
    return {plane(rc), plane(gc), plane(bc), _mm_set1_epi8(-1)};
  }
};

}  // namespace

void fill_sse41_fast(Table& t) { fill_fast<SseFast>(t); }

}  // namespace lanecast::kernels::convert
