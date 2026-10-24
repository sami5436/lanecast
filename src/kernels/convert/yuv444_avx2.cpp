// AVX2 4:4:4 Q7 yuv -> packed RGB row kernels, 16 pixels per 256-bit core.
// Per-lane unpack and per-lane packs cancel, so the int16 results stay in
// pixel order and only the final 8-bit pack joins the two lanes. Arithmetic
// follows the formula in kernels.hpp.
#include <immintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

struct Coef {
  __m256i y_off, c128, bias, ky, kr, kg, kb;
};

inline Coef make_coef(const YuvToRgbCoeffs& c) {
  return {_mm256_set1_epi16(static_cast<short>(c.y_offset << 7)),
          _mm256_set1_epi16(128 << 7),
          _mm256_set1_epi32(1 << 19),
          _mm256_set1_epi32(pair16(c.cy, 0)),
          _mm256_set1_epi32(pair16(0, c.crv)),
          _mm256_set1_epi32(pair16(-c.cgu, -c.cgv)),
          _mm256_set1_epi32(pair16(c.cbu, 0))};
}

inline __m256i load16s(const std::int16_t* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

inline __m128i to_u8(__m256i x) {
  return _mm_packus_epi16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
}

template <int P>
void yuv444_row(const std::int16_t* y, const std::int16_t* u, const std::int16_t* v,
                std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  const Coef c = make_coef(cf);
  const __m256i zero = _mm256_setzero_si256();
  for (int x = 0; x < width; x += 16) {
    const __m256i y16 = _mm256_sub_epi16(load16s(y + x), c.y_off);
    const __m256i u16 = _mm256_sub_epi16(load16s(u + x), c.c128);
    const __m256i v16 = _mm256_sub_epi16(load16s(v + x), c.c128);
    const __m256i ylo = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_unpacklo_epi16(y16, zero), c.ky), c.bias);
    const __m256i yhi = _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_unpackhi_epi16(y16, zero), c.ky), c.bias);
    const __m256i uvlo = _mm256_unpacklo_epi16(u16, v16), uvhi = _mm256_unpackhi_epi16(u16, v16);
    const auto plane = [&](const __m256i& k) {
      const __m256i lo = _mm256_add_epi32(ylo, _mm256_madd_epi16(uvlo, k));
      const __m256i hi = _mm256_add_epi32(yhi, _mm256_madd_epi16(uvhi, k));
      return to_u8(_mm256_packs_epi32(_mm256_srai_epi32(lo, 20), _mm256_srai_epi32(hi, 20)));
    };
    store16<P>(dst + x * Layout<P>::bpp,
               {plane(c.kr), plane(c.kg), plane(c.kb), _mm_set1_epi8(-1)});
  }
}

}  // namespace

void fill_avx2_yuv444(Table& t) {
  t.yuv444_q7_to_packed[kRgb24] = &yuv444_row<kRgb24>;
  t.yuv444_q7_to_packed[kRgba] = &yuv444_row<kRgba>;
  t.yuv444_q7_to_packed[kBgra] = &yuv444_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
