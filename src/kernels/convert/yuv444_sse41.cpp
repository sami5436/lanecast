// SSE4.1 4:4:4 Q7 yuv -> packed RGB row kernels, 16 pixels per iteration.
// Arithmetic follows the formula in kernels.hpp: pmaddwd with int32
// accumulation, one arithmetic shift by 20, saturation to 8 bits.
#include <smmintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// pmaddwd operands. ky pairs with (Y, 0); kr, kg, kb pair with (U, V).
struct Coef {
  __m128i y_off, c128, bias, ky, kr, kg, kb;
};

inline Coef make_coef(const YuvToRgbCoeffs& c) {
  return {_mm_set1_epi16(static_cast<short>(c.y_offset << 7)),
          _mm_set1_epi16(128 << 7),
          _mm_set1_epi32(1 << 19),
          _mm_set1_epi32(pair16(c.cy, 0)),
          _mm_set1_epi32(pair16(0, c.crv)),
          _mm_set1_epi32(pair16(-c.cgu, -c.cgv)),
          _mm_set1_epi32(pair16(c.cbu, 0))};
}

inline __m128i load16s(const std::int16_t* p) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

// 8 pixels from int16 Q7 inputs; r, g, b come back as int16, unclamped.
inline void core8(const std::int16_t* yp, const std::int16_t* up, const std::int16_t* vp,
                  const Coef& c, __m128i& r, __m128i& g, __m128i& b) {
  const __m128i y = _mm_sub_epi16(load16s(yp), c.y_off);
  const __m128i u = _mm_sub_epi16(load16s(up), c.c128);
  const __m128i v = _mm_sub_epi16(load16s(vp), c.c128);
  const __m128i zero = _mm_setzero_si128();
  const __m128i ylo = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(y, zero), c.ky), c.bias);
  const __m128i yhi = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(y, zero), c.ky), c.bias);
  const __m128i uvlo = _mm_unpacklo_epi16(u, v), uvhi = _mm_unpackhi_epi16(u, v);
  const auto plane = [&](const __m128i& k) {
    const __m128i lo = _mm_add_epi32(ylo, _mm_madd_epi16(uvlo, k));
    const __m128i hi = _mm_add_epi32(yhi, _mm_madd_epi16(uvhi, k));
    return _mm_packs_epi32(_mm_srai_epi32(lo, 20), _mm_srai_epi32(hi, 20));
  };
  r = plane(c.kr);
  g = plane(c.kg);
  b = plane(c.kb);
}

template <int P>
void yuv444_row(const std::int16_t* y, const std::int16_t* u, const std::int16_t* v,
                std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  const Coef c = make_coef(cf);
  for (int x = 0; x < width; x += 16) {
    __m128i r0, g0, b0, r1, g1, b1;
    core8(y + x, u + x, v + x, c, r0, g0, b0);
    core8(y + x + 8, u + x + 8, v + x + 8, c, r1, g1, b1);
    store16<P>(dst + x * Layout<P>::bpp,
               {_mm_packus_epi16(r0, r1), _mm_packus_epi16(g0, g1), _mm_packus_epi16(b0, b1),
                _mm_set1_epi8(-1)});
  }
}

}  // namespace

void fill_sse41_yuv444(Table& t) {
  t.yuv444_q7_to_packed[kRgb24] = &yuv444_row<kRgb24>;
  t.yuv444_q7_to_packed[kRgba] = &yuv444_row<kRgba>;
  t.yuv444_q7_to_packed[kBgra] = &yuv444_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
