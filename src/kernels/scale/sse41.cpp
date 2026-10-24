// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
//
// All math is pmaddwd on int16 pairs (sample, sample) x (weight, weight):
// samples are 0..32640 and weights are 0..16384, so every product and pair
// sum is exact in int32 and the result equals the scalar reference bit for
// bit. Kernels use whole-vector loads and stores into the row padding
// instead of scalar tails.
#include <smmintrin.h>

#include <cstddef>
#include <cstdint>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

// ---- vertical ------------------------------------------------------------

// (acc + 2^20) >> 21 for four lanes.
inline __m128i vround(__m128i acc) {
  return _mm_srai_epi32(_mm_add_epi32(acc, _mm_set1_epi32(1 << 20)), 21);
}

inline __m128i vpack(__m128i a0, __m128i a1, __m128i a2, __m128i a3) {
  return _mm_packus_epi16(_mm_packs_epi32(vround(a0), vround(a1)),
                          _mm_packs_epi32(vround(a2), vround(a3)));
}

// Weight pair (w0, w1) in every 32 bit lane, matching unpack(row0, row1).
inline __m128i wpair(std::int16_t w0, std::int16_t w1) {
  return _mm_set1_epi32(static_cast<std::int32_t>(
      static_cast<std::uint32_t>(static_cast<std::uint16_t>(w0)) |
      (static_cast<std::uint32_t>(static_cast<std::uint16_t>(w1)) << 16)));
}

// 16 samples per step; rows are consumed two taps at a time. An odd last tap
// pairs with itself and a zero weight.
void vscale(const std::int16_t* const* rows, const std::int16_t* w, int taps, std::uint8_t* dst,
            int count) {
  for (int i = 0; i < count; i += 16) {
    __m128i c0 = _mm_setzero_si128(), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < taps; k += 2) {
      const bool pair = k + 1 < taps;
      const std::int16_t* r0 = rows[k] + i;
      const std::int16_t* r1 = pair ? rows[k + 1] + i : r0;
      const __m128i wp = wpair(w[k], pair ? w[k + 1] : std::int16_t{0});
      const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0));
      const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + 8));
      const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1));
      const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + 8));
      c0 = _mm_add_epi32(c0, _mm_madd_epi16(_mm_unpacklo_epi16(a0, b0), wp));
      c1 = _mm_add_epi32(c1, _mm_madd_epi16(_mm_unpackhi_epi16(a0, b0), wp));
      c2 = _mm_add_epi32(c2, _mm_madd_epi16(_mm_unpacklo_epi16(a1, b1), wp));
      c3 = _mm_add_epi32(c3, _mm_madd_epi16(_mm_unpackhi_epi16(a1, b1), wp));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), vpack(c0, c1, c2, c3));
  }
}

}  // namespace

void fill_sse41(Table& t) { t.vscale = &vscale; }

}  // namespace lanecast::kernels::scale
