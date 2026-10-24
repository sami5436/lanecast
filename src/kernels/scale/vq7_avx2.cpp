// AVX2 Q7 vertical pass (formula in kernels.hpp), 32 samples per step, two
// taps at a time. Per lane unpacklo gives samples (0-3 | 8-11) and unpackhi
// (4-7 | 12-15); packs_epi32(lo, hi) per lane then yields (0-7 | 8-15), in
// order, so no permute is needed.
#include <immintrin.h>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

inline __m256i vround(__m256i acc) {
  return _mm256_srai_epi32(_mm256_add_epi32(acc, _mm256_set1_epi32(1 << 13)), 14);
}

inline __m256i wpair(std::int16_t w0, std::int16_t w1) {
  return _mm256_set1_epi32(static_cast<std::int32_t>(
      static_cast<std::uint32_t>(static_cast<std::uint16_t>(w0)) |
      (static_cast<std::uint32_t>(static_cast<std::uint16_t>(w1)) << 16)));
}

void vscale_q7_row(const std::int16_t* const* rows, const std::int16_t* w, int taps,
                   std::int16_t* dst, int count) {
  for (int i = 0; i < count; i += 32) {
    __m256i lo0 = _mm256_setzero_si256(), hi0 = lo0, lo1 = lo0, hi1 = lo0;
    for (int k = 0; k < taps; k += 2) {
      const bool pair = k + 1 < taps;
      const std::int16_t* r0 = rows[k] + i;
      const std::int16_t* r1 = pair ? rows[k + 1] + i : r0;
      const __m256i wp = wpair(w[k], pair ? w[k + 1] : std::int16_t{0});
      const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0));
      const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 16));
      const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1));
      const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 16));
      lo0 = _mm256_add_epi32(lo0, _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, b0), wp));
      hi0 = _mm256_add_epi32(hi0, _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, b0), wp));
      lo1 = _mm256_add_epi32(lo1, _mm256_madd_epi16(_mm256_unpacklo_epi16(a1, b1), wp));
      hi1 = _mm256_add_epi32(hi1, _mm256_madd_epi16(_mm256_unpackhi_epi16(a1, b1), wp));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                        _mm256_packs_epi32(vround(lo0), vround(hi0)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 16),
                        _mm256_packs_epi32(vround(lo1), vround(hi1)));
  }
}

}  // namespace

void fill_avx2_q7(Table& t) { t.vscale_q7 = &vscale_q7_row; }

}  // namespace lanecast::kernels::scale
