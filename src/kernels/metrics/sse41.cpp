// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
//
// Rows are consumed in 16 byte blocks. The last partial block loads past the
// visible end (row padding guarantee, kernels.hpp) and masks the extra lanes.
#include <smmintrin.h>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {
namespace {

inline __m128i load(const std::uint8_t* p) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

// Lanes below `rem` (0..15) all ones, the rest zero.
inline __m128i tail_mask(int rem) {
  const __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  return _mm_cmpgt_epi8(_mm_set1_epi8(static_cast<char>(rem)), iota);
}

inline __m128i abs_diff(__m128i a, __m128i b) {
  return _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
}

// Sum of d^2 as four u32 lanes (each gains at most 4 * 65025 per call).
inline __m128i square_sum(__m128i d) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i lo = _mm_unpacklo_epi8(d, zero);
  const __m128i hi = _mm_unpackhi_epi8(d, zero);
  return _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi));
}

inline std::uint64_t hsum_u32(__m128i v) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i q = _mm_add_epi64(_mm_unpacklo_epi32(v, zero), _mm_unpackhi_epi32(v, zero));
  return static_cast<std::uint64_t>(_mm_cvtsi128_si64(q)) +
         static_cast<std::uint64_t>(_mm_extract_epi64(q, 1));
}

// 8192 blocks keep every u32 lane below 2^32 (8192 * 260100 < 2^31).
std::uint64_t sse_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t total = 0;
  while (count >= 16) {
    const int n = count / 16 < 8192 ? count / 16 : 8192;
    __m128i acc0 = _mm_setzero_si128(), acc1 = acc0;
    int i = 0;
    for (; i + 2 <= n; i += 2, a += 32, b += 32) {
      acc0 = _mm_add_epi32(acc0, square_sum(abs_diff(load(a), load(b))));
      acc1 = _mm_add_epi32(acc1, square_sum(abs_diff(load(a + 16), load(b + 16))));
    }
    if (i < n) {
      acc0 = _mm_add_epi32(acc0, square_sum(abs_diff(load(a), load(b))));
      a += 16;
      b += 16;
    }
    total += hsum_u32(_mm_add_epi32(acc0, acc1));
    count -= n * 16;
  }
  if (count > 0) {
    const __m128i d = _mm_and_si128(abs_diff(load(a), load(b)), tail_mask(count));
    total += hsum_u32(square_sum(d));
  }
  return total;
}

std::uint64_t sad_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  __m128i acc = _mm_setzero_si128();  // two u64 partial sums
  for (; count >= 16; count -= 16, a += 16, b += 16) {
    acc = _mm_add_epi64(acc, _mm_sad_epu8(load(a), load(b)));
  }
  if (count > 0) {
    const __m128i m = tail_mask(count);
    acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_and_si128(load(a), m), _mm_and_si128(load(b), m)));
  }
  return static_cast<std::uint64_t>(_mm_cvtsi128_si64(acc)) +
         static_cast<std::uint64_t>(_mm_extract_epi64(acc, 1));
}

}  // namespace

void fill_sse41(Table& t) {
  t.sse = &sse_row;
  t.sad = &sad_row;
}

}  // namespace lanecast::kernels::metrics
