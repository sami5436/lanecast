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

// Sums for 4 adjacent 4x4 blocks (16 columns). out[k] receives block k as
// {sum a, sum b, sum a^2 + b^2, sum a*b}.
inline void ssim_group(const std::uint8_t* a, std::ptrdiff_t a_stride, const std::uint8_t* b,
                       std::ptrdiff_t b_stride, __m128i out[4]) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i ones8 = _mm_set1_epi8(1);
  const __m128i ones16 = _mm_set1_epi16(1);
  __m128i s1 = zero, s2 = zero, ss_lo = zero, ss_hi = zero, ab_lo = zero, ab_hi = zero;
  for (int y = 0; y < 4; ++y, a += a_stride, b += b_stride) {
    const __m128i va = load(a);
    const __m128i vb = load(b);
    s1 = _mm_add_epi16(s1, _mm_maddubs_epi16(va, ones8));
    s2 = _mm_add_epi16(s2, _mm_maddubs_epi16(vb, ones8));
    const __m128i al = _mm_unpacklo_epi8(va, zero), ah = _mm_unpackhi_epi8(va, zero);
    const __m128i bl = _mm_unpacklo_epi8(vb, zero), bh = _mm_unpackhi_epi8(vb, zero);
    ss_lo = _mm_add_epi32(ss_lo, _mm_add_epi32(_mm_madd_epi16(al, al), _mm_madd_epi16(bl, bl)));
    ss_hi = _mm_add_epi32(ss_hi, _mm_add_epi32(_mm_madd_epi16(ah, ah), _mm_madd_epi16(bh, bh)));
    ab_lo = _mm_add_epi32(ab_lo, _mm_madd_epi16(al, bl));
    ab_hi = _mm_add_epi32(ab_hi, _mm_madd_epi16(ah, bh));
  }
  // Each vector below holds one statistic for blocks 0..3; transpose to records.
  const __m128i f1 = _mm_madd_epi16(s1, ones16);
  const __m128i f2 = _mm_madd_epi16(s2, ones16);
  const __m128i fs = _mm_hadd_epi32(ss_lo, ss_hi);
  const __m128i fa = _mm_hadd_epi32(ab_lo, ab_hi);
  const __m128i t0 = _mm_unpacklo_epi32(f1, f2), t1 = _mm_unpacklo_epi32(fs, fa);
  const __m128i t2 = _mm_unpackhi_epi32(f1, f2), t3 = _mm_unpackhi_epi32(fs, fa);
  out[0] = _mm_unpacklo_epi64(t0, t1);
  out[1] = _mm_unpackhi_epi64(t0, t1);
  out[2] = _mm_unpacklo_epi64(t2, t3);
  out[3] = _mm_unpackhi_epi64(t2, t3);
}

void ssim_4x4_row(const std::uint8_t* a, std::ptrdiff_t a_stride, const std::uint8_t* b,
                  std::ptrdiff_t b_stride, int blocks, std::int32_t (*sums)[4]) {
  __m128i g[4];
  int i = 0;
  for (; i + 4 <= blocks; i += 4) {
    ssim_group(a + 4 * i, a_stride, b + 4 * i, b_stride, g);
    for (int k = 0; k < 4; ++k) _mm_storeu_si128(reinterpret_cast<__m128i*>(sums[i + k]), g[k]);
  }
  if (i < blocks) {
    // sums holds exactly `blocks` entries: store only those of the last group.
    ssim_group(a + 4 * i, a_stride, b + 4 * i, b_stride, g);
    for (int k = 0; i + k < blocks; ++k) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(sums[i + k]), g[k]);
    }
  }
}

}  // namespace

void fill_sse41(Table& t) {
  t.sse = &sse_row;
  t.sad = &sad_row;
  t.ssim_4x4 = &ssim_4x4_row;
}

}  // namespace lanecast::kernels::metrics
