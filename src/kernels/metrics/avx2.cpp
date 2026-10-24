// AVX2, compiled with -mavx2. Runs on top of the sse41 table.
// Store only the kernels this tier specializes; others are inherited.
//
// Same math as sse41.cpp, twice as wide, in 32 byte blocks. The last partial
// block loads past the visible end (row padding guarantee, kernels.hpp) and
// masks the extra lanes.
#include <immintrin.h>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {
namespace {

inline __m256i load(const std::uint8_t* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

// Lanes below `rem` (0..31) all ones, the rest zero.
inline __m256i tail_mask(int rem) {
  const __m256i iota = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  return _mm256_cmpgt_epi8(_mm256_set1_epi8(static_cast<char>(rem)), iota);
}

inline __m256i abs_diff(__m256i a, __m256i b) {
  return _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
}

// Sum of d^2 as eight u32 lanes (each gains at most 4 * 65025 per call).
inline __m256i square_sum(__m256i d) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i lo = _mm256_unpacklo_epi8(d, zero);
  const __m256i hi = _mm256_unpackhi_epi8(d, zero);
  return _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
}

inline std::uint64_t hsum_u32(__m256i v) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i q = _mm256_add_epi64(_mm256_unpacklo_epi32(v, zero), _mm256_unpackhi_epi32(v, zero));
  const __m128i r = _mm_add_epi64(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
  return static_cast<std::uint64_t>(_mm_cvtsi128_si64(r)) +
         static_cast<std::uint64_t>(_mm_extract_epi64(r, 1));
}

// 8192 blocks keep every u32 lane below 2^32 (8192 * 260100 < 2^31).
std::uint64_t sse_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t total = 0;
  while (count >= 32) {
    const int n = count / 32 < 8192 ? count / 32 : 8192;
    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;
    for (; i + 2 <= n; i += 2, a += 64, b += 64) {
      acc0 = _mm256_add_epi32(acc0, square_sum(abs_diff(load(a), load(b))));
      acc1 = _mm256_add_epi32(acc1, square_sum(abs_diff(load(a + 32), load(b + 32))));
    }
    if (i < n) {
      acc0 = _mm256_add_epi32(acc0, square_sum(abs_diff(load(a), load(b))));
      a += 32;
      b += 32;
    }
    total += hsum_u32(_mm256_add_epi32(acc0, acc1));
    count -= n * 32;
  }
  if (count > 0) {
    const __m256i d = _mm256_and_si256(abs_diff(load(a), load(b)), tail_mask(count));
    total += hsum_u32(square_sum(d));
  }
  return total;
}

std::uint64_t sad_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  __m256i acc = _mm256_setzero_si256();  // four u64 partial sums
  for (; count >= 32; count -= 32, a += 32, b += 32) {
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(load(a), load(b)));
  }
  if (count > 0) {
    const __m256i m = tail_mask(count);
    acc = _mm256_add_epi64(
        acc, _mm256_sad_epu8(_mm256_and_si256(load(a), m), _mm256_and_si256(load(b), m)));
  }
  const __m128i r = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  return static_cast<std::uint64_t>(_mm_cvtsi128_si64(r)) +
         static_cast<std::uint64_t>(_mm_extract_epi64(r, 1));
}

// Sums for 8 adjacent 4x4 blocks (32 columns). Everything runs per 128 bit
// lane, so out[k] holds block k in its low lane and block 4 + k in its high
// lane as {sum a, sum b, sum a^2 + b^2, sum a*b}.
inline void ssim_group(const std::uint8_t* a, std::ptrdiff_t a_stride, const std::uint8_t* b,
                       std::ptrdiff_t b_stride, __m256i out[4]) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i ones8 = _mm256_set1_epi8(1);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256i s1 = zero, s2 = zero, ss_lo = zero, ss_hi = zero, ab_lo = zero, ab_hi = zero;
  for (int y = 0; y < 4; ++y, a += a_stride, b += b_stride) {
    const __m256i va = load(a);
    const __m256i vb = load(b);
    s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(va, ones8));
    s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(vb, ones8));
    const __m256i al = _mm256_unpacklo_epi8(va, zero), ah = _mm256_unpackhi_epi8(va, zero);
    const __m256i bl = _mm256_unpacklo_epi8(vb, zero), bh = _mm256_unpackhi_epi8(vb, zero);
    ss_lo = _mm256_add_epi32(
        ss_lo, _mm256_add_epi32(_mm256_madd_epi16(al, al), _mm256_madd_epi16(bl, bl)));
    ss_hi = _mm256_add_epi32(
        ss_hi, _mm256_add_epi32(_mm256_madd_epi16(ah, ah), _mm256_madd_epi16(bh, bh)));
    ab_lo = _mm256_add_epi32(ab_lo, _mm256_madd_epi16(al, bl));
    ab_hi = _mm256_add_epi32(ab_hi, _mm256_madd_epi16(ah, bh));
  }
  const __m256i f1 = _mm256_madd_epi16(s1, ones16);
  const __m256i f2 = _mm256_madd_epi16(s2, ones16);
  const __m256i fs = _mm256_hadd_epi32(ss_lo, ss_hi);
  const __m256i fa = _mm256_hadd_epi32(ab_lo, ab_hi);
  const __m256i t0 = _mm256_unpacklo_epi32(f1, f2), t1 = _mm256_unpacklo_epi32(fs, fa);
  const __m256i t2 = _mm256_unpackhi_epi32(f1, f2), t3 = _mm256_unpackhi_epi32(fs, fa);
  out[0] = _mm256_unpacklo_epi64(t0, t1);
  out[1] = _mm256_unpackhi_epi64(t0, t1);
  out[2] = _mm256_unpacklo_epi64(t2, t3);
  out[3] = _mm256_unpackhi_epi64(t2, t3);
}

void ssim_4x4_row(const std::uint8_t* a, std::ptrdiff_t a_stride, const std::uint8_t* b,
                  std::ptrdiff_t b_stride, int blocks, std::int32_t (*sums)[4]) {
  __m256i g[4];
  for (int i = 0; i < blocks; i += 8) {
    ssim_group(a + 4 * i, a_stride, b + 4 * i, b_stride, g);
    // sums holds exactly `blocks` entries: store only those that exist.
    for (int k = 0; k < 4; ++k) {
      if (i + k < blocks) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sums[i + k]), _mm256_castsi256_si128(g[k]));
      }
      if (i + 4 + k < blocks) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sums[i + 4 + k]),
                         _mm256_extracti128_si256(g[k], 1));
      }
    }
  }
}

}  // namespace

void fill_avx2(Table& t) {
  t.sse = &sse_row;
  t.sad = &sad_row;
  t.ssim_4x4 = &ssim_4x4_row;
}

}  // namespace lanecast::kernels::metrics
