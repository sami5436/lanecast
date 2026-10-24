// AVX2, compiled with -mavx2. Runs on top of the sse41 table.
// Store only the kernels this tier specializes; others are inherited.
//
// Blocks stay at 16 pixels (rgba) or 32 chroma samples (uv): wider blocks
// would run past the 64 bytes of row padding that kernels may touch.
#include <immintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

inline __m256i loadu256(const std::uint8_t* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
inline void storeu256(std::uint8_t* p, __m256i v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}

// rgba <-> bgra: swap channels 0 and 2 (the mask repeats per 128-bit lane).
void swap_rb_row(const std::uint8_t* src, std::uint8_t* dst, int width) {
  const __m256i m = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15));
  for (int x = 0; x < width; x += 16) {
    storeu256(dst + 4 * x, _mm256_shuffle_epi8(loadu256(src + 4 * x), m));
    storeu256(dst + 4 * x + 32, _mm256_shuffle_epi8(loadu256(src + 4 * x + 32), m));
  }
}

// The unpack works per 128-bit lane, so the two output halves are rejoined
// with permute2x128.
void interleave_uv_row(const std::uint8_t* u, const std::uint8_t* v, std::uint8_t* uv, int cw) {
  for (int i = 0; i < cw; i += 32) {
    const __m256i a = loadu256(u + i), b = loadu256(v + i);
    const __m256i lo = _mm256_unpacklo_epi8(a, b), hi = _mm256_unpackhi_epi8(a, b);
    storeu256(uv + 2 * i, _mm256_permute2x128_si256(lo, hi, 0x20));
    storeu256(uv + 2 * i + 32, _mm256_permute2x128_si256(lo, hi, 0x31));
  }
}

// Per lane: evens to the low 8 bytes, odds to the high 8. Then unpack64 pairs
// the lanes of the two loads, and permute4x64 (0, 2, 1, 3) restores order.
void deinterleave_uv_row(const std::uint8_t* uv, std::uint8_t* u, std::uint8_t* v, int cw) {
  const __m256i m = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
  for (int i = 0; i < cw; i += 32) {
    const __m256i a = _mm256_shuffle_epi8(loadu256(uv + 2 * i), m);
    const __m256i b = _mm256_shuffle_epi8(loadu256(uv + 2 * i + 32), m);
    storeu256(u + i, _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(a, b), 0xD8));
    storeu256(v + i, _mm256_permute4x64_epi64(_mm256_unpackhi_epi64(a, b), 0xD8));
  }
}

}  // namespace

void fill_avx2(Table& t) {
  t.swizzle[kRgba][kBgra] = &swap_rb_row;
  t.swizzle[kBgra][kRgba] = &swap_rb_row;
  t.interleave_uv = &interleave_uv_row;
  t.deinterleave_uv = &deinterleave_uv_row;
  fill_avx2_yuv2rgb(t);
  fill_avx2_rgb2yuv(t);
}

}  // namespace lanecast::kernels::convert
