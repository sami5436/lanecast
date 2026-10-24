// AVX2, compiled with -mavx2. Runs on top of the sse41 table.
// Store only the kernels this tier specializes; others are inherited.
//
// Same exact pmaddwd math as sse41.cpp, twice as wide. Lane crossing rules:
// unpack, pack and cvtepu8_epi16 on 256 bit vectors act per 128 bit lane, so
// every kernel below either keeps the data order lane-local on purpose (the
// vertical unpack/madd/packs chain returns samples in order) or fixes the one
// place where it is not with a single permute4x64.
#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

using U16U = std::uint16_t __attribute__((__may_alias__, __aligned__(1)));
using I32U = std::int32_t __attribute__((__may_alias__, __aligned__(1)));

// ---- horizontal ----------------------------------------------------------

// (sum + 64) >> 7 for eight int32 sums (0..32640), narrowed to eight int16 in
// order: the 128 bit halves are packed against each other, not per lane.
inline __m128i hround_pack(__m256i acc) {
  const __m256i r = _mm256_srai_epi32(_mm256_add_epi32(acc, _mm256_set1_epi32(64)), 7);
  return _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
}

// 1 channel, 2 taps, eight outputs: eight gathered u16 source pairs widen to
// (a0 b0 a1 b1 ...), which pairs with the row-major weights under vpmaddwd.
inline void h1_t2_block(const std::uint8_t* src, const std::int32_t* st, const std::int16_t* w,
                        std::int16_t* mid) {
  __m128i pair = _mm_setzero_si128();
#define LC_PAIR(j) pair = _mm_insert_epi16(pair, *reinterpret_cast<const U16U*>(src + st[j]), j)
  LC_PAIR(0);
  LC_PAIR(1);
  LC_PAIR(2);
  LC_PAIR(3);
  LC_PAIR(4);
  LC_PAIR(5);
  LC_PAIR(6);
  LC_PAIR(7);
#undef LC_PAIR
  const __m256i acc = _mm256_madd_epi16(_mm256_cvtepu8_epi16(pair),
                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w)));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(mid), hround_pack(acc));
}

// The last block may run past n: its extra lanes reuse the last output's start
// (weights past the last row are zero) and land in the row padding.
void h1_t2(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  int x = 0;
  for (; x + 8 <= n; x += 8) h1_t2_block(src, f.start + x, f.weights + 2 * x, mid + x);
  if (x < n) {
    std::int32_t st[8];
    for (int j = 0; j < 8; ++j) st[j] = f.start[x + j < n ? x + j : n - 1];
    h1_t2_block(src, st, f.weights + 2 * x, mid + x);
  }
}

// Four int32 partial sums of one output, 1 channel, any taps >= 3. Weight rows
// are padded to a multiple of 4, so steps of 16, 8 then 4 never leave the row.
inline __m128i h1_acc(const std::uint8_t* s, const std::int16_t* w, int stride) {
  __m256i acc = _mm256_setzero_si256();
  int k = 0;
  for (; k + 16 <= stride; k += 16) {
    const __m256i v =
        _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + k)));
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k))));
  }
  __m128i r = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  if (k + 8 <= stride) {
    const __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + k)));
    r = _mm_add_epi32(
        r, _mm_madd_epi16(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k))));
    k += 8;
  }
  if (k < stride) {  // 4 weights, upper lanes zero
    const __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + k)));
    r = _mm_add_epi32(
        r, _mm_madd_epi16(v, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k))));
  }
  return r;
}

void h1_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  const int stride = f.stride;
  const __m128i r = _mm_set1_epi32(64);
  for (int x = 0; x < n; x += 4) {
    __m128i a[4];
    for (int j = 0; j < 4; ++j) {
      const int xj = x + j < n ? x + j : n - 1;
      a[j] = h1_acc(src + f.start[xj], f.weights + static_cast<std::ptrdiff_t>(xj) * stride,
                    stride);
    }
    const __m128i sums = _mm_hadd_epi32(_mm_hadd_epi32(a[0], a[1]), _mm_hadd_epi32(a[2], a[3]));
    const __m128i q = _mm_srai_epi32(_mm_add_epi32(sums, r), 7);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(mid + x), _mm_packs_epi32(q, q));
  }
}

// Interleaved channels (C = 2..4), taps >= 2: two outputs per step, one in
// each 128 bit lane, two taps at a time. Per lane, an 8 byte load holds both
// pixels (2C <= 8); shifting by C bytes and interleaving gives
// (p0c0 p1c0 p0c1 p1c1 ...), widened by unpacking against zero (also per lane,
// so each lane widens its own low 8 bytes) and multiplied by that output's
// weight pair. Lanes beyond C are overwritten by the next output or land in
// the row padding. An odd last tap meets a zero padding weight.
template <int C>
void hc_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  const __m256i zero = _mm256_setzero_si256();
  for (int x = 0; x < n; x += 2) {
    const int x1 = x + 1 < n ? x + 1 : n - 1;
    const std::uint8_t* s0 = src + static_cast<std::ptrdiff_t>(f.start[x]) * C;
    const std::uint8_t* s1 = src + static_cast<std::ptrdiff_t>(f.start[x1]) * C;
    const std::int16_t* w0 = f.weights + static_cast<std::ptrdiff_t>(x) * f.stride;
    const std::int16_t* w1 = f.weights + static_cast<std::ptrdiff_t>(x1) * f.stride;
    __m256i acc = zero;
    for (int k = 0; k < f.taps; k += 2) {
      const __m256i v = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s0 + k * C))),
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s1 + k * C)), 1);
      const __m256i il = _mm256_unpacklo_epi8(v, _mm256_srli_epi64(v, 8 * C));
      const __m256i wp = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_set1_epi32(*reinterpret_cast<const I32U*>(w0 + k))),
          _mm_set1_epi32(*reinterpret_cast<const I32U*>(w1 + k)), 1);
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_unpacklo_epi8(il, zero), wp));
    }
    const __m128i out = hround_pack(acc);  // output x in the low 4, x1 in the high 4
    if constexpr (C == 4) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(mid + x * C), out);
    } else {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(mid + x * C), out);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(mid + x1 * C), _mm_srli_si128(out, 8));
    }
  }
}

// 1 pixel wide source: the single weight is exactly 16384, so mid = src * 128.
template <int C>
void h_copy(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  for (int x = 0; x < n; ++x) {
    for (int c = 0; c < C; ++c) {
      mid[x * C + c] = static_cast<std::int16_t>(src[f.start[x] * C + c] << 7);
    }
  }
}

void hscale1(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 1) return h_copy<1>(src, mid, n, f);
  if (f.taps == 2) return h1_t2(src, mid, n, f);
  h1_taps(src, mid, n, f);
}
void hscale2(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 1) return h_copy<2>(src, mid, n, f);
  hc_taps<2>(src, mid, n, f);
}
void hscale3(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 1) return h_copy<3>(src, mid, n, f);
  hc_taps<3>(src, mid, n, f);
}
void hscale4(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  if (f.taps == 1) return h_copy<4>(src, mid, n, f);
  hc_taps<4>(src, mid, n, f);
}

// ---- vertical ------------------------------------------------------------

inline __m256i vround(__m256i acc) {
  return _mm256_srai_epi32(_mm256_add_epi32(acc, _mm256_set1_epi32(1 << 20)), 21);
}

inline __m256i wpair(std::int16_t w0, std::int16_t w1) {
  return _mm256_set1_epi32(static_cast<std::int32_t>(
      static_cast<std::uint32_t>(static_cast<std::uint16_t>(w0)) |
      (static_cast<std::uint32_t>(static_cast<std::uint16_t>(w1)) << 16)));
}

// 32 samples per step, two taps at a time. With a = samples 0..15 and
// b = the paired row, per lane unpacklo gives samples (0-3 | 8-11) and
// unpackhi (4-7 | 12-15). packs_epi32(lo, hi) per lane then yields
// (0-7 | 8-15): in order. The two 16 sample halves are then packed to bytes,
// which leaves quadwords (0-7, 16-23, 8-15, 24-31); permute4x64 restores order.
void vscale(const std::int16_t* const* rows, const std::int16_t* w, int taps, std::uint8_t* dst,
            int count) {
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
    const __m256i p0 = _mm256_packs_epi32(vround(lo0), vround(hi0));
    const __m256i p1 = _mm256_packs_epi32(vround(lo1), vround(hi1));
    const __m256i bytes = _mm256_packus_epi16(p0, p1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                        _mm256_permute4x64_epi64(bytes, _MM_SHUFFLE(3, 1, 2, 0)));
  }
}

}  // namespace

void fill_avx2(Table& t) {
  t.hscale[1] = &hscale1;
  t.hscale[2] = &hscale2;
  t.hscale[3] = &hscale3;
  t.hscale[4] = &hscale4;
  t.vscale = &vscale;
}

}  // namespace lanecast::kernels::scale
