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

using U16U = std::uint16_t __attribute__((__may_alias__, __aligned__(1)));
using I32U = std::int32_t __attribute__((__may_alias__, __aligned__(1)));

// ---- horizontal ----------------------------------------------------------

// (sum + 64) >> 7 for eight int32 sums (0..32640), narrowed to eight int16.
inline __m128i hround_pack(__m128i lo, __m128i hi) {
  const __m128i r = _mm_set1_epi32(64);
  return _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(lo, r), 7),
                         _mm_srai_epi32(_mm_add_epi32(hi, r), 7));
}

// 1 channel, 2 taps, eight outputs. Each output's two source bytes are
// gathered as one u16; widened, the vector reads (a0 b0 a1 b1 ...), which pairs
// directly with the row-major weights (w0 w1 w2 w3 ...) under pmaddwd.
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
  const __m128i lo = _mm_madd_epi16(_mm_cvtepu8_epi16(pair),
                                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
  const __m128i hi = _mm_madd_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(pair, 8)),
                                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 8)));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(mid), hround_pack(lo, hi));
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
// are padded to a multiple of 4, so steps of 8 then 4 never leave the row.
inline __m128i h1_acc(const std::uint8_t* s, const std::int16_t* w, int stride) {
  __m128i acc = _mm_setzero_si128();
  int k = 0;
  for (; k + 8 <= stride; k += 8) {
    const __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + k)));
    acc = _mm_add_epi32(
        acc, _mm_madd_epi16(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k))));
  }
  if (k < stride) {  // 4 weights, upper lanes zero
    const __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + k)));
    acc = _mm_add_epi32(
        acc, _mm_madd_epi16(v, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k))));
  }
  return acc;
}

// Four outputs at a time; the horizontal sums of four accumulators come out
// as one vector via two hadd levels.
void h1_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  const int stride = f.stride;
  for (int x = 0; x < n; x += 4) {
    __m128i a[4];
    for (int j = 0; j < 4; ++j) {
      const int xj = x + j < n ? x + j : n - 1;
      a[j] = h1_acc(src + f.start[xj], f.weights + static_cast<std::ptrdiff_t>(xj) * stride,
                    stride);
    }
    const __m128i sums = _mm_hadd_epi32(_mm_hadd_epi32(a[0], a[1]), _mm_hadd_epi32(a[2], a[3]));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(mid + x), hround_pack(sums, sums));
  }
}

// 1 channel, 2..6 taps, eight outputs per block from the FilterView block
// tables. The 32 byte source window is two vectors; pshufb zeroes lanes whose
// index byte has the top bit set, so each half is looked up with the other
// half's indices masked out and the results are or-ed. Each lookup yields two
// taps of all eight outputs; interleaving them against the matching weight
// pair gives the pmaddwd layout (tap 2m, tap 2m + 1). An odd last tap meets a
// zero weight. Outputs past the table blocks go to the gather kernels.
template <int T>
void h1_tbl(const std::uint8_t* src, std::int16_t* mid, const FilterView& f) {
  const std::uint8_t* ti = f.tbl;
  const std::int16_t* wt = f.wt;
  const __m128i c15 = _mm_set1_epi8(15), c16 = _mm_set1_epi8(16);
  for (int b = 0; b < f.tbl_blocks; ++b) {
    const std::uint8_t* s = src + f.start[b * 8];
    const __m128i lo_win = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
    const __m128i hi_win = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 16));
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = acc_lo;
    for (int p = 0; p < (T + 1) / 2; ++p) {
      const __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ti));
      ti += 16;
      const __m128i g = _mm_or_si128(
          _mm_shuffle_epi8(lo_win, _mm_or_si128(idx, _mm_cmpgt_epi8(idx, c15))),
          _mm_shuffle_epi8(hi_win, _mm_sub_epi8(idx, c16)));
      const __m128i il = _mm_unpacklo_epi8(g, _mm_srli_si128(g, 8));
      const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wt));
      const __m128i w1 = 2 * p + 1 < T
                             ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(wt + 8))
                             : _mm_setzero_si128();
      wt += 2 * p + 1 < T ? 16 : 8;
      acc_lo = _mm_add_epi32(
          acc_lo, _mm_madd_epi16(_mm_cvtepu8_epi16(il), _mm_unpacklo_epi16(w0, w1)));
      acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(il, 8)),
                                                   _mm_unpackhi_epi16(w0, w1)));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(mid + b * 8), hround_pack(acc_lo, acc_hi));
  }
}

// Interleaved channels (C = 2..4), taps >= 2: two taps per step. One 8 byte
// load holds both pixels (2C <= 8); shifting by C bytes and interleaving gives
// (p0c0 p1c0 p0c1 p1c1 ...), which pmaddwd multiplies by the weight pair
// (w0 w1) in every lane. Lanes beyond C are overwritten by the next output or
// land in the row padding. An odd last tap meets a zero padding weight.
template <int C>
void hc_taps(const std::uint8_t* src, std::int16_t* mid, int n, const FilterView& f) {
  for (int x = 0; x < n; ++x) {
    const std::uint8_t* s = src + static_cast<std::ptrdiff_t>(f.start[x]) * C;
    const std::int16_t* w = f.weights + static_cast<std::ptrdiff_t>(x) * f.stride;
    __m128i acc = _mm_setzero_si128();
    for (int k = 0; k < f.taps; k += 2) {
      const __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + k * C));
      const __m128i il = _mm_unpacklo_epi8(v, _mm_srli_epi64(v, 8 * C));
      const __m128i wp = _mm_set1_epi32(*reinterpret_cast<const I32U*>(w + k));
      acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_cvtepu8_epi16(il), wp));
    }
    _mm_storel_epi64(reinterpret_cast<__m128i*>(mid + x * C), hround_pack(acc, acc));
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
  if (f.tbl_blocks > 0) {
    switch (f.taps) {
      case 2: h1_tbl<2>(src, mid, f); break;
      case 3: h1_tbl<3>(src, mid, f); break;
      case 4: h1_tbl<4>(src, mid, f); break;
      case 5: h1_tbl<5>(src, mid, f); break;
      default: h1_tbl<6>(src, mid, f); break;
    }
    const int done = f.tbl_blocks * 8;
    if (done == n) return;
    const FilterView rest{f.taps, f.stride, f.start + done,
                          f.weights + static_cast<std::ptrdiff_t>(done) * f.stride};
    return hscale1(src, mid + done, n - done, rest);
  }
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

void fill_sse41(Table& t) {
  t.hscale[1] = &hscale1;
  t.hscale[2] = &hscale2;
  t.hscale[3] = &hscale3;
  t.hscale[4] = &hscale4;
  t.vscale = &vscale;
  fill_sse41_q7(t);
}

}  // namespace lanecast::kernels::scale
