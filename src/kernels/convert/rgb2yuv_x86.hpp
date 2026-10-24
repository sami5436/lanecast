// packed RGB -> yuv420p / nv12 / gray row drivers shared by the SSE4.1 and
// AVX2 translation units. Chroma is the same 128-bit code in both; each TU
// supplies a core K for luma:
//   typename K::Coef; static Coef K::make(const RgbToYuvCoeffs&);
//   static __m128i K::luma16(const Planar16&, const Coef&);  // 16 u8 of Y
// Arithmetic follows the Q15 contract in convert.hpp: int32 accumulation, one
// arithmetic shift, saturation to 8 bits.
#pragma once

#include <cstring>

#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// pmaddwd operands for U and V: (R, G) pairs and (B, 0) pairs of the sums.
struct ChromaCoef {
  __m128i u_rg, u_b, v_rg, v_b;
};

inline ChromaCoef make_chroma(const RgbToYuvCoeffs& k) {
  return {_mm_set1_epi32(pair16(k.ur, k.ug)), _mm_set1_epi32(pair16(k.ub, 0)),
          _mm_set1_epi32(pair16(k.vr, k.vg)), _mm_set1_epi32(pair16(k.vb, 0))};
}

// U and V for 8 chroma samples from int16 sums:
// (k.r*SR + k.g*SG + k.b*SB + (128 << Shift) + (1 << (Shift - 1))) >> Shift.
// The low 8 bytes of cu, cv are the samples.
template <int Shift>
inline void chroma_uv(__m128i sr, __m128i sg, __m128i sb, const ChromaCoef& c, __m128i& cu,
                      __m128i& cv) {
  const __m128i rg_lo = _mm_unpacklo_epi16(sr, sg), rg_hi = _mm_unpackhi_epi16(sr, sg);
  const __m128i zero = _mm_setzero_si128();
  const __m128i b_lo = _mm_unpacklo_epi16(sb, zero), b_hi = _mm_unpackhi_epi16(sb, zero);
  const __m128i bias = _mm_set1_epi32((128 << Shift) + (1 << (Shift - 1)));
  const auto comp = [&](const __m128i& krg, const __m128i& kb) {
    const __m128i lo = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(rg_lo, krg),
                                                   _mm_madd_epi16(b_lo, kb)), bias);
    const __m128i hi = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(rg_hi, krg),
                                                   _mm_madd_epi16(b_hi, kb)), bias);
    const __m128i w = _mm_packs_epi32(_mm_srai_epi32(lo, Shift), _mm_srai_epi32(hi, Shift));
    return _mm_packus_epi16(w, w);
  };
  cu = comp(c.u_rg, c.u_b);
  cv = comp(c.v_rg, c.v_b);
}

// Sum of each horizontal pixel pair over both rows, as int16.
inline __m128i sum2x2(__m128i a, __m128i b) {
  const __m128i ones = _mm_set1_epi8(1);
  return _mm_add_epi16(_mm_maddubs_epi16(a, ones), _mm_maddubs_epi16(b, ones));
}

// [1 2 1] taps over both rows centered on the even columns, as the pair sums
// (x[2i-1] + x[2i]) + (x[2i] + x[2i+1]). pa, pb (the previous block) supply
// x[-1] across blocks; for the first block they repeat x[0] (edge clamp).
inline __m128i sum121(__m128i a, __m128i b, __m128i pa, __m128i pb) {
  const __m128i cur = sum2x2(a, b);
  const __m128i left = sum2x2(_mm_alignr_epi8(a, pa, 15), _mm_alignr_epi8(b, pb, 15));
  return _mm_add_epi16(cur, left);
}

// Previous block's channel vectors (two rows), for the [1 2 1] left filter.
struct Prev {
  __m128i r0, g0, b0, r1, g1, b1;
};

template <class K, int P, bool Nv12, bool Left>
inline void block(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                  std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v,
                  const typename K::Coef& c, const ChromaCoef& cc, Prev& prev, bool first) {
  const Planar16 a = load16<P>(s0), b = load16<P>(s1);
  storeu(y0, K::luma16(a, c));
  storeu(y1, K::luma16(b, c));
  __m128i sr, sg, sb;
  __m128i cu, cv;
  if constexpr (Left) {
    if (first) {
      const __m128i z = _mm_setzero_si128();
      prev = {_mm_shuffle_epi8(a.r, z), _mm_shuffle_epi8(a.g, z), _mm_shuffle_epi8(a.b, z),
              _mm_shuffle_epi8(b.r, z), _mm_shuffle_epi8(b.g, z), _mm_shuffle_epi8(b.b, z)};
    }
    sr = sum121(a.r, b.r, prev.r0, prev.r1);
    sg = sum121(a.g, b.g, prev.g0, prev.g1);
    sb = sum121(a.b, b.b, prev.b0, prev.b1);
    prev = {a.r, a.g, a.b, b.r, b.g, b.b};
    chroma_uv<18>(sr, sg, sb, cc, cu, cv);
  } else {
    sr = sum2x2(a.r, b.r);
    sg = sum2x2(a.g, b.g);
    sb = sum2x2(a.b, b.b);
    chroma_uv<17>(sr, sg, sb, cc, cu, cv);
  }
  if constexpr (Nv12) {
    storeu(u, _mm_unpacklo_epi8(cu, cv));
  } else {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(u), cu);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(v), cv);
  }
}

// Shared row driver. For odd widths the last chroma sample pairs the last
// pixel with itself, so the final block runs on a copy whose tail repeats the
// last pixel instead of reading padding.
template <class K, int P, bool Nv12, bool Left>
void packed_to_yuv_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                       std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v, int width,
                       const RgbToYuvCoeffs& k) {
  constexpr int bpp = Layout<P>::bpp;
  const typename K::Coef c = K::make(k);
  const ChromaCoef cc = make_chroma(k);
  const int step = Nv12 ? 2 : 1;  // chroma bytes per sample in u
  Prev prev{};
  int x = 0;
  const int body = (width & 1) ? width - 16 : width;
  for (; x < body; x += 16) {
    block<K, P, Nv12, Left>(s0 + x * bpp, s1 + x * bpp, y0 + x, y1 + x, u + (x / 2) * step,
                            v + (x / 2) * step, c, cc, prev, x == 0);
  }
  if (x < width) {
    alignas(16) std::uint8_t t0[16 * bpp], t1[16 * bpp];
    const int rem = width - x;
    std::memcpy(t0, s0 + x * bpp, static_cast<std::size_t>(rem * bpp));
    std::memcpy(t1, s1 + x * bpp, static_cast<std::size_t>(rem * bpp));
    for (int i = rem; i < 16; ++i) {
      std::memcpy(t0 + i * bpp, t0 + (rem - 1) * bpp, bpp);
      std::memcpy(t1 + i * bpp, t1 + (rem - 1) * bpp, bpp);
    }
    block<K, P, Nv12, Left>(t0, t1, y0 + x, y1 + x, u + (x / 2) * step, v + (x / 2) * step, c, cc,
                            prev, x == 0);
  }
}

template <class K, int P>
void packed_to_yuv420_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                          std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v, int width,
                          const RgbToYuvCoeffs& k, ChromaLocation loc) {
  if (loc == ChromaLocation::center) {
    packed_to_yuv_row<K, P, false, false>(s0, s1, y0, y1, u, v, width, k);
  } else {
    packed_to_yuv_row<K, P, false, true>(s0, s1, y0, y1, u, v, width, k);
  }
}

template <class K, int P>
void packed_to_nv12_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                        std::uint8_t* y1, std::uint8_t* uv, int width, const RgbToYuvCoeffs& k,
                        ChromaLocation loc) {
  if (loc == ChromaLocation::center) {
    packed_to_yuv_row<K, P, true, false>(s0, s1, y0, y1, uv, uv, width, k);
  } else {
    packed_to_yuv_row<K, P, true, true>(s0, s1, y0, y1, uv, uv, width, k);
  }
}

template <class K, int P>
void packed_to_gray_row(const std::uint8_t* src, std::uint8_t* gray, int width,
                        const RgbToYuvCoeffs& k) {
  const typename K::Coef c = K::make(k);
  for (int x = 0; x < width; x += 16) {
    storeu(gray + x, K::luma16(load16<P>(src + x * Layout<P>::bpp), c));
  }
}

template <class K>
void fill_rgb2yuv(Table& t) {
  t.packed_to_yuv420[kRgb24] = &packed_to_yuv420_row<K, kRgb24>;
  t.packed_to_yuv420[kRgba] = &packed_to_yuv420_row<K, kRgba>;
  t.packed_to_yuv420[kBgra] = &packed_to_yuv420_row<K, kBgra>;
  t.packed_to_nv12[kRgb24] = &packed_to_nv12_row<K, kRgb24>;
  t.packed_to_nv12[kRgba] = &packed_to_nv12_row<K, kRgba>;
  t.packed_to_nv12[kBgra] = &packed_to_nv12_row<K, kBgra>;
  t.packed_to_gray[kRgb24] = &packed_to_gray_row<K, kRgb24>;
  t.packed_to_gray[kRgba] = &packed_to_gray_row<K, kRgba>;
  t.packed_to_gray[kBgra] = &packed_to_gray_row<K, kBgra>;
}

}  // namespace
}  // namespace lanecast::kernels::convert
