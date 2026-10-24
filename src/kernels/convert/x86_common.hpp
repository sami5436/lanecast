// Helpers shared by the SSE4.1 and AVX2 convert translation units. Everything
// is in an anonymous namespace so each TU gets its own copy (see the SIMD TU
// rules in kernels.hpp). Includes only 128-bit intrinsics: the AVX2 TUs reuse
// them as VEX-encoded code.
#pragma once

#include <smmintrin.h>

#include <cstdint>

#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {
namespace {

// Channel positions inside a pixel, and the pixel size in bytes.
template <int P>
struct Layout;
template <>
struct Layout<kRgb24> {
  static constexpr int r = 0, g = 1, b = 2, a = -1, bpp = 3;
};
template <>
struct Layout<kRgba> {
  static constexpr int r = 0, g = 1, b = 2, a = 3, bpp = 4;
};
template <>
struct Layout<kBgra> {
  static constexpr int r = 2, g = 1, b = 0, a = 3, bpp = 4;
};

// 16 pixels split into channels.
struct Planar16 {
  __m128i r, g, b, a;
};

// A pshufb control vector. Index -128 (0x80) zeroes the byte.
struct alignas(16) Shuf {
  std::int8_t b[16];
};

inline __m128i load(const Shuf& s) { return _mm_load_si128(reinterpret_cast<const __m128i*>(s.b)); }
inline __m128i loadu(const std::uint8_t* p) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
inline void storeu(std::uint8_t* p, __m128i v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
inline __m128i pshufb(__m128i v, const Shuf& s) { return _mm_shuffle_epi8(v, load(s)); }

// rgb24 is 48 bytes for 16 pixels, i.e. three 16-byte chunks. Channel c of
// pixel p sits at byte 3p + c. gather[c][k] picks channel c out of chunk k
// into pixel order; scatter[c][k] places channel c into output chunk k. OR
// the three results of one channel (or one chunk) to finish.
struct Rgb24Masks {
  Shuf gather[3][3];
  Shuf scatter[3][3];
};

constexpr Rgb24Masks make_rgb24_masks() {
  constexpr std::int8_t kZero = -128;
  Rgb24Masks m{};
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 16; ++i) {
        const int pos = 3 * i + c;  // gather: pixel i of channel c
        m.gather[c][k].b[i] = (pos / 16 == k) ? static_cast<std::int8_t>(pos % 16) : kZero;
        const int out = 16 * k + i;  // scatter: byte i of output chunk k
        m.scatter[c][k].b[i] = (out % 3 == c) ? static_cast<std::int8_t>(out / 3) : kZero;
      }
    }
  }
  return m;
}
constexpr Rgb24Masks kRgb24Masks = make_rgb24_masks();

// Within one rgba / bgra group of four pixels (16 bytes): moves the channels
// to planar order r r r r g g g g b b b b a a a a.
constexpr Shuf kQuadPlanar = {{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}};

// Loads 16 pixels. Alpha is 255 when the layout has none.
template <int P>
inline Planar16 load16(const std::uint8_t* p) {
  using L = Layout<P>;
  if constexpr (L::a < 0) {
    const __m128i a0 = loadu(p), a1 = loadu(p + 16), a2 = loadu(p + 32);
    const auto gather = [&](int c) {
      return _mm_or_si128(_mm_or_si128(pshufb(a0, kRgb24Masks.gather[c][0]),
                                       pshufb(a1, kRgb24Masks.gather[c][1])),
                          pshufb(a2, kRgb24Masks.gather[c][2]));
    };
    return {gather(L::r), gather(L::g), gather(L::b), _mm_set1_epi8(-1)};
  } else {
    // Four groups t_k = [R G B A] x 4 pixels, then a 4x4 transpose of dwords.
    const __m128i t0 = pshufb(loadu(p), kQuadPlanar), t1 = pshufb(loadu(p + 16), kQuadPlanar);
    const __m128i t2 = pshufb(loadu(p + 32), kQuadPlanar), t3 = pshufb(loadu(p + 48), kQuadPlanar);
    const __m128i rg01 = _mm_unpacklo_epi32(t0, t1), ba01 = _mm_unpackhi_epi32(t0, t1);
    const __m128i rg23 = _mm_unpacklo_epi32(t2, t3), ba23 = _mm_unpackhi_epi32(t2, t3);
    const __m128i c[4] = {_mm_unpacklo_epi64(rg01, rg23), _mm_unpackhi_epi64(rg01, rg23),
                          _mm_unpacklo_epi64(ba01, ba23), _mm_unpackhi_epi64(ba01, ba23)};
    return {c[L::r], c[L::g], c[L::b], c[L::a]};
  }
}

// Stores 16 pixels. Alpha is dropped when the layout has none.
template <int P>
inline void store16(std::uint8_t* p, const Planar16& v) {
  using L = Layout<P>;
  if constexpr (L::a < 0) {
    for (int k = 0; k < 3; ++k) {
      storeu(p + 16 * k, _mm_or_si128(_mm_or_si128(pshufb(v.r, kRgb24Masks.scatter[0][k]),
                                                   pshufb(v.g, kRgb24Masks.scatter[1][k])),
                                      pshufb(v.b, kRgb24Masks.scatter[2][k])));
    }
  } else {
    __m128i c[4];
    c[L::r] = v.r;
    c[L::g] = v.g;
    c[L::b] = v.b;
    c[L::a] = v.a;
    const __m128i lo01 = _mm_unpacklo_epi8(c[0], c[1]), hi01 = _mm_unpackhi_epi8(c[0], c[1]);
    const __m128i lo23 = _mm_unpacklo_epi8(c[2], c[3]), hi23 = _mm_unpackhi_epi8(c[2], c[3]);
    storeu(p, _mm_unpacklo_epi16(lo01, lo23));
    storeu(p + 16, _mm_unpackhi_epi16(lo01, lo23));
    storeu(p + 32, _mm_unpacklo_epi16(hi01, hi23));
    storeu(p + 48, _mm_unpackhi_epi16(hi01, hi23));
  }
}

}  // namespace
}  // namespace lanecast::kernels::convert
