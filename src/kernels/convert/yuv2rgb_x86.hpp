// yuv420p / nv12 -> packed RGB row drivers shared by the SSE4.1 and AVX2
// translation units. The chroma upsampling is the same 128-bit code in both;
// each TU supplies a core K that turns 16 pixels of Y, U, V (u8x16 each, full
// resolution) into planar RGB:
//   typename K::Coef; static Coef K::make(const YuvToRgbCoeffs&);
//   static Planar16 K::yuv16(__m128i y, __m128i u, __m128i v, const Coef&);
// Arithmetic follows the Q13 contract in convert.hpp: int32 accumulation, one
// arithmetic shift, saturation to 8 bits.
#pragma once

#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// Two int16 values as one 32-bit lane, for pmaddwd against interleaved data.
inline std::int32_t pair16(int lo, int hi) {
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(static_cast<std::uint16_t>(lo)) |
                                   static_cast<std::uint32_t>(static_cast<std::uint16_t>(hi)) << 16);
}

// Duplicates each of the even (or odd) bytes: u u u u ... for nv12 nearest.
constexpr Shuf kDupEven = {{0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14}};
constexpr Shuf kDupOdd = {{1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15}};
constexpr Shuf kLane0 = {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};

template <class K, int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  const typename K::Coef c = K::make(cf);
  for (int x = 0; x < width; x += 16) {
    const __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(u + x / 2));
    const __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v + x / 2));
    store16<P>(dst + x * Layout<P>::bpp,
               K::yuv16(loadu(y + x), _mm_unpacklo_epi8(u8, u8), _mm_unpacklo_epi8(v8, v8), c));
  }
}

template <class K, int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& cf) {
  const typename K::Coef c = K::make(cf);
  for (int x = 0; x < width; x += 16) {
    const __m128i p = loadu(uv + x);
    store16<P>(dst + x * Layout<P>::bpp,
               K::yuv16(loadu(y + x), pshufb(p, kDupEven), pshufb(p, kDupOdd), c));
  }
}

// Vertical pass of the bilinear filter for 8 chroma samples (int16 lanes):
// 3 * a + b.
inline __m128i vert(__m128i a, __m128i b) {
  return _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a, a), a), b);
}

// Horizontal pass for 8 chroma samples, giving 16 u8 values in pixel order.
// v is the vertical result of columns cx..cx+7, vp and vn the same shifted to
// the left and right neighbor columns. Left siting: even pixels take 2 * v,
// odd ones v + vn, rounded once as (h + 4) >> 3. Center siting: 3 * v plus
// the neighbor on the pixel's side (vp even, vn odd), (h + 8) >> 4.
template <bool Center>
inline __m128i horiz(__m128i v, __m128i vp, __m128i vn) {
  __m128i even, odd;
  if constexpr (Center) {
    const __m128i v3 = _mm_add_epi16(_mm_add_epi16(v, v), _mm_add_epi16(v, _mm_set1_epi16(8)));
    even = _mm_srli_epi16(_mm_add_epi16(v3, vp), 4);
    odd = _mm_srli_epi16(_mm_add_epi16(v3, vn), 4);
  } else {
    (void)vp;
    even = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(v, v), _mm_set1_epi16(4)), 3);
    odd = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(v, vn), _mm_set1_epi16(4)), 3);
  }
  return _mm_packus_epi16(_mm_unpacklo_epi16(even, odd), _mm_unpackhi_epi16(even, odd));
}

// Vertical results for the U and V planes of one 8-sample chroma block.
struct Vert {
  __m128i u, v;
};

// Walks the row 8 chroma samples (16 pixels) at a time. load(cx) returns the
// vertical pass for chroma columns cx..cx+7 and may read past the plane's
// visible width (padding). The right neighbor of the last visible chroma
// column is the column itself, so the final block patches that lane and never
// looks at the lookahead block. The left neighbor of column 0 is column 0.
template <class K, int P, bool Center, class Load>
inline void bilinear_row(const std::uint8_t* y, std::uint8_t* dst, int width,
                         const typename K::Coef& c, Load load) {
  const int cw = (width + 1) >> 1;
  Vert cur = load(0);
  Vert prev = {pshufb(cur.u, kLane0), pshufb(cur.v, kLane0)};
  int cx = 0;
  for (; cx + 8 < cw; cx += 8) {
    const Vert next = load(cx + 8);
    const __m128i u = horiz<Center>(cur.u, _mm_alignr_epi8(cur.u, prev.u, 14),
                                    _mm_alignr_epi8(next.u, cur.u, 2));
    const __m128i v = horiz<Center>(cur.v, _mm_alignr_epi8(cur.v, prev.v, 14),
                                    _mm_alignr_epi8(next.v, cur.v, 2));
    store16<P>(dst + 2 * cx * Layout<P>::bpp, K::yuv16(loadu(y + 2 * cx), u, v, c));
    prev = cur;
    cur = next;
  }
  // Last block: lane cw - 1 - cx (0..7) is its own right neighbor.
  const __m128i self =
      _mm_cmpeq_epi16(_mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7), _mm_set1_epi16(static_cast<short>(cw - 1 - cx)));
  const __m128i un = _mm_blendv_epi8(_mm_alignr_epi8(cur.u, cur.u, 2), cur.u, self);
  const __m128i vn = _mm_blendv_epi8(_mm_alignr_epi8(cur.v, cur.v, 2), cur.v, self);
  store16<P>(dst + 2 * cx * Layout<P>::bpp,
             K::yuv16(loadu(y + 2 * cx),
                      horiz<Center>(cur.u, _mm_alignr_epi8(cur.u, prev.u, 14), un),
                      horiz<Center>(cur.v, _mm_alignr_epi8(cur.v, prev.v, 14), vn), c));
}

template <class K, int P, class Load>
inline void bilinear_dispatch(const std::uint8_t* y, std::uint8_t* dst, int width,
                              const YuvToRgbCoeffs& cf, ChromaLocation loc, Load load) {
  const typename K::Coef c = K::make(cf);
  if (loc == ChromaLocation::center) {
    bilinear_row<K, P, true>(y, dst, width, c, load);
  } else {
    bilinear_row<K, P, false>(y, dst, width, c, load);
  }
}

// 8 chroma bytes widened to int16 lanes.
inline __m128i load8_u16(const std::uint8_t* p) {
  return _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

template <class K, int P>
void yuv420_bilinear_row(const std::uint8_t* y, const std::uint8_t* u0, const std::uint8_t* u1,
                         const std::uint8_t* v0, const std::uint8_t* v1, std::uint8_t* dst,
                         int width, const YuvToRgbCoeffs& cf, ChromaLocation loc) {
  bilinear_dispatch<K, P>(y, dst, width, cf, loc, [&](int cx) {
    return Vert{vert(load8_u16(u0 + cx), load8_u16(u1 + cx)),
                vert(load8_u16(v0 + cx), load8_u16(v1 + cx))};
  });
}

template <class K, int P>
void nv12_bilinear_row(const std::uint8_t* y, const std::uint8_t* uv0, const std::uint8_t* uv1,
                       std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf,
                       ChromaLocation loc) {
  const __m128i lo = _mm_set1_epi16(0x00FF);
  bilinear_dispatch<K, P>(y, dst, width, cf, loc, [&](int cx) {
    const __m128i a = loadu(uv0 + 2 * cx), b = loadu(uv1 + 2 * cx);
    return Vert{vert(_mm_and_si128(a, lo), _mm_and_si128(b, lo)),
                vert(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8))};
  });
}

template <class K>
void fill_yuv2rgb(Table& t) {
  t.yuv420_to_packed[kRgb24] = &yuv420_nearest_row<K, kRgb24>;
  t.yuv420_to_packed[kRgba] = &yuv420_nearest_row<K, kRgba>;
  t.yuv420_to_packed[kBgra] = &yuv420_nearest_row<K, kBgra>;
  t.nv12_to_packed[kRgb24] = &nv12_nearest_row<K, kRgb24>;
  t.nv12_to_packed[kRgba] = &nv12_nearest_row<K, kRgba>;
  t.nv12_to_packed[kBgra] = &nv12_nearest_row<K, kBgra>;
  t.yuv420_bilinear_to_packed[kRgb24] = &yuv420_bilinear_row<K, kRgb24>;
  t.yuv420_bilinear_to_packed[kRgba] = &yuv420_bilinear_row<K, kRgba>;
  t.yuv420_bilinear_to_packed[kBgra] = &yuv420_bilinear_row<K, kBgra>;
  t.nv12_bilinear_to_packed[kRgb24] = &nv12_bilinear_row<K, kRgb24>;
  t.nv12_bilinear_to_packed[kRgba] = &nv12_bilinear_row<K, kRgba>;
  t.nv12_bilinear_to_packed[kBgra] = &nv12_bilinear_row<K, kBgra>;
}

}  // namespace
}  // namespace lanecast::kernels::convert
