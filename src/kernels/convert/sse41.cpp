// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
//
// Every kernel works in blocks of 16 pixels and relies on the row padding
// guarantee (kernels.hpp): the last block may read and write past the visible
// width instead of running a scalar tail. Results never depend on the bytes
// read from the padding.
#include <smmintrin.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/x86_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// rgba <-> bgra: swap channels 0 and 2.
constexpr Shuf kSwap02 = {{2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15}};

// rgb24 -> 4 bytes per pixel, for the 4 pixels (12 bytes) at the start of the
// input vector. The alpha byte is zeroed here and ORed in later. Bgr selects
// the bgra channel order.
constexpr Shuf make_expand(bool bgr) {
  Shuf s{};
  for (int q = 0; q < 4; ++q) {
    for (int i = 0; i < 3; ++i) s.b[4 * q + i] = static_cast<std::int8_t>(3 * q + (bgr ? 2 - i : i));
    s.b[4 * q + 3] = -128;
  }
  return s;
}
// 4 bytes per pixel -> rgb24: packs the color bytes of 4 pixels into the low
// 12 bytes.
constexpr Shuf make_compress(bool bgr) {
  Shuf s{};
  for (int q = 0; q < 4; ++q) {
    for (int i = 0; i < 3; ++i) s.b[3 * q + i] = static_cast<std::int8_t>(4 * q + (bgr ? 2 - i : i));
  }
  for (int i = 12; i < 16; ++i) s.b[i] = -128;
  return s;
}
constexpr Shuf kExpand[2] = {make_expand(false), make_expand(true)};
constexpr Shuf kCompress[2] = {make_compress(false), make_compress(true)};

void swap_rb_row(const std::uint8_t* src, std::uint8_t* dst, int width) {
  const __m128i m = load(kSwap02);
  for (int x = 0; x < width; x += 16) {
    for (int k = 0; k < 4; ++k) {
      storeu(dst + 4 * x + 16 * k, _mm_shuffle_epi8(loadu(src + 4 * x + 16 * k), m));
    }
  }
}

// rgb24 -> rgba (Bgr = false) or bgra (Bgr = true), alpha 255. Output group j
// (pixels 4j..4j+3) starts at source byte 12j, which straddles two loads for
// j = 1, 2, so it is realigned first.
template <bool Bgr>
void rgb24_to_4bpp_row(const std::uint8_t* src, std::uint8_t* dst, int width) {
  const __m128i m = load(kExpand[Bgr]);
  const __m128i alpha = _mm_set1_epi32(static_cast<int>(0xFF000000u));
  for (int x = 0; x < width; x += 16) {
    const std::uint8_t* s = src + 3 * x;
    const __m128i s0 = loadu(s), s1 = loadu(s + 16), s2 = loadu(s + 32);
    std::uint8_t* d = dst + 4 * x;
    storeu(d, _mm_or_si128(_mm_shuffle_epi8(s0, m), alpha));
    storeu(d + 16, _mm_or_si128(_mm_shuffle_epi8(_mm_alignr_epi8(s1, s0, 12), m), alpha));
    storeu(d + 32, _mm_or_si128(_mm_shuffle_epi8(_mm_alignr_epi8(s2, s1, 8), m), alpha));
    storeu(d + 48, _mm_or_si128(_mm_shuffle_epi8(_mm_srli_si128(s2, 4), m), alpha));
  }
}

// rgba (Bgr = false) or bgra (Bgr = true) -> rgb24. Each group packs to 12
// bytes; the four are stitched into 48 with byte shifts.
template <bool Bgr>
void from_4bpp_to_rgb24_row(const std::uint8_t* src, std::uint8_t* dst, int width) {
  const __m128i m = load(kCompress[Bgr]);
  for (int x = 0; x < width; x += 16) {
    const std::uint8_t* s = src + 4 * x;
    const __m128i g0 = _mm_shuffle_epi8(loadu(s), m), g1 = _mm_shuffle_epi8(loadu(s + 16), m);
    const __m128i g2 = _mm_shuffle_epi8(loadu(s + 32), m), g3 = _mm_shuffle_epi8(loadu(s + 48), m);
    std::uint8_t* d = dst + 3 * x;
    storeu(d, _mm_or_si128(g0, _mm_slli_si128(g1, 12)));
    storeu(d + 16, _mm_or_si128(_mm_srli_si128(g1, 4), _mm_slli_si128(g2, 8)));
    storeu(d + 32, _mm_or_si128(_mm_srli_si128(g2, 8), _mm_slli_si128(g3, 4)));
  }
}

void interleave_uv_row(const std::uint8_t* u, const std::uint8_t* v, std::uint8_t* uv, int cw) {
  for (int i = 0; i < cw; i += 16) {
    const __m128i a = loadu(u + i), b = loadu(v + i);
    storeu(uv + 2 * i, _mm_unpacklo_epi8(a, b));
    storeu(uv + 2 * i + 16, _mm_unpackhi_epi8(a, b));
  }
}

// Evens to the low half, odds to the high half of 16 bytes.
constexpr Shuf kEvenOdd = {{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}};

void deinterleave_uv_row(const std::uint8_t* uv, std::uint8_t* u, std::uint8_t* v, int cw) {
  const __m128i m = load(kEvenOdd);
  for (int i = 0; i < cw; i += 16) {
    const __m128i a = _mm_shuffle_epi8(loadu(uv + 2 * i), m);
    const __m128i b = _mm_shuffle_epi8(loadu(uv + 2 * i + 16), m);
    storeu(u + i, _mm_unpacklo_epi64(a, b));
    storeu(v + i, _mm_unpackhi_epi64(a, b));
  }
}

template <int P>
void gray_to_packed_row(const std::uint8_t* gray, std::uint8_t* dst, int width) {
  const __m128i opaque = _mm_set1_epi8(-1);
  for (int x = 0; x < width; x += 16) {
    const __m128i g = loadu(gray + x);
    store16<P>(dst + x * Layout<P>::bpp, Planar16{g, g, g, opaque});
  }
}

}  // namespace

void fill_sse41(Table& t) {
  t.swizzle[kRgb24][kRgba] = &rgb24_to_4bpp_row<false>;
  t.swizzle[kRgb24][kBgra] = &rgb24_to_4bpp_row<true>;
  t.swizzle[kRgba][kRgb24] = &from_4bpp_to_rgb24_row<false>;
  t.swizzle[kRgba][kBgra] = &swap_rb_row;
  t.swizzle[kBgra][kRgb24] = &from_4bpp_to_rgb24_row<true>;
  t.swizzle[kBgra][kRgba] = &swap_rb_row;
  t.interleave_uv = &interleave_uv_row;
  t.deinterleave_uv = &deinterleave_uv_row;
  t.gray_to_packed[kRgb24] = &gray_to_packed_row<kRgb24>;
  t.gray_to_packed[kRgba] = &gray_to_packed_row<kRgba>;
  t.gray_to_packed[kBgra] = &gray_to_packed_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
