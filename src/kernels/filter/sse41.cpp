// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
//
// Blocks of 4 pixels (blend), 8 elements (blur h) or 16 (blur v) rely on the row padding
// guarantee (kernels.hpp). The brightness_contrast LUT is left to the scalar
// kernel: a 256-entry byte table needs 16 pshufb per 16 bytes without a gather,
// which does not beat two scalar loads per byte.
#include <smmintrin.h>

#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {
namespace {

// (x + 128 + ((x + 128) >> 8)) >> 8 per u16 lane; x <= 65025 so nothing wraps.
inline __m128i div255(__m128i x) {
  x = _mm_add_epi16(x, _mm_set1_epi16(128));
  return _mm_srli_epi16(_mm_add_epi16(x, _mm_srli_epi16(x, 8)), 8);
}

// o * wo + b * wb per u16 lane, where wo + wb <= 255 so the sum fits.
inline __m128i mix(__m128i o, __m128i b, __m128i wo, __m128i wb) {
  return div255(_mm_add_epi16(_mm_mullo_epi16(o, wo), _mm_mullo_epi16(b, wb)));
}

void blend_row(const std::uint8_t* over, std::uint8_t* base, int pixels) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i spread_alpha = _mm_setr_epi8(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15);
  const __m128i color_mask = _mm_set1_epi32(0x00FFFFFF);
  const __m128i alpha_mask = _mm_set1_epi32(static_cast<int>(0xFF000000u));
  for (int i = 0; i < pixels; i += 4, over += 16, base += 16) {
    const __m128i o = _mm_loadu_si128(reinterpret_cast<const __m128i*>(over));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base));
    // Per pixel: color lanes weigh over by a, alpha lanes not at all; every
    // lane weighs base by 255 - a. The alpha lane then adds a back.
    const __m128i a = _mm_shuffle_epi8(o, spread_alpha);
    const __m128i wo = _mm_and_si128(a, color_mask);
    const __m128i wb = _mm_xor_si128(a, _mm_set1_epi8(-1));
    const __m128i lo = mix(_mm_unpacklo_epi8(o, zero), _mm_unpacklo_epi8(b, zero),
                           _mm_unpacklo_epi8(wo, zero), _mm_unpacklo_epi8(wb, zero));
    const __m128i hi = mix(_mm_unpackhi_epi8(o, zero), _mm_unpackhi_epi8(b, zero),
                           _mm_unpackhi_epi8(wo, zero), _mm_unpackhi_epi8(wb, zero));
    const __m128i r = _mm_add_epi8(_mm_packus_epi16(lo, hi), _mm_and_si128(a, alpha_mask));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(base), r);
  }
}

inline __m128i load_u8x8(const std::uint8_t* p) {
  return _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

template <int R>
inline __m128i hsum8(const std::uint8_t* s, int c) {
  const __m128i t0 = load_u8x8(s);
  const __m128i t1 = load_u8x8(s + c);
  const __m128i t2 = load_u8x8(s + 2 * c);
  if constexpr (R == 1) {
    return _mm_add_epi16(_mm_add_epi16(t0, t2), _mm_slli_epi16(t1, 1));
  } else {
    const __m128i t3 = load_u8x8(s + 3 * c);
    const __m128i t4 = load_u8x8(s + 4 * c);
    const __m128i outer = _mm_add_epi16(t0, t4);
    const __m128i mid = _mm_slli_epi16(_mm_add_epi16(t1, t3), 2);
    return _mm_add_epi16(_mm_add_epi16(outer, mid), _mm_mullo_epi16(t2, _mm_set1_epi16(6)));
  }
}

template <int R>
void blur_h_impl(const std::uint8_t* src, std::uint16_t* dst, int count, int channels) {
  for (int i = 0; i < count; i += 8) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), hsum8<R>(src + i, channels));
  }
}

template <int R>
inline __m128i vsum8(const std::uint16_t* const* rows, int i) {
  const auto ld = [&](int k) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[k] + i));
  };
  if constexpr (R == 1) {
    const __m128i v = _mm_add_epi16(_mm_add_epi16(ld(0), ld(2)), _mm_slli_epi16(ld(1), 1));
    return _mm_srli_epi16(_mm_add_epi16(v, _mm_set1_epi16(8)), 4);
  } else {
    const __m128i outer = _mm_add_epi16(ld(0), ld(4));
    const __m128i mid = _mm_slli_epi16(_mm_add_epi16(ld(1), ld(3)), 2);
    const __m128i v = _mm_add_epi16(_mm_add_epi16(outer, mid), _mm_mullo_epi16(ld(2), _mm_set1_epi16(6)));
    return _mm_srli_epi16(_mm_add_epi16(v, _mm_set1_epi16(128)), 8);
  }
}

template <int R>
void blur_v_impl(const std::uint16_t* const* rows, std::uint8_t* dst, int count) {
  for (int i = 0; i < count; i += 16) {
    const __m128i r = _mm_packus_epi16(vsum8<R>(rows, i), vsum8<R>(rows, i + 8));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), r);
  }
}

void blur_h_row(const std::uint8_t* src, std::uint16_t* dst, int count, int channels,
                int radius) {
  if (radius == 1) {
    blur_h_impl<1>(src, dst, count, channels);
  } else {
    blur_h_impl<2>(src, dst, count, channels);
  }
}

void blur_v_row(const std::uint16_t* const* rows, std::uint8_t* dst, int count, int radius) {
  if (radius == 1) {
    blur_v_impl<1>(rows, dst, count);
  } else {
    blur_v_impl<2>(rows, dst, count);
  }
}

}  // namespace

void fill_sse41(Table& t) {
  t.blend = &blend_row;
  t.blur_h = &blur_h_row;
  t.blur_v = &blur_v_row;
}

}  // namespace lanecast::kernels::filter
