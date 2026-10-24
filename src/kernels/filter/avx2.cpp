// AVX2, compiled with -mavx2. Runs on top of the sse41 table.
// Store only the kernels this tier specializes; others are inherited.
//
// Same math as sse41.cpp, twice as wide. Lane crossing rules: byte/word
// unpack and pack work per 128 bit lane and are used in matching pairs so
// pixel order is restored; blur passes widen with vpmovzx instead. Blocks of
// 8 pixels (blend), 16 elements (blur h) or 32 (blur v) rely on the row
// padding guarantee (kernels.hpp).
#include <immintrin.h>

#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {
namespace {

inline __m256i div255(__m256i x) {
  x = _mm256_add_epi16(x, _mm256_set1_epi16(128));
  return _mm256_srli_epi16(_mm256_add_epi16(x, _mm256_srli_epi16(x, 8)), 8);
}

inline __m256i mix(__m256i o, __m256i b, __m256i wo, __m256i wb) {
  return div255(_mm256_add_epi16(_mm256_mullo_epi16(o, wo), _mm256_mullo_epi16(b, wb)));
}

void blend_row(const std::uint8_t* over, std::uint8_t* base, int pixels) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i spread_alpha =
      _mm256_setr_epi8(3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15, 3, 3, 3, 3, 7, 7, 7,
                       7, 11, 11, 11, 11, 15, 15, 15, 15);
  const __m256i color_mask = _mm256_set1_epi32(0x00FFFFFF);
  const __m256i alpha_mask = _mm256_set1_epi32(static_cast<int>(0xFF000000u));
  for (int i = 0; i < pixels; i += 8, over += 32, base += 32) {
    const __m256i o = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(over));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base));
    const __m256i a = _mm256_shuffle_epi8(o, spread_alpha);
    const __m256i wo = _mm256_and_si256(a, color_mask);
    const __m256i wb = _mm256_xor_si256(a, _mm256_set1_epi8(-1));
    const __m256i lo = mix(_mm256_unpacklo_epi8(o, zero), _mm256_unpacklo_epi8(b, zero),
                           _mm256_unpacklo_epi8(wo, zero), _mm256_unpacklo_epi8(wb, zero));
    const __m256i hi = mix(_mm256_unpackhi_epi8(o, zero), _mm256_unpackhi_epi8(b, zero),
                           _mm256_unpackhi_epi8(wo, zero), _mm256_unpackhi_epi8(wb, zero));
    const __m256i r =
        _mm256_add_epi8(_mm256_packus_epi16(lo, hi), _mm256_and_si256(a, alpha_mask));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(base), r);
  }
}

inline __m256i load_u8x16(const std::uint8_t* p) {
  return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

template <int R>
inline __m256i hsum16(const std::uint8_t* s, int c) {
  const __m256i t0 = load_u8x16(s);
  const __m256i t1 = load_u8x16(s + c);
  const __m256i t2 = load_u8x16(s + 2 * c);
  if constexpr (R == 1) {
    return _mm256_add_epi16(_mm256_add_epi16(t0, t2), _mm256_slli_epi16(t1, 1));
  } else {
    const __m256i t3 = load_u8x16(s + 3 * c);
    const __m256i t4 = load_u8x16(s + 4 * c);
    const __m256i outer = _mm256_add_epi16(t0, t4);
    const __m256i mid = _mm256_slli_epi16(_mm256_add_epi16(t1, t3), 2);
    return _mm256_add_epi16(_mm256_add_epi16(outer, mid),
                            _mm256_mullo_epi16(t2, _mm256_set1_epi16(6)));
  }
}

template <int R>
void blur_h_impl(const std::uint8_t* src, std::uint16_t* dst, int count, int channels) {
  for (int i = 0; i < count; i += 16) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), hsum16<R>(src + i, channels));
  }
}

template <int R>
inline __m256i vsum16(const std::uint16_t* const* rows, int i) {
  const auto ld = [&](int k) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[k] + i));
  };
  if constexpr (R == 1) {
    const __m256i v =
        _mm256_add_epi16(_mm256_add_epi16(ld(0), ld(2)), _mm256_slli_epi16(ld(1), 1));
    return _mm256_srli_epi16(_mm256_add_epi16(v, _mm256_set1_epi16(8)), 4);
  } else {
    const __m256i outer = _mm256_add_epi16(ld(0), ld(4));
    const __m256i mid = _mm256_slli_epi16(_mm256_add_epi16(ld(1), ld(3)), 2);
    const __m256i v = _mm256_add_epi16(_mm256_add_epi16(outer, mid),
                                       _mm256_mullo_epi16(ld(2), _mm256_set1_epi16(6)));
    return _mm256_srli_epi16(_mm256_add_epi16(v, _mm256_set1_epi16(128)), 8);
  }
}

template <int R>
void blur_v_impl(const std::uint16_t* const* rows, std::uint8_t* dst, int count) {
  for (int i = 0; i < count; i += 32) {
    // packus interleaves the two inputs per lane (qwords 0 2 1 3); undo it.
    const __m256i p = _mm256_packus_epi16(vsum16<R>(rows, i), vsum16<R>(rows, i + 16));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_permute4x64_epi64(p, 0xD8));
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

void fill_avx2(Table& t) {
  t.blend = &blend_row;
  t.blur_h = &blur_h_row;
  t.blur_v = &blur_v_row;
}

}  // namespace lanecast::kernels::filter
