// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
//
// Every kernel works in blocks of 16 elements and relies on the row padding
// guarantee (kernels.hpp): the last block may read and write past the visible
// width instead of running a scalar tail.
#include <arm_neon.h>

#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {
namespace {

// (x + 128 + ((x + 128) >> 8)) >> 8 per u16 lane, narrowed to u8.
inline uint8x8_t div255(uint16x8_t x) { return vraddhn_u16(x, vrshrq_n_u16(x, 8)); }

inline uint8x16_t div255_pair(uint16x8_t lo, uint16x8_t hi) {
  return vcombine_u8(div255(lo), div255(hi));
}

void blend_row(const std::uint8_t* over, std::uint8_t* base, int pixels) {
  for (int i = 0; i < pixels; i += 16, over += 64, base += 64) {
    const uint8x16x4_t o = vld4q_u8(over);
    uint8x16x4_t b = vld4q_u8(base);
    const uint8x16_t a = o.val[3];
    const uint8x16_t na = vmvnq_u8(a);
    for (int c = 0; c < 3; ++c) {
      const uint16x8_t lo = vmlal_u8(vmull_u8(vget_low_u8(o.val[c]), vget_low_u8(a)),
                                     vget_low_u8(b.val[c]), vget_low_u8(na));
      const uint16x8_t hi = vmlal_high_u8(vmull_high_u8(o.val[c], a), b.val[c], na);
      b.val[c] = div255_pair(lo, hi);
    }
    const uint16x8_t lo = vmull_u8(vget_low_u8(b.val[3]), vget_low_u8(na));
    const uint16x8_t hi = vmull_high_u8(b.val[3], na);
    b.val[3] = vaddq_u8(a, div255_pair(lo, hi));
    vst4q_u8(base, b);
  }
}

template <int R>
inline uint16x8_t hsum8(const std::uint8_t* s, int c) {
  const uint8x16_t t0 = vld1q_u8(s);
  const uint8x16_t t1 = vld1q_u8(s + c);
  const uint8x16_t t2 = vld1q_u8(s + 2 * c);
  if constexpr (R == 1) {
    return vmlal_u8(vaddl_u8(vget_low_u8(t0), vget_low_u8(t2)), vget_low_u8(t1), vdup_n_u8(2));
  } else {
    const uint8x16_t t3 = vld1q_u8(s + 3 * c);
    const uint8x16_t t4 = vld1q_u8(s + 4 * c);
    uint16x8_t v = vmlaq_n_u16(vaddl_u8(vget_low_u8(t0), vget_low_u8(t4)),
                               vaddl_u8(vget_low_u8(t1), vget_low_u8(t3)), 4);
    return vmlal_u8(v, vget_low_u8(t2), vdup_n_u8(6));
  }
}

template <int R>
inline uint16x8_t hsum8_high(const std::uint8_t* s, int c) {
  const uint8x16_t t0 = vld1q_u8(s);
  const uint8x16_t t1 = vld1q_u8(s + c);
  const uint8x16_t t2 = vld1q_u8(s + 2 * c);
  if constexpr (R == 1) {
    return vmlal_high_u8(vaddl_high_u8(t0, t2), t1, vdupq_n_u8(2));
  } else {
    const uint8x16_t t3 = vld1q_u8(s + 3 * c);
    const uint8x16_t t4 = vld1q_u8(s + 4 * c);
    uint16x8_t v = vmlaq_n_u16(vaddl_high_u8(t0, t4), vaddl_high_u8(t1, t3), 4);
    return vmlal_high_u8(v, t2, vdupq_n_u8(6));
  }
}

template <int R>
void blur_h_impl(const std::uint8_t* src, std::uint16_t* dst, int count, int channels) {
  for (int i = 0; i < count; i += 16) {
    vst1q_u16(dst + i, hsum8<R>(src + i, channels));
    vst1q_u16(dst + i + 8, hsum8_high<R>(src + i, channels));
  }
}

template <int R>
inline uint8x8_t vsum8(const std::uint16_t* const* rows, int i) {
  const uint16x8_t r0 = vld1q_u16(rows[0] + i);
  const uint16x8_t r1 = vld1q_u16(rows[1] + i);
  const uint16x8_t r2 = vld1q_u16(rows[2] + i);
  if constexpr (R == 1) {
    return vrshrn_n_u16(vmlaq_n_u16(vaddq_u16(r0, r2), r1, 2), 4);
  } else {
    const uint16x8_t r3 = vld1q_u16(rows[3] + i);
    const uint16x8_t r4 = vld1q_u16(rows[4] + i);
    uint16x8_t v = vmlaq_n_u16(vaddq_u16(r0, r4), vaddq_u16(r1, r3), 4);
    return vrshrn_n_u16(vmlaq_n_u16(v, r2, 6), 8);
  }
}

template <int R>
void blur_v_impl(const std::uint16_t* const* rows, std::uint8_t* dst, int count) {
  for (int i = 0; i < count; i += 16) {
    vst1q_u8(dst + i, vcombine_u8(vsum8<R>(rows, i), vsum8<R>(rows, i + 8)));
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

void fill_neon(Table& t) {
  t.blend = &blend_row;
  t.blur_h = &blur_h_row;
  t.blur_v = &blur_v_row;
}

}  // namespace lanecast::kernels::filter
