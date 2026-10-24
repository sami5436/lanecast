// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
//
// Every kernel works in blocks of 16 pixels and relies on the row padding
// guarantee (kernels.hpp): the last block may read and write past the visible
// width instead of running a scalar tail.
#include <arm_neon.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/neon_common.hpp"

namespace lanecast::kernels::convert {
namespace {

template <int S, int D>
void swizzle_row(const std::uint8_t* src, std::uint8_t* dst, int width) {
  for (int x = 0; x < width; x += 16) {
    const Planar16 p = load16<S>(src + x * Layout<S>::bpp);
    store16<D>(dst + x * Layout<D>::bpp, p);
  }
}

void interleave_uv_row(const std::uint8_t* u, const std::uint8_t* v, std::uint8_t* uv, int cw) {
  for (int i = 0; i < cw; i += 16) {
    uint8x16x2_t p;
    p.val[0] = vld1q_u8(u + i);
    p.val[1] = vld1q_u8(v + i);
    vst2q_u8(uv + 2 * i, p);
  }
}

void deinterleave_uv_row(const std::uint8_t* uv, std::uint8_t* u, std::uint8_t* v, int cw) {
  for (int i = 0; i < cw; i += 16) {
    const uint8x16x2_t p = vld2q_u8(uv + 2 * i);
    vst1q_u8(u + i, p.val[0]);
    vst1q_u8(v + i, p.val[1]);
  }
}

template <int P>
void gray_to_packed_row(const std::uint8_t* gray, std::uint8_t* dst, int width) {
  for (int x = 0; x < width; x += 16) {
    const uint8x16_t g = vld1q_u8(gray + x);
    store16<P>(dst + x * Layout<P>::bpp, Planar16{g, g, g, vdupq_n_u8(255)});
  }
}

}  // namespace

void fill_neon(Table& t) {
  t.swizzle[kRgb24][kRgba] = &swizzle_row<kRgb24, kRgba>;
  t.swizzle[kRgb24][kBgra] = &swizzle_row<kRgb24, kBgra>;
  t.swizzle[kRgba][kRgb24] = &swizzle_row<kRgba, kRgb24>;
  t.swizzle[kRgba][kBgra] = &swizzle_row<kRgba, kBgra>;
  t.swizzle[kBgra][kRgb24] = &swizzle_row<kBgra, kRgb24>;
  t.swizzle[kBgra][kRgba] = &swizzle_row<kBgra, kRgba>;
  t.interleave_uv = &interleave_uv_row;
  t.deinterleave_uv = &deinterleave_uv_row;
  t.gray_to_packed[kRgb24] = &gray_to_packed_row<kRgb24>;
  t.gray_to_packed[kRgba] = &gray_to_packed_row<kRgba>;
  t.gray_to_packed[kBgra] = &gray_to_packed_row<kBgra>;
  fill_neon_yuv2rgb(t);
  fill_neon_yuv444(t);
  fill_neon_rgb2yuv(t);
}

}  // namespace lanecast::kernels::convert
