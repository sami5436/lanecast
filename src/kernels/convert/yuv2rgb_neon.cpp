// NEON yuv420p / nv12 -> packed RGB row kernels (nearest and bilinear chroma).
// Arithmetic follows the Q13 contract in convert.hpp exactly: int32
// accumulation with one rounding shift (vqrshrun) and saturation to 8 bits.
#include <arm_neon.h>

#include "kernels/convert/kernels.hpp"
#include "kernels/convert/neon_common.hpp"

namespace lanecast::kernels::convert {
namespace {

// Coefficient lanes: cy, crv, cgu, cgv, cbu. Plus the byte constants.
struct Coef {
  int16x8_t k;
  uint8x16_t y_off;
  uint8x16_t c128;
};

inline Coef make_coef(const YuvToRgbCoeffs& c) {
  const std::int16_t a[8] = {c.cy, c.crv, c.cgu, c.cgv, c.cbu, 0, 0, 0};
  return {vld1q_s16(a), vdupq_n_u8(static_cast<std::uint8_t>(c.y_offset)), vdupq_n_u8(128)};
}

// 8 pixels: y, u, v are already offset (Y - y_offset, U - 128, V - 128).
// Results are u16 saturated to [0, 65535]; the caller narrows to u8.
inline void core8(int16x8_t y, int16x8_t u, int16x8_t v, int16x8_t k, uint16x8_t& r, uint16x8_t& g,
                  uint16x8_t& b) {
  const int32x4_t ylo = vmull_laneq_s16(vget_low_s16(y), k, 0);
  const int32x4_t yhi = vmull_high_laneq_s16(y, k, 0);
  const int32x4_t rl = vmlal_laneq_s16(ylo, vget_low_s16(v), k, 1);
  const int32x4_t rh = vmlal_high_laneq_s16(yhi, v, k, 1);
  int32x4_t gl = vmlsl_laneq_s16(ylo, vget_low_s16(u), k, 2);
  int32x4_t gh = vmlsl_high_laneq_s16(yhi, u, k, 2);
  gl = vmlsl_laneq_s16(gl, vget_low_s16(v), k, 3);
  gh = vmlsl_high_laneq_s16(gh, v, k, 3);
  const int32x4_t bl = vmlal_laneq_s16(ylo, vget_low_s16(u), k, 4);
  const int32x4_t bh = vmlal_high_laneq_s16(yhi, u, k, 4);
  r = vqrshrun_high_n_s32(vqrshrun_n_s32(rl, 13), rh, 13);
  g = vqrshrun_high_n_s32(vqrshrun_n_s32(gl, 13), gh, 13);
  b = vqrshrun_high_n_s32(vqrshrun_n_s32(bl, 13), bh, 13);
}

// 16 pixels with full-resolution chroma.
inline Planar16 yuv16(uint8x16_t y, uint8x16_t u, uint8x16_t v, const Coef& c) {
  uint16x8_t rl, gl, bl, rh, gh, bh;
  core8(vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(y), vget_low_u8(c.y_off))),
        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(u), vget_low_u8(c.c128))),
        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(v), vget_low_u8(c.c128))), c.k, rl, gl, bl);
  core8(vreinterpretq_s16_u16(vsubl_high_u8(y, c.y_off)),
        vreinterpretq_s16_u16(vsubl_high_u8(u, c.c128)),
        vreinterpretq_s16_u16(vsubl_high_u8(v, c.c128)), c.k, rh, gh, bh);
  return {vqmovn_high_u16(vqmovn_u16(rl), rh), vqmovn_high_u16(vqmovn_u16(gl), gh),
          vqmovn_high_u16(vqmovn_u16(bl), bh), vdupq_n_u8(255)};
}

// Duplicates 8 chroma bytes to 16 (nearest upsampling).
inline uint8x16_t dup2(uint8x8_t c) { return vcombine_u8(vzip1_u8(c, c), vzip2_u8(c, c)); }

template <int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  const Coef c = make_coef(cf);
  for (int x = 0; x < width; x += 16) {
    const Planar16 o =
        yuv16(vld1q_u8(y + x), dup2(vld1_u8(u + x / 2)), dup2(vld1_u8(v + x / 2)), c);
    store16<P>(dst + x * Layout<P>::bpp, o);
  }
}

template <int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& cf) {
  const Coef c = make_coef(cf);
  for (int x = 0; x < width; x += 16) {
    const uint8x8x2_t p = vld2_u8(uv + x);
    store16<P>(dst + x * Layout<P>::bpp, yuv16(vld1q_u8(y + x), dup2(p.val[0]), dup2(p.val[1]), c));
  }
}

// Vertical pass of the bilinear filter for 8 chroma samples: 3 * a + b.
inline uint16x8_t vert(uint8x8_t a, uint8x8_t b) { return vmlal_u8(vmovl_u8(b), a, vdup_n_u8(3)); }

// Horizontal pass: v is column cx.. and vn the right neighbors. Even luma
// columns take 2 * v, odd ones v + vn; both are rounded once as (h + 4) >> 3.
inline uint8x16_t horiz(uint16x8_t v, uint16x8_t vn) {
  const uint8x8_t even = vrshrn_n_u16(vshlq_n_u16(v, 1), 3);
  const uint8x8_t odd = vrshrn_n_u16(vaddq_u16(v, vn), 3);
  return vcombine_u8(vzip1_u8(even, odd), vzip2_u8(even, odd));
}

// Vertical results for the U and V planes of one 8-sample chroma block.
struct Vert {
  uint16x8_t u, v;
};

// Walks the row 8 chroma samples (16 pixels) at a time. load(cx) returns the
// vertical pass for chroma columns cx..cx+7 and may read past the plane's
// visible width (padding). The right neighbor of the last visible chroma
// column is the column itself, so the final block patches that lane.
template <int P, class Load>
inline void bilinear_row(const std::uint8_t* y, std::uint8_t* dst, int width, const Coef& c,
                         Load load) {
  const int cw = (width + 1) >> 1;
  Vert cur = load(0);
  int cx = 0;
  for (; cx + 8 < cw; cx += 8) {
    const Vert next = load(cx + 8);
    const uint8x16_t u = horiz(cur.u, vextq_u16(cur.u, next.u, 1));
    const uint8x16_t v = horiz(cur.v, vextq_u16(cur.v, next.v, 1));
    store16<P>(dst + 2 * cx * Layout<P>::bpp, yuv16(vld1q_u8(y + 2 * cx), u, v, c));
    cur = next;
  }
  // Last block: lane cw - 1 - cx (0..7) must see itself as its right neighbor.
  static const std::uint16_t kLane[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  const uint16x8_t at_or_after =
      vcgeq_u16(vld1q_u16(kLane), vdupq_n_u16(static_cast<std::uint16_t>(cw - 1 - cx)));
  const uint16x8_t un = vbslq_u16(at_or_after, cur.u, vextq_u16(cur.u, cur.u, 1));
  const uint16x8_t vn = vbslq_u16(at_or_after, cur.v, vextq_u16(cur.v, cur.v, 1));
  store16<P>(dst + 2 * cx * Layout<P>::bpp,
             yuv16(vld1q_u8(y + 2 * cx), horiz(cur.u, un), horiz(cur.v, vn), c));
}

template <int P>
void yuv420_bilinear_row(const std::uint8_t* y, const std::uint8_t* u0, const std::uint8_t* u1,
                         const std::uint8_t* v0, const std::uint8_t* v1, std::uint8_t* dst,
                         int width, const YuvToRgbCoeffs& cf) {
  bilinear_row<P>(y, dst, width, make_coef(cf), [&](int cx) {
    return Vert{vert(vld1_u8(u0 + cx), vld1_u8(u1 + cx)),
                vert(vld1_u8(v0 + cx), vld1_u8(v1 + cx))};
  });
}

template <int P>
void nv12_bilinear_row(const std::uint8_t* y, const std::uint8_t* uv0, const std::uint8_t* uv1,
                       std::uint8_t* dst, int width, const YuvToRgbCoeffs& cf) {
  bilinear_row<P>(y, dst, width, make_coef(cf), [&](int cx) {
    const uint8x8x2_t a = vld2_u8(uv0 + 2 * cx), b = vld2_u8(uv1 + 2 * cx);
    return Vert{vert(a.val[0], b.val[0]), vert(a.val[1], b.val[1])};
  });
}

}  // namespace

void fill_neon_yuv2rgb(Table& t) {
  t.yuv420_to_packed[kRgb24] = &yuv420_nearest_row<kRgb24>;
  t.yuv420_to_packed[kRgba] = &yuv420_nearest_row<kRgba>;
  t.yuv420_to_packed[kBgra] = &yuv420_nearest_row<kBgra>;
  t.nv12_to_packed[kRgb24] = &nv12_nearest_row<kRgb24>;
  t.nv12_to_packed[kRgba] = &nv12_nearest_row<kRgba>;
  t.nv12_to_packed[kBgra] = &nv12_nearest_row<kBgra>;
  t.yuv420_bilinear_to_packed[kRgb24] = &yuv420_bilinear_row<kRgb24>;
  t.yuv420_bilinear_to_packed[kRgba] = &yuv420_bilinear_row<kRgba>;
  t.yuv420_bilinear_to_packed[kBgra] = &yuv420_bilinear_row<kBgra>;
  t.nv12_bilinear_to_packed[kRgb24] = &nv12_bilinear_row<kRgb24>;
  t.nv12_bilinear_to_packed[kRgba] = &nv12_bilinear_row<kRgba>;
  t.nv12_bilinear_to_packed[kBgra] = &nv12_bilinear_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
