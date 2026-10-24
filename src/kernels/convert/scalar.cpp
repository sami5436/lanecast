// Scalar reference kernels for convert. This file defines the behavior every
// SIMD tier must match bit for bit (formulas in include/lanecast/convert.hpp).
#include <algorithm>

#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {
namespace {

inline std::uint8_t clamp8(int v) {
  return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// Byte offsets of R, G, B, A inside a pixel, and the pixel size.
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

template <int P>
inline void store_rgb(std::uint8_t* d, int yv, int u, int v, const YuvToRgbCoeffs& c) {
  using L = Layout<P>;
  u -= 128;
  v -= 128;
  const int ly = c.cy * (yv - c.y_offset) + 4096;
  d[L::r] = clamp8((ly + c.crv * v) >> 13);
  d[L::g] = clamp8((ly - c.cgu * u - c.cgv * v) >> 13);
  d[L::b] = clamp8((ly + c.cbu * u) >> 13);
  if constexpr (L::a >= 0) d[L::a] = 255;
}

// Bilinear chroma for pixel x (formula in convert.hpp). cx is the chroma
// column of x, cxn / cxo the right and side neighbor columns (already
// clamped); a is the near row, b the vertical neighbor row.
inline int bilinear(const std::uint8_t* a, const std::uint8_t* b, int cx, int cxn, int cxo,
                    bool odd, ChromaLocation loc) {
  const int v0 = 3 * a[cx] + b[cx];
  if (loc == ChromaLocation::center) return (3 * v0 + 3 * a[cxo] + b[cxo] + 8) >> 4;
  const int h = odd ? v0 + 3 * a[cxn] + b[cxn] : 2 * v0;
  return (h + 4) >> 3;
}

template <int P>
void yuv420_nearest_row(const std::uint8_t* y, const std::uint8_t* u, const std::uint8_t* v,
                        std::uint8_t* dst, int width, const YuvToRgbCoeffs& c) {
  for (int x = 0; x < width; ++x) {
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], u[x >> 1], v[x >> 1], c);
  }
}

template <int P>
void nv12_nearest_row(const std::uint8_t* y, const std::uint8_t* uv, std::uint8_t* dst, int width,
                      const YuvToRgbCoeffs& c) {
  for (int x = 0; x < width; ++x) {
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], uv[(x >> 1) * 2], uv[(x >> 1) * 2 + 1], c);
  }
}

template <int P>
void yuv420_bilinear_row(const std::uint8_t* y, const std::uint8_t* u0, const std::uint8_t* u1,
                         const std::uint8_t* v0, const std::uint8_t* v1, std::uint8_t* dst,
                         int width, const YuvToRgbCoeffs& c, ChromaLocation loc) {
  const int cw = (width + 1) >> 1;
  for (int x = 0; x < width; ++x) {
    const int cx = x >> 1, cxn = std::min(cx + 1, cw - 1);
    const bool odd = (x & 1) != 0;
    const int cxo = odd ? cxn : std::max(cx - 1, 0);
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], bilinear(u0, u1, cx, cxn, cxo, odd, loc),
                 bilinear(v0, v1, cx, cxn, cxo, odd, loc), c);
  }
}

template <int P>
void nv12_bilinear_row(const std::uint8_t* y, const std::uint8_t* uv0, const std::uint8_t* uv1,
                       std::uint8_t* dst, int width, const YuvToRgbCoeffs& c, ChromaLocation loc) {
  const int cw = (width + 1) >> 1;
  for (int x = 0; x < width; ++x) {
    const int cx = x >> 1, cxn = std::min(cx + 1, cw - 1);
    const bool odd = (x & 1) != 0;
    const int cxo = odd ? cxn : std::max(cx - 1, 0);
    const int i = cx * 2, in = cxn * 2, io = cxo * 2;
    store_rgb<P>(dst + x * Layout<P>::bpp, y[x], bilinear(uv0, uv1, i, in, io, odd, loc),
                 bilinear(uv0 + 1, uv1 + 1, i, in, io, odd, loc), c);
  }
}

template <int P>
inline int luma(const std::uint8_t* p, const RgbToYuvCoeffs& k) {
  using L = Layout<P>;
  return clamp8((k.yr * p[L::r] + k.yg * p[L::g] + k.yb * p[L::b] + (k.y_offset << 15) + 16384) >>
                15);
}

// Chroma sample cx from the box (center) or [1 2 1] (left) sums in
// convert.hpp; edge columns are clamped. s0, s1 are the two source rows,
// either may repeat.
template <int P>
inline void chroma_at(const std::uint8_t* s0, const std::uint8_t* s1, int cx, int width,
                      const RgbToYuvCoeffs& k, ChromaLocation loc, int& u, int& v) {
  using L = Layout<P>;
  const int x0 = cx * 2, x1 = std::min(x0 + 1, width - 1);
  int sr = 0, sg = 0, sb = 0;
  int shift = 17;
  if (loc == ChromaLocation::center) {
    const std::uint8_t* const q[4] = {s0 + x0 * L::bpp, s0 + x1 * L::bpp, s1 + x0 * L::bpp,
                                      s1 + x1 * L::bpp};
    for (const std::uint8_t* p : q) {
      sr += p[L::r];
      sg += p[L::g];
      sb += p[L::b];
    }
  } else {
    shift = 18;
    const int xm = std::max(x0 - 1, 0);
    for (const std::uint8_t* s : {s0, s1}) {
      const std::uint8_t* const q[3] = {s + xm * L::bpp, s + x0 * L::bpp, s + x1 * L::bpp};
      const int w[3] = {1, 2, 1};
      for (int t = 0; t < 3; ++t) {
        sr += w[t] * q[t][L::r];
        sg += w[t] * q[t][L::g];
        sb += w[t] * q[t][L::b];
      }
    }
  }
  const int bias = (128 << shift) + (1 << (shift - 1));
  u = clamp8((k.ur * sr + k.ug * sg + k.ub * sb + bias) >> shift);
  v = clamp8((k.vr * sr + k.vg * sg + k.vb * sb + bias) >> shift);
}

template <int P>
void luma_rows(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0, std::uint8_t* y1,
               int width, const RgbToYuvCoeffs& k) {
  for (int x = 0; x < width; ++x) {
    y0[x] = static_cast<std::uint8_t>(luma<P>(s0 + x * Layout<P>::bpp, k));
    y1[x] = static_cast<std::uint8_t>(luma<P>(s1 + x * Layout<P>::bpp, k));
  }
}

template <int P>
void packed_to_yuv420_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                          std::uint8_t* y1, std::uint8_t* u, std::uint8_t* v, int width,
                          const RgbToYuvCoeffs& k, ChromaLocation loc) {
  luma_rows<P>(s0, s1, y0, y1, width, k);
  for (int cx = 0; cx < (width + 1) / 2; ++cx) {
    int cu, cv;
    chroma_at<P>(s0, s1, cx, width, k, loc, cu, cv);
    u[cx] = static_cast<std::uint8_t>(cu);
    v[cx] = static_cast<std::uint8_t>(cv);
  }
}

template <int P>
void packed_to_nv12_row(const std::uint8_t* s0, const std::uint8_t* s1, std::uint8_t* y0,
                        std::uint8_t* y1, std::uint8_t* uv, int width, const RgbToYuvCoeffs& k,
                        ChromaLocation loc) {
  luma_rows<P>(s0, s1, y0, y1, width, k);
  for (int cx = 0; cx < (width + 1) / 2; ++cx) {
    int cu, cv;
    chroma_at<P>(s0, s1, cx, width, k, loc, cu, cv);
    uv[cx * 2] = static_cast<std::uint8_t>(cu);
    uv[cx * 2 + 1] = static_cast<std::uint8_t>(cv);
  }
}

// Packed to packed: reorder channels, alpha = 255 when added, dropped when not.
template <int S, int D>
void swizzle_row(const std::uint8_t* src, std::uint8_t* dst, int width) {
  using LS = Layout<S>;
  using LD = Layout<D>;
  for (int x = 0; x < width; ++x) {
    const std::uint8_t* s = src + x * LS::bpp;
    std::uint8_t* d = dst + x * LD::bpp;
    const std::uint8_t r = s[LS::r], g = s[LS::g], b = s[LS::b];
    d[LD::r] = r;
    d[LD::g] = g;
    d[LD::b] = b;
    if constexpr (LD::a >= 0) d[LD::a] = LS::a >= 0 ? s[LS::a] : 255;
  }
}

void interleave_uv_row(const std::uint8_t* u, const std::uint8_t* v, std::uint8_t* uv, int cw) {
  for (int i = 0; i < cw; ++i) {
    uv[2 * i] = u[i];
    uv[2 * i + 1] = v[i];
  }
}

void deinterleave_uv_row(const std::uint8_t* uv, std::uint8_t* u, std::uint8_t* v, int cw) {
  for (int i = 0; i < cw; ++i) {
    u[i] = uv[2 * i];
    v[i] = uv[2 * i + 1];
  }
}

template <int P>
void packed_to_gray_row(const std::uint8_t* src, std::uint8_t* gray, int width,
                        const RgbToYuvCoeffs& k) {
  for (int x = 0; x < width; ++x) {
    gray[x] = static_cast<std::uint8_t>(luma<P>(src + x * Layout<P>::bpp, k));
  }
}

template <int P>
void gray_to_packed_row(const std::uint8_t* gray, std::uint8_t* dst, int width) {
  using L = Layout<P>;
  for (int x = 0; x < width; ++x) {
    std::uint8_t* d = dst + x * L::bpp;
    d[L::r] = d[L::g] = d[L::b] = gray[x];
    if constexpr (L::a >= 0) d[L::a] = 255;
  }
}

}  // namespace

void fill_scalar(Table& t) {
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

  fill_scalar_yuv444(t);

  t.packed_to_yuv420[kRgb24] = &packed_to_yuv420_row<kRgb24>;
  t.packed_to_yuv420[kRgba] = &packed_to_yuv420_row<kRgba>;
  t.packed_to_yuv420[kBgra] = &packed_to_yuv420_row<kBgra>;
  t.packed_to_nv12[kRgb24] = &packed_to_nv12_row<kRgb24>;
  t.packed_to_nv12[kRgba] = &packed_to_nv12_row<kRgba>;
  t.packed_to_nv12[kBgra] = &packed_to_nv12_row<kBgra>;

  t.swizzle[kRgb24][kRgba] = &swizzle_row<kRgb24, kRgba>;
  t.swizzle[kRgb24][kBgra] = &swizzle_row<kRgb24, kBgra>;
  t.swizzle[kRgba][kRgb24] = &swizzle_row<kRgba, kRgb24>;
  t.swizzle[kRgba][kBgra] = &swizzle_row<kRgba, kBgra>;
  t.swizzle[kBgra][kRgb24] = &swizzle_row<kBgra, kRgb24>;
  t.swizzle[kBgra][kRgba] = &swizzle_row<kBgra, kRgba>;

  t.interleave_uv = &interleave_uv_row;
  t.deinterleave_uv = &deinterleave_uv_row;

  t.packed_to_gray[kRgb24] = &packed_to_gray_row<kRgb24>;
  t.packed_to_gray[kRgba] = &packed_to_gray_row<kRgba>;
  t.packed_to_gray[kBgra] = &packed_to_gray_row<kBgra>;
  t.gray_to_packed[kRgb24] = &gray_to_packed_row<kRgb24>;
  t.gray_to_packed[kRgba] = &gray_to_packed_row<kRgba>;
  t.gray_to_packed[kBgra] = &gray_to_packed_row<kBgra>;
}

}  // namespace lanecast::kernels::convert
