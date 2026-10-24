// Pixel format conversion (colorspace).
//
// Supported pairs (convert_supported() returns true exactly for these):
//   yuv420p, nv12          -> rgb24, rgba, bgra   (YUV to RGB, below)
//   rgb24, rgba, bgra      -> yuv420p, nv12       (RGB to YUV, below)
//   yuv420p <-> nv12                              (chroma (de)interleave, exact)
//   rgb24, rgba, bgra      -> rgb24, rgba, bgra   (swizzle; added alpha = 255)
//   gray8 -> yuv420p, nv12 (Y copied, U = V = 128)
//   yuv420p, nv12 -> gray8 (Y copied, no range change)
//   gray8 -> rgb24, rgba, bgra (R = G = B = Y, no range change)
//   rgb24, rgba, bgra -> gray8 (Y formula below with range = full)
//   any format -> same format (copy)
//
// Bit-exact contract. Every ISA must produce exactly what these integer
// formulas give. All arithmetic is int32; ">>" is an arithmetic shift
// (floor, defined for negative values in C++20); clamp8 clamps to [0, 255].
//
// YUV to RGB, Q13 coefficients c = yuv_to_rgb_coeffs(matrix, range):
//   y = Y - c.y_offset, u = U - 128, v = V - 128
//   R = clamp8((c.cy * y + c.crv * v           + 4096) >> 13)
//   G = clamp8((c.cy * y - c.cgu * u - c.cgv * v + 4096) >> 13)
//   B = clamp8((c.cy * y + c.cbu * u           + 4096) >> 13)
//   Chroma upsampling is nearest: pixel (x, y) uses chroma sample
//   (x >> 1, y >> 1). Every coefficient fits int16 (for madd / vmull).
//
// RGB to YUV, Q15 coefficients k = rgb_to_yuv_coeffs(matrix, range):
//   Y = clamp8((k.yr * R + k.yg * G + k.yb * B + (k.y_offset << 15) + 16384) >> 15)
//   Chroma sample (cx, cy) uses the 2x2 sums SR, SG, SB of pixels
//   (2cx + i, 2cy + j), i, j in {0, 1}; coordinates past the right or bottom
//   edge are clamped to the last column or row (odd sizes replicate):
//   U = clamp8((k.ur * SR + k.ug * SG + k.ub * SB + (128 << 17) + 65536) >> 17)
//   V = clamp8((k.vr * SR + k.vg * SG + k.vb * SB + (128 << 17) + 65536) >> 17)
//   Coefficients were rounded once, then yg, ug, vg were adjusted so that
//   yr + yg + yb == round(32768 * (limited ? 219/255 : 1)) and the U and V
//   rows sum to 0: gray input gives exactly U = V = 128, white gives 235/255.
#pragma once

#include <cstdint>

#include "lanecast/frame.hpp"
#include "lanecast/pixfmt.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

struct ConvertOptions {
  ColorMatrix matrix = ColorMatrix::bt601;
  ColorRange range = ColorRange::limited;  // range of the YUV side
};

struct YuvToRgbCoeffs {
  std::int16_t y_offset;  // 16 limited, 0 full
  std::int16_t cy, crv, cgu, cgv, cbu;
};

struct RgbToYuvCoeffs {
  std::int16_t y_offset;  // 16 limited, 0 full
  std::int16_t yr, yg, yb;
  std::int16_t ur, ug, ub;
  std::int16_t vr, vg, vb;
};

// Derived from Kr, Kb (BT.601: 0.299, 0.114; BT.709: 0.2126, 0.0722) with
// limited scale 255/219 (luma) and 255/224 (chroma).
constexpr YuvToRgbCoeffs yuv_to_rgb_coeffs(ColorMatrix m, ColorRange r) noexcept {
  if (m == ColorMatrix::bt601) {
    return r == ColorRange::limited ? YuvToRgbCoeffs{16, 9539, 13075, 3209, 6660, 16525}
                                    : YuvToRgbCoeffs{0, 8192, 11485, 2819, 5850, 14516};
  }
  return r == ColorRange::limited ? YuvToRgbCoeffs{16, 9539, 14686, 1747, 4366, 17305}
                                  : YuvToRgbCoeffs{0, 8192, 12901, 1535, 3835, 15201};
}

constexpr RgbToYuvCoeffs rgb_to_yuv_coeffs(ColorMatrix m, ColorRange r) noexcept {
  if (m == ColorMatrix::bt601) {
    return r == ColorRange::limited
               ? RgbToYuvCoeffs{16, 8414, 16520, 3208, -4857, -9535, 14392, 14392, -12051, -2341}
               : RgbToYuvCoeffs{0, 9798, 19234, 3736, -5529, -10855, 16384, 16384, -13720, -2664};
  }
  return r == ColorRange::limited
             ? RgbToYuvCoeffs{16, 5983, 20127, 2032, -3298, -11094, 14392, 14392, -13072, -1320}
             : RgbToYuvCoeffs{0, 6966, 23436, 2366, -3754, -12630, 16384, 16384, -14882, -1502};
}

bool convert_supported(PixelFormat src, PixelFormat dst) noexcept;

// Converts src into dst. dst must already be allocated with the target
// format and the same width and height as src (invalid_argument otherwise).
// rows selects destination rows; see RowRange. No allocation.
Status convert(const Frame& src, Frame& dst, const ConvertOptions& opts = {},
               RowRange rows = {});

}  // namespace lanecast
