// Pixel filters. All integer, bit-exact across ISAs.
#pragma once

#include <array>
#include <cstdint>

#include "lanecast/frame.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

// div255(x) = (x + 128 + ((x + 128) >> 8)) >> 8, equal to round(x / 255)
// for x in [0, 65025].
//
// Alpha blend ("over", straight alpha). overlay and base share a format,
// rgba or bgra, and size. For each pixel, a = overlay alpha:
//   color c: base_c = div255(over_c * a + base_c * (255 - a))
//   alpha:   base_a = a + div255(base_a * (255 - a))
// In place on base. rows selects base rows.
Status alpha_blend(const Frame& overlay, Frame& base, RowRange rows = {});

// Binomial blur, separable, radius 1 (3x3, taps 1 2 1) or 2 (5x5, taps
// 1 4 6 4 1). Edges replicate (coordinates clamp to the plane). Every
// plane and interleaved channel independently, alpha included.
//   h = sum_k b_k * src[clamp(x + k - r)]                (exact, <= 4080)
//   v = sum_k b_k * h[clamp(y + k - r)]                  (exact, <= 65280)
//   out = (v + (radius == 1 ? 8 : 128)) >> (radius == 1 ? 4 : 8)
// src and dst share format and size and must be different frames.
Status blur(const Frame& src, Frame& dst, int radius, RowRange rows = {});

// Brightness and contrast via a 256-entry LUT.
//   t = ((v - 128) * contrast_q8 + 128) >> 8
//   lut[v] = clamp8(128 + t + brightness)
// brightness in [-255, 255]; contrast_q8 in [0, 1024] (256 = unchanged).
Status build_levels_lut(int brightness, int contrast_q8, std::array<std::uint8_t, 256>& lut);

// Applies the LUT to luma for gray8, yuv420p, nv12 (chroma copied), to every
// byte for rgb24, and to bytes 0..2 for rgba, bgra (alpha copied).
// src may be the same frame as dst (in place).
Status brightness_contrast(const Frame& src, Frame& dst, int brightness, int contrast_q8,
                           RowRange rows = {});

}  // namespace lanecast
