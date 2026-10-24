// Per-plane separable scaling, fixed point.
//
// Bit-exact contract. Scaling is horizontal first, then vertical, for each
// plane independently (chroma planes use their own plane sizes; nv12 UV and
// packed RGB scale each interleaved channel independently; alpha is scaled
// like any channel).
//
// Filter table (one per axis, built by build_scale_filter, shared by every
// ISA): for each output index o, taps weights w[o][0..taps) in Q14 that are
// non-negative and sum to exactly 16384, and a start index with
// 0 <= start[o] and start[o] + taps <= src_size.
//
// The weights follow swscale's initFilter (SWS_BILINEAR and SWS_AREA), all in
// integers. The triangle (bilinear) and trapezoid (area) kernels widen with
// the scale ratio when downscaling, so a 3x reduction averages about 6
// source pixels instead of aliasing from 2.
//
//   inc    = ((src << 16) + dst / 2) / dst       // source step, 16.16
//   down   = inc > 65536
//   kernel = area && down ? box : triangle       // area upscale is bilinear
//   width  = down ? 1 + (f * src + dst - 1) / dst : 1 + f, clamped to
//            [1, src]; f = 1 for box, 2 for triangle
//   fone   = 1 << (54 - min(floor(log2(src / dst)), 8))   // 1.0 in coef units
//   x0     = (dst_pos * inc >> 7) - (src_pos * 65536 >> 7)
//   for output o, with x = x0 + 2 * o * inc (2^-17 source pixels):
//     first = (x - (width - 2) * 65536) / 131072        // truncating
//     for j in [0, width), i = first + j:
//       d = (|i * 131072 - x| << 13) * (down ? dst / src : 1)  // 2^30 = 1 px
//       triangle: c = max(2^30 - d, 0) * (fone >> 30)
//       box:      t = (d - 2^29) * inc
//                 c = t < -2^45 ? 2^46 : t < 2^45 ? 2^45 - t : 0, times (fone >> 46)
//   Leading taps are dropped while their running sum is <= fone / 500, and
//   the common width is the widest row after trailing taps whose running sum
//   is <= fone / 500 are dropped. Taps whose index is outside [0, src) merge
//   into the nearest valid index. With sum = round(total c / 2^14), the
//   Q14 weights are round(c + carry / sum) with the rounding error carried to
//   the next tap (round half away from zero); any remaining difference to
//   16384 goes to the largest tap. Columns that are zero in every row are
//   then removed: taps = widest nonzero span, start[o] = min(first nonzero
//   index, src - taps), unused slots are 0.
//
// src_pos and dst_pos place the sample inside its pixel, in 1/256 pixel:
// 128 is the pixel center (luma, and chroma sited at the center), 64 is a
// half-subsampled chroma plane whose samples sit on the left luma column
// (swscale's src_h_chr_pos / dst_h_chr_pos after get_local_pos).
//
// Pass math (int32):
//   mid[x] = (sum_k wx[x][k] * src[start_x[x] + k] + 64) >> 7   // 0..32640, int16
//   out[y] = clamp8((sum_k wy[y][k] * mid_row[start_y[y] + k] + (1 << 20)) >> 21)
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "lanecast/frame.hpp"
#include "lanecast/pixfmt.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

enum class ScaleFilter : std::uint8_t { bilinear, area };

struct ScaleFilterTable {
  int taps = 0;
  std::vector<std::int32_t> start;    // size dst_size
  std::vector<std::int16_t> weights;  // size dst_size * taps, row-major by output
};

// src_size, dst_size in [1, kMaxDimension]; src_pos, dst_pos in [0, 256].
Status build_scale_filter(ScaleFilter filter, int src_size, int dst_size, ScaleFilterTable& out,
                          int src_pos = 128, int dst_pos = 128);

// Prebuilt scaler: filter tables are built once in create(). run() does no
// heap allocation after the first call on each thread (thread-local scratch).
class Scaler {
 public:
  static Status create(PixelFormat format, int src_width, int src_height, int dst_width,
                       int dst_height, ScaleFilter filter, std::unique_ptr<Scaler>& out);
  ~Scaler();
  Scaler(const Scaler&) = delete;
  Scaler& operator=(const Scaler&) = delete;

  // src and dst must match the shapes given to create(). rows selects
  // destination rows. Safe to call concurrently on disjoint row ranges.
  Status run(const Frame& src, Frame& dst, RowRange rows = {}) const;

  struct Impl;

 private:
  explicit Scaler(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Convenience: builds a Scaler and runs it. src and dst share a format;
// dst must be allocated at the target size.
Status scale(const Frame& src, Frame& dst, ScaleFilter filter = ScaleFilter::bilinear,
             RowRange rows = {});

}  // namespace lanecast
