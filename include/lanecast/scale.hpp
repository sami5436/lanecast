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
//   bilinear (center aligned, like swscale SWS_BILINEAR without corner
//   alignment). int64 math:
//     num = (2o + 1) * src - dst          // 2 * dst * source center
//     p   = num <= 0 ? 0 : (num << 14) / (2 * dst)
//     i0  = p >> 14, f = p & 16383
//     if i0 >= src - 1: i0 = src - 1, f = 0
//     ideal taps: (i0, 16384 - f), (i0 + 1, f)
//   area (box coverage, like SWS_AREA for downscale). Output o covers
//   [o * src, (o + 1) * src) and source i covers [i * dst, (i + 1) * dst):
//     overlap_i = max(0, min((o+1)*src, (i+1)*dst) - max(o*src, i*dst))
//     w_i = (overlap_i * 16384 + src / 2) / src
//     then add (16384 - sum w_i) to the largest w_i (lowest i on ties).
//   Packing: taps = max contributing taps over all o (bilinear: min(2, src)).
//   Taps whose index is outside [0, src) merge into the nearest valid index.
//   start[o] = min(first contributing index, src - taps); unused slots are 0.
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

// src_size, dst_size in [1, kMaxDimension].
Status build_scale_filter(ScaleFilter filter, int src_size, int dst_size, ScaleFilterTable& out);

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
