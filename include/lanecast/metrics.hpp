// Quality metrics. Integer sums are bit-exact across ISAs; the final
// floating point math runs in shared scalar code, so results are identical
// on every ISA of the same build.
#pragma once

#include <cstdint>

#include "lanecast/frame.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

// PSNR per plane over every visible byte (packed formats: one plane, all
// channels). psnr = 10 * log10(255^2 * samples / sse), +infinity when
// sse == 0. average uses the summed sse and samples of all planes (ffmpeg's
// "average").
struct PsnrResult {
  int planes = 0;
  std::uint64_t sse[kMaxPlanes] = {};
  std::uint64_t samples[kMaxPlanes] = {};
  double psnr[kMaxPlanes] = {};
  double average = 0.0;
};

// SSIM per plane, x264 / ffmpeg style. Supports gray8 and yuv420p; every
// plane must be at least 8x8 (invalid_argument otherwise).
//   1. Split the plane into 4x4 blocks (bw = w / 4, bh = h / 4, remainder
//      ignored). Per block, int sums: s1 = sum a, s2 = sum b,
//      ss = sum a^2 + sum b^2, s12 = sum a*b.
//   2. Each 8x8 window is a 2x2 group of adjacent blocks, stepping one
//      block: (bw - 1) * (bh - 1) windows; window sums add the 4 blocks.
//   3. Per window in double, c1 = 416, c2 = 235963:
//        vars = ss * 64 - s1^2 - s2^2, covar = s12 * 64 - s1 * s2
//        ssim = (2 s1 s2 + c1) (2 covar + c2) / ((s1^2 + s2^2 + c1) (vars + c2))
//   4. Plane SSIM = mean over windows. average weights planes by sample count.
struct SsimResult {
  int planes = 0;
  double ssim[kMaxPlanes] = {};
  double average = 0.0;
};

double psnr_from_sse(std::uint64_t sse, std::uint64_t samples) noexcept;

// a and b must have the same format and size.
Status compute_psnr(const Frame& a, const Frame& b, PsnrResult& out);
Status compute_ssim(const Frame& a, const Frame& b, SsimResult& out);
Status compute_sad(const Frame& a, const Frame& b, std::uint64_t& out);  // all planes

}  // namespace lanecast
