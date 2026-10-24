// Front end: validation, per-plane row loops, PSNR and SSIM final math.
#include "lanecast/metrics.hpp"

#include <cmath>
#include <limits>

#include "kernels/metrics/kernels.hpp"

namespace lanecast {

double psnr_from_sse(std::uint64_t sse, std::uint64_t samples) noexcept {
  if (sse == 0) return std::numeric_limits<double>::infinity();
  return 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(samples) /
                           static_cast<double>(sse));
}

Status compute_psnr(const Frame& a, const Frame& b, PsnrResult& out) {
  (void)a;
  (void)b;
  (void)out;
  return Status::not_implemented;
}

Status compute_ssim(const Frame& a, const Frame& b, SsimResult& out) {
  (void)a;
  (void)b;
  (void)out;
  return Status::not_implemented;
}

Status compute_sad(const Frame& a, const Frame& b, std::uint64_t& out) {
  (void)a;
  (void)b;
  (void)out;
  return Status::not_implemented;
}

}  // namespace lanecast
