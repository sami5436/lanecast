// Front end: validation, per-plane row loops, PSNR and SSIM final math.
#include "lanecast/metrics.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "kernels/metrics/kernels.hpp"

namespace lanecast {

double psnr_from_sse(std::uint64_t sse, std::uint64_t samples) noexcept {
  if (sse == 0) return std::numeric_limits<double>::infinity();
  return 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(samples) /
                           static_cast<double>(sse));
}

namespace {

Status check_pair(const Frame& a, const Frame& b) {
  if (a.empty() || b.empty() || !a.same_shape(b)) return Status::invalid_argument;
  return Status::ok;
}

}  // namespace

Status compute_psnr(const Frame& a, const Frame& b, PsnrResult& out) {
  if (Status s = check_pair(a, b); s != Status::ok) return s;
  const auto sse = kernels::metrics::active_table().sse;
  out = {};
  out.planes = a.plane_count();
  std::uint64_t total_sse = 0;
  std::uint64_t total_samples = 0;
  for (int p = 0; p < out.planes; ++p) {
    const auto pi = static_cast<std::size_t>(p);
    const Plane pa = a.plane(p);
    const Plane pb = b.plane(p);
    std::uint64_t sum = 0;
    for (int y = 0; y < pa.height; ++y) sum += sse(pa.row(y), pb.row(y), pa.row_bytes());
    out.sse[pi] = sum;
    out.samples[pi] = static_cast<std::uint64_t>(pa.row_bytes()) * static_cast<std::uint64_t>(pa.height);
    out.psnr[pi] = psnr_from_sse(out.sse[pi], out.samples[pi]);
    total_sse += out.sse[pi];
    total_samples += out.samples[pi];
  }
  out.average = psnr_from_sse(total_sse, total_samples);
  return Status::ok;
}

Status compute_ssim(const Frame& a, const Frame& b, SsimResult& out) {
  if (Status s = check_pair(a, b); s != Status::ok) return s;
  if (a.format() != PixelFormat::gray8 && a.format() != PixelFormat::yuv420p) {
    return Status::unsupported_format;
  }
  for (int p = 0; p < a.plane_count(); ++p) {
    if (a.plane(p).width < 8 || a.plane(p).height < 8) return Status::invalid_argument;
  }
  const auto ssim_4x4 = kernels::metrics::active_table().ssim_4x4;
  constexpr double kC1 = 416.0;     // (0.01 * 255)^2 * 64
  constexpr double kC2 = 235963.0;  // (0.03 * 255)^2 * 64 * 63
  out = {};
  out.planes = a.plane_count();
  double weighted = 0.0;
  double total_samples = 0.0;
  for (int p = 0; p < out.planes; ++p) {
    const Plane pa = a.plane(p);
    const Plane pb = b.plane(p);
    const int bw = pa.width / 4;
    const int bh = pa.height / 4;
    struct Sums {
      std::int32_t v[4];
    };
    static_assert(sizeof(Sums) == 4 * sizeof(std::int32_t));
    std::vector<Sums> rows(static_cast<std::size_t>(2 * bw));
    Sums* prev = rows.data();
    Sums* cur = rows.data() + bw;
    double total = 0.0;
    for (int by = 0; by < bh; ++by) {
      ssim_4x4(pa.row(4 * by), pa.stride, pb.row(4 * by), pb.stride, bw,
               reinterpret_cast<std::int32_t(*)[4]>(cur));
      if (by > 0) {
        for (int bx = 0; bx + 1 < bw; ++bx) {
          std::int64_t w[4];
          for (int k = 0; k < 4; ++k) {
            w[k] = std::int64_t{prev[bx].v[k]} + prev[bx + 1].v[k] + cur[bx].v[k] + cur[bx + 1].v[k];
          }
          const std::int64_t vars = w[2] * 64 - w[0] * w[0] - w[1] * w[1];
          const std::int64_t covar = w[3] * 64 - w[0] * w[1];
          const double num = (2.0 * static_cast<double>(w[0] * w[1]) + kC1) *
                             (2.0 * static_cast<double>(covar) + kC2);
          const double den = (static_cast<double>(w[0] * w[0] + w[1] * w[1]) + kC1) *
                             (static_cast<double>(vars) + kC2);
          total += num / den;
        }
      }
      std::swap(prev, cur);
    }
    const double windows = static_cast<double>(bw - 1) * static_cast<double>(bh - 1);
    const double samples = static_cast<double>(pa.width) * static_cast<double>(pa.height);
    out.ssim[static_cast<std::size_t>(p)] = total / windows;
    weighted += out.ssim[static_cast<std::size_t>(p)] * samples;
    total_samples += samples;
  }
  out.average = weighted / total_samples;
  return Status::ok;
}

Status compute_sad(const Frame& a, const Frame& b, std::uint64_t& out) {
  if (Status s = check_pair(a, b); s != Status::ok) return s;
  const auto sad = kernels::metrics::active_table().sad;
  std::uint64_t sum = 0;
  for (int p = 0; p < a.plane_count(); ++p) {
    const Plane pa = a.plane(p);
    const Plane pb = b.plane(p);
    for (int y = 0; y < pa.height; ++y) sum += sad(pa.row(y), pb.row(y), pa.row_bytes());
  }
  out = sum;
  return Status::ok;
}

}  // namespace lanecast
