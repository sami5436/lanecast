// Metric cases. Whole-frame reductions: single thread only.
#include "bench.hpp"
#include "lanecast/metrics.hpp"

namespace lanecast::bench {
namespace {

template <class Fn>
Case metric_case(const char* name, PixelFormat fmt, Fn fn) {
  Case c;
  c.name = std::string(name) + " " + std::string(pixfmt_name(fmt));
  c.make = [=](Size s) {
    auto a = make_frame(fmt, s.width, s.height, 7);
    auto b = make_frame(fmt, s.width, s.height, 8);
    Instance in;
    in.bytes = 2 * frame_bytes(*a);
    in.sliceable = false;
    in.run = [a, b, fn](ThreadPool*) { return fn(*a, *b); };
    return in;
  };
  return c;
}

}  // namespace

void register_metrics(std::vector<Case>& out) {
  using F = PixelFormat;
  out.push_back(metric_case("psnr", F::yuv420p, [](const Frame& a, const Frame& b) {
    PsnrResult r;
    return compute_psnr(a, b, r);
  }));
  out.push_back(metric_case("ssim", F::yuv420p, [](const Frame& a, const Frame& b) {
    SsimResult r;
    return compute_ssim(a, b, r);
  }));
  out.push_back(metric_case("sad", F::yuv420p, [](const Frame& a, const Frame& b) {
    std::uint64_t sad = 0;
    return compute_sad(a, b, sad);
  }));
}

}  // namespace lanecast::bench
