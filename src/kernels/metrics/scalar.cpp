// Scalar reference kernels for metrics. Defines the behavior every SIMD tier
// must match bit for bit (formulas in include/lanecast/metrics.hpp).
#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {
namespace {

std::uint64_t sse_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t sum = 0;
  for (int i = 0; i < count; ++i) {
    const int d = a[i] - b[i];
    sum += static_cast<std::uint64_t>(d * d);
  }
  return sum;
}

std::uint64_t sad_row(const std::uint8_t* a, const std::uint8_t* b, int count) {
  std::uint64_t sum = 0;
  for (int i = 0; i < count; ++i) {
    const int d = a[i] - b[i];
    sum += static_cast<std::uint64_t>(d < 0 ? -d : d);
  }
  return sum;
}

void ssim_4x4_row(const std::uint8_t* a, std::ptrdiff_t a_stride, const std::uint8_t* b,
                  std::ptrdiff_t b_stride, int blocks, std::int32_t (*sums)[4]) {
  for (int i = 0; i < blocks; ++i) {
    std::int32_t s1 = 0, s2 = 0, ss = 0, s12 = 0;
    for (int y = 0; y < 4; ++y) {
      const std::uint8_t* pa = a + y * a_stride + i * 4;
      const std::uint8_t* pb = b + y * b_stride + i * 4;
      for (int x = 0; x < 4; ++x) {
        s1 += pa[x];
        s2 += pb[x];
        ss += pa[x] * pa[x] + pb[x] * pb[x];
        s12 += pa[x] * pb[x];
      }
    }
    sums[i][0] = s1;
    sums[i][1] = s2;
    sums[i][2] = ss;
    sums[i][3] = s12;
  }
}

}  // namespace

void fill_scalar(Table& t) {
  t.sse = &sse_row;
  t.sad = &sad_row;
  t.ssim_4x4 = &ssim_4x4_row;
}

}  // namespace lanecast::kernels::metrics
