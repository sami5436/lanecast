// metrics suite. Owned by the metrics unit.
// Required: sse, sad and psnr known values; ssim of identical planes is 1;
// ssim matches a straightforward double reference of the spec; planes under
// 8x8 rejected; isa_value_mismatch over test::test_sizes() for every kernel.
#include <cmath>

#include "lanecast/metrics.hpp"
#include "test_support.hpp"

using namespace lanecast;

TEST(psnr_from_sse_formula) {
  CHECK(std::isinf(psnr_from_sse(0, 100)));
  CHECK(std::abs(psnr_from_sse(65025, 1) - 0.0) < 1e-12);
  CHECK(std::abs(psnr_from_sse(1, 1) - 20.0 * std::log10(255.0)) < 1e-9);
}
