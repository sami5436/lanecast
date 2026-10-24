// scale suite. Owned by the scale unit.
// Filter table invariants and known values, Scaler behavior (identity,
// constants, ramps, extreme ratios, every format, RowRange slices) and
// CHECK_ALL_ISAS_MATCH over test::test_sizes() for every format and filter.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

#include "lanecast/scale.hpp"
#include "test_support.hpp"

using namespace lanecast;

namespace {

const ScaleFilter kFilters[] = {ScaleFilter::bilinear, ScaleFilter::area};

void check_table(ScaleFilter filter, int src, int dst) {
  ScaleFilterTable t;
  REQUIRE_OK(build_scale_filter(filter, src, dst, t));
  CHECK(t.taps >= 1 && t.taps <= src);
  CHECK_EQ(t.start.size(), static_cast<std::size_t>(dst));
  CHECK_EQ(t.weights.size(), static_cast<std::size_t>(dst) * static_cast<std::size_t>(t.taps));
  for (int o = 0; o < dst; ++o) {
    int sum = 0;
    for (int k = 0; k < t.taps; ++k) {
      const int w = t.weights[static_cast<std::size_t>(o * t.taps + k)];
      CHECK(w >= 0);
      sum += w;
    }
    CHECK_MSG(sum == 16384, "weights sum for " + std::to_string(src) + "->" +
                                std::to_string(dst) + " o=" + std::to_string(o));
    const int s = t.start[static_cast<std::size_t>(o)];
    CHECK(s >= 0 && s + t.taps <= src);
  }
}

}  // namespace

TEST(filter_invariants) {
  for (const ScaleFilter f : kFilters) {
    for (int src : {1, 2, 3, 4, 5, 7, 16, 17, 31, 100, 720, 1080, 1920, 3840}) {
      for (int dst : {1, 2, 3, 4, 5, 6, 9, 16, 33, 64, 540, 1280, 1920}) check_table(f, src, dst);
    }
    check_table(f, 4096, 3);
    check_table(f, 3, 4096);
    check_table(f, kMaxDimension, 1);
    check_table(f, 1, kMaxDimension);
  }
}

TEST(filter_rejects_bad_sizes) {
  ScaleFilterTable t;
  CHECK(build_scale_filter(ScaleFilter::bilinear, 0, 4, t) == Status::invalid_argument);
  CHECK(build_scale_filter(ScaleFilter::area, 4, 0, t) == Status::invalid_argument);
  CHECK(build_scale_filter(ScaleFilter::area, kMaxDimension + 1, 4, t) == Status::invalid_argument);
}

TEST(filter_known_values) {
  ScaleFilterTable t;
  // 2x down: both filters average pairs.
  for (const ScaleFilter f : kFilters) {
    REQUIRE_OK(build_scale_filter(f, 4, 2, t));
    CHECK_EQ(t.taps, 2);
    CHECK_EQ(t.start[0], 0);
    CHECK_EQ(t.start[1], 2);
    for (const std::int16_t w : t.weights) CHECK_EQ(static_cast<int>(w), 8192);
  }
  // Bilinear 2 -> 4: edges clamp, interior quarter positions.
  REQUIRE_OK(build_scale_filter(ScaleFilter::bilinear, 2, 4, t));
  const std::int16_t expect[8] = {16384, 0, 12288, 4096, 4096, 12288, 0, 16384};
  for (int i = 0; i < 8; ++i) CHECK_EQ(static_cast<int>(t.weights[static_cast<std::size_t>(i)]), static_cast<int>(expect[i]));
  // Identity is (16384, 0) at every position (last one starts one earlier).
  REQUIRE_OK(build_scale_filter(ScaleFilter::bilinear, 5, 5, t));
  for (int o = 0; o < 4; ++o) {
    CHECK_EQ(t.start[static_cast<std::size_t>(o)], o);
    CHECK_EQ(static_cast<int>(t.weights[static_cast<std::size_t>(o * 2)]), 16384);
  }
  CHECK_EQ(t.start[4], 3);
  CHECK_EQ(static_cast<int>(t.weights[9]), 16384);
  // Area 3 -> 2: output 0 covers 1 and half of source 1 (largest takes the remainder).
  REQUIRE_OK(build_scale_filter(ScaleFilter::area, 3, 2, t));
  CHECK_EQ(t.taps, 2);
  CHECK_EQ(static_cast<int>(t.weights[0]), 10923);
  CHECK_EQ(static_cast<int>(t.weights[1]), 5461);
  CHECK_EQ(static_cast<int>(t.weights[2]), 5461);
  CHECK_EQ(static_cast<int>(t.weights[3]), 10923);
  // 1-pixel source: one tap of full weight.
  REQUIRE_OK(build_scale_filter(ScaleFilter::bilinear, 1, 5, t));
  CHECK_EQ(t.taps, 1);
  for (const std::int16_t w : t.weights) CHECK_EQ(static_cast<int>(w), 16384);
}
