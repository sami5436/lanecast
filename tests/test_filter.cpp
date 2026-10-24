// filter suite. Owned by the filter unit.
#include <cstdlib>

#include "lanecast/filter.hpp"
#include "test_support.hpp"

using namespace lanecast;

namespace {

Frame make_frame(PixelFormat fmt, int w, int h) {
  Frame f;
  REQUIRE_OK(Frame::allocate(fmt, w, h, f));
  return f;
}

int div255_ref(int x) { return (x + 127) / 255; }

constexpr PixelFormat kAlphaFormats[] = {PixelFormat::rgba, PixelFormat::bgra};

}  // namespace

TEST(blend_known_values) {
  Frame over = make_frame(PixelFormat::rgba, 4, 1);
  Frame base = make_frame(PixelFormat::rgba, 4, 1);
  const std::uint8_t o[16] = {200, 100, 50, 0, 200, 100, 50, 255, 200, 100, 50, 128, 255, 0, 0, 1};
  const std::uint8_t b[16] = {10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 255};
  for (int i = 0; i < 16; ++i) {
    over.plane(0).row(0)[i] = o[i];
    base.plane(0).row(0)[i] = b[i];
  }
  REQUIRE_OK(alpha_blend(over, base));
  const std::uint8_t* r = base.plane(0).row(0);
  for (int i = 0; i < 4; ++i) CHECK(r[i] == b[i]);        // alpha 0: base unchanged
  for (int i = 4; i < 8; ++i) CHECK(r[i] == o[i]);        // alpha 255: overlay wins
  CHECK(r[8] == div255_ref(200 * 128 + 10 * 127));        // mid alpha
  CHECK(r[11] == 128 + div255_ref(40 * 127));
  CHECK(r[15] == 1 + div255_ref(255 * 254));
}

TEST(blend_matches_formula_for_all_inputs) {
  Frame over = make_frame(PixelFormat::bgra, 256, 1);
  Frame base = make_frame(PixelFormat::bgra, 256, 1);
  test::Rng rng(7);
  for (int v = 0; v < 256; ++v) {
    std::uint8_t* o = over.plane(0).row(0) + v * 4;
    std::uint8_t* b = base.plane(0).row(0) + v * 4;
    o[0] = rng.byte();
    o[1] = rng.byte();
    o[2] = static_cast<std::uint8_t>(v);
    o[3] = static_cast<std::uint8_t>(v);
    for (int c = 0; c < 4; ++c) b[c] = rng.byte();
  }
  Frame orig;
  REQUIRE_OK(base.clone(orig));
  REQUIRE_OK(alpha_blend(over, base));
  for (int v = 0; v < 256; ++v) {
    const std::uint8_t* o = over.plane(0).row(0) + v * 4;
    const std::uint8_t* b = orig.plane(0).row(0) + v * 4;
    const std::uint8_t* r = base.plane(0).row(0) + v * 4;
    const int a = o[3];
    for (int c = 0; c < 3; ++c) CHECK(r[c] == div255_ref(o[c] * a + b[c] * (255 - a)));
    CHECK(r[3] == a + div255_ref(b[3] * (255 - a)));
  }
}

TEST(blend_row_ranges_equal_whole_frame) {
  const Frame over = test::random_frame(PixelFormat::rgba, 37, 9, 3);
  Frame whole = test::random_frame(PixelFormat::rgba, 37, 9, 4);
  Frame parts;
  REQUIRE_OK(whole.clone(parts));
  REQUIRE_OK(alpha_blend(over, whole));
  REQUIRE_OK(alpha_blend(over, parts, RowRange{0, 4}));
  REQUIRE_OK(alpha_blend(over, parts, RowRange{4, 9}));
  CHECK(test::first_difference(whole, parts).empty());
}

TEST(blend_rejects_bad_arguments) {
  Frame rgba = make_frame(PixelFormat::rgba, 4, 4);
  Frame bgra = make_frame(PixelFormat::bgra, 4, 4);
  Frame small = make_frame(PixelFormat::rgba, 3, 4);
  Frame rgb = make_frame(PixelFormat::rgb24, 4, 4);
  Frame empty;
  CHECK(alpha_blend(rgba, bgra) == Status::invalid_argument);
  CHECK(alpha_blend(small, rgba) == Status::invalid_argument);
  CHECK(alpha_blend(empty, rgba) == Status::invalid_argument);
  CHECK(alpha_blend(rgb, rgb) == Status::unsupported_format);
  CHECK(alpha_blend(rgba, rgba, RowRange{0, 5}) == Status::invalid_argument);
}

TEST(blend_all_isas_match) {
  for (const PixelFormat fmt : kAlphaFormats) {
    for (const auto& [w, h] : test::test_sizes()) {
      const Frame over = test::random_frame(fmt, w, h, 5);
      const Frame base = test::random_frame(fmt, w, h, 6);
      CHECK_ALL_ISAS_MATCH(
          [&] {
            Frame f;
            REQUIRE_OK(base.clone(f));
            return f;
          },
          [&](Frame& out) { return alpha_blend(over, out); });
    }
  }
}
