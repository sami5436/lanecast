// scale suite. Owned by the scale unit.
// Filter table invariants and known values, Scaler behavior (identity,
// constants, ramps, extreme ratios, every format, RowRange slices) and
// CHECK_ALL_ISAS_MATCH over test::test_sizes() for every format and filter.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <tuple>

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

namespace {

const PixelFormat kFormats[] = {PixelFormat::gray8, PixelFormat::yuv420p, PixelFormat::nv12,
                                PixelFormat::rgb24, PixelFormat::rgba,    PixelFormat::bgra};

Frame make_frame(PixelFormat fmt, int w, int h) {
  Frame f;
  REQUIRE_OK(Frame::allocate(fmt, w, h, f));
  return f;
}

// Value of plane p, channel c in constant frames: distinct per plane and channel.
std::uint8_t const_value(int p, int c) { return static_cast<std::uint8_t>(17 + 61 * p + 37 * c); }

Frame constant_frame(PixelFormat fmt, int w, int h) {
  Frame f = test::random_frame(fmt, w, h, 99);
  for (int p = 0; p < f.plane_count(); ++p) {
    const Plane pl = f.plane(p);
    for (int y = 0; y < pl.height; ++y) {
      for (int x = 0; x < pl.width; ++x) {
        for (int c = 0; c < pl.bytes_per_pixel; ++c) {
          pl.row(y)[x * pl.bytes_per_pixel + c] = const_value(p, c);
        }
      }
    }
  }
  return f;
}

// Returns "" when every visible sample of dst equals its plane/channel constant.
std::string check_constant(const Frame& dst) {
  for (int p = 0; p < dst.plane_count(); ++p) {
    const Plane pl = dst.plane(p);
    for (int y = 0; y < pl.height; ++y) {
      for (int i = 0; i < pl.row_bytes(); ++i) {
        const int c = i % pl.bytes_per_pixel;
        if (pl.row(y)[i] != const_value(p, c)) {
          return "plane " + std::to_string(p) + " row " + std::to_string(y) + " byte " +
                 std::to_string(i) + " got " + std::to_string(pl.row(y)[i]);
        }
      }
    }
  }
  return {};
}

}  // namespace

TEST(identity_is_exact) {
  for (const PixelFormat fmt : kFormats) {
    for (const ScaleFilter f : kFilters) {
      for (const auto& [w, h] : test::test_sizes()) {
        const Frame src = test::random_frame(fmt, w, h, 7);
        Frame dst = make_frame(fmt, w, h);
        REQUIRE_OK(scale(src, dst, f));
        const std::string d = test::first_difference(src, dst);
        CHECK_MSG(d.empty(), std::string(pixfmt_name(fmt)) + " " + std::to_string(w) + "x" +
                                 std::to_string(h) + ": " + d);
      }
    }
  }
}

TEST(constant_image_stays_constant) {
  const std::pair<int, int> targets[] = {{1, 1}, {2, 3}, {5, 4}, {33, 17}, {64, 9}, {100, 61}};
  for (const PixelFormat fmt : kFormats) {
    for (const ScaleFilter f : kFilters) {
      for (const auto& [sw, sh] : {std::pair{1, 1}, std::pair{7, 5}, std::pair{64, 48}}) {
        const Frame src = constant_frame(fmt, sw, sh);
        for (const auto& [dw, dh] : targets) {
          Frame dst = test::random_frame(fmt, dw, dh, 3);
          REQUIRE_OK(scale(src, dst, f));
          const std::string d = check_constant(dst);
          CHECK_MSG(d.empty(), std::string(pixfmt_name(fmt)) + " " + std::to_string(sw) + "x" +
                                   std::to_string(sh) + "->" + std::to_string(dw) + "x" +
                                   std::to_string(dh) + ": " + d);
        }
      }
    }
  }
}

TEST(extreme_ratios) {
  for (const ScaleFilter f : kFilters) {
    for (const auto& [sw, sh, dw, dh] :
         {std::tuple{4096, 2, 3, 1}, std::tuple{3, 1, 4096, 2}, std::tuple{2, 4096, 1, 3},
          std::tuple{1, 3, 2, 4096}, std::tuple{4096, 4096, 3, 3}}) {
      const Frame src = constant_frame(PixelFormat::gray8, sw, sh);
      Frame dst = make_frame(PixelFormat::gray8, dw, dh);
      REQUIRE_OK(scale(src, dst, f));
      CHECK_MSG(check_constant(dst).empty(), "constant " + std::to_string(sw) + "x" +
                                                 std::to_string(sh) + "->" + std::to_string(dw) +
                                                 "x" + std::to_string(dh));
    }
  }
}

TEST(known_values) {
  // 2x down averages 2x2 blocks.
  Frame src = make_frame(PixelFormat::gray8, 4, 2);
  const std::uint8_t px[2][4] = {{10, 20, 30, 40}, {50, 60, 70, 80}};
  for (int y = 0; y < 2; ++y) std::memcpy(src.plane(0).row(y), px[y], 4);
  for (const ScaleFilter f : kFilters) {
    Frame dst = make_frame(PixelFormat::gray8, 2, 1);
    REQUIRE_OK(scale(src, dst, f));
    CHECK_EQ(static_cast<int>(dst.plane(0).row(0)[0]), 35);
    CHECK_EQ(static_cast<int>(dst.plane(0).row(0)[1]), 55);
  }
  // Bilinear 2x up of [0, 255]: edges hold, interior at quarter weights.
  Frame two = make_frame(PixelFormat::gray8, 2, 1);
  two.plane(0).row(0)[0] = 0;
  two.plane(0).row(0)[1] = 255;
  Frame four = make_frame(PixelFormat::gray8, 4, 1);
  REQUIRE_OK(scale(two, four, ScaleFilter::bilinear));
  const int expect[4] = {0, 64, 191, 255};
  for (int i = 0; i < 4; ++i) CHECK_EQ(static_cast<int>(four.plane(0).row(0)[i]), expect[i]);
}

TEST(ramps_stay_within_bounds) {
  // Horizontal and vertical ramps; the output must match the ideal sample
  // position of the same linear function to within rounding.
  for (const bool horizontal : {true, false}) {
    for (const ScaleFilter f : kFilters) {
      for (const auto& [sn, dn] : {std::pair{256, 100}, std::pair{256, 128}, std::pair{97, 31},
                                   std::pair{100, 256}, std::pair{31, 97}, std::pair{512, 3}}) {
        if (f == ScaleFilter::area && dn > sn) continue;  // box upscale is not smooth
        const int sw = horizontal ? sn : 8, sh = horizontal ? 8 : sn;
        const int dw = horizontal ? dn : 8, dh = horizontal ? 8 : dn;
        Frame src = make_frame(PixelFormat::gray8, sw, sh);
        for (int y = 0; y < sh; ++y) {
          for (int x = 0; x < sw; ++x) {
            const int i = horizontal ? x : y;
            src.plane(0).row(y)[x] =
                static_cast<std::uint8_t>(std::lround(255.0 * i / (sn - 1)));
          }
        }
        Frame dst = make_frame(PixelFormat::gray8, dw, dh);
        REQUIRE_OK(scale(src, dst, f));
        double worst = 0;
        for (int y = 0; y < dh; ++y) {
          for (int x = 0; x < dw; ++x) {
            const int o = horizontal ? x : y;
            const double pos = std::clamp((o + 0.5) * sn / dn - 0.5, 0.0, sn - 1.0);
            const double ideal = 255.0 * pos / (sn - 1);
            worst = std::max(worst, std::abs(ideal - dst.plane(0).row(y)[x]));
          }
        }
        CHECK_MSG(worst <= 2.0, std::string(horizontal ? "h " : "v ") + std::to_string(sn) +
                                    "->" + std::to_string(dn) + " worst " + std::to_string(worst));
      }
    }
  }
}

TEST(row_ranges_equal_whole_frame) {
  for (const PixelFormat fmt : {PixelFormat::yuv420p, PixelFormat::nv12, PixelFormat::rgba}) {
    for (const ScaleFilter f : kFilters) {
      for (const auto& [dw, dh] : {std::pair{40, 21}, std::pair{101, 60}}) {
        const Frame src = test::random_frame(fmt, 73, 47, 11);
        std::unique_ptr<Scaler> sc;
        REQUIRE_OK(Scaler::create(fmt, 73, 47, dw, dh, f, sc));
        Frame whole = make_frame(fmt, dw, dh);
        REQUIRE_OK(sc->run(src, whole));
        Frame parts = make_frame(fmt, dw, dh);
        const int cuts[] = {0, 8, 14, dh};
        for (int i = 0; i < 3; ++i) REQUIRE_OK(sc->run(src, parts, RowRange{cuts[i], cuts[i + 1]}));
        CHECK_MSG(test::first_difference(whole, parts).empty(),
                  std::string(pixfmt_name(fmt)) + " slices differ");
      }
    }
  }
}

TEST(rejects_bad_arguments) {
  std::unique_ptr<Scaler> sc;
  CHECK(Scaler::create(PixelFormat::rgb24, 0, 4, 4, 4, ScaleFilter::area, sc) ==
        Status::invalid_argument);
  CHECK(Scaler::create(PixelFormat::rgb24, 4, 4, kMaxDimension + 1, 4, ScaleFilter::area, sc) ==
        Status::invalid_argument);
  REQUIRE_OK(Scaler::create(PixelFormat::yuv420p, 8, 8, 4, 4, ScaleFilter::bilinear, sc));
  const Frame src = make_frame(PixelFormat::yuv420p, 8, 8);
  Frame wrong_size = make_frame(PixelFormat::yuv420p, 5, 4);
  Frame wrong_fmt = make_frame(PixelFormat::nv12, 4, 4);
  Frame ok = make_frame(PixelFormat::yuv420p, 4, 4);
  Frame empty;
  CHECK(sc->run(src, wrong_size) == Status::invalid_argument);
  CHECK(sc->run(src, wrong_fmt) == Status::invalid_argument);
  CHECK(sc->run(src, empty) == Status::invalid_argument);
  CHECK(sc->run(src, ok, RowRange{1, 4}) == Status::invalid_argument);  // odd start in 4:2:0
  CHECK(sc->run(src, ok, RowRange{0, 9}) == Status::invalid_argument);
  CHECK(sc->run(src, ok) == Status::ok);
}

TEST(all_isas_match) {
  for (const PixelFormat fmt : kFormats) {
    for (const ScaleFilter f : kFilters) {
      for (const auto& [w, h] : test::test_sizes()) {
        const Frame src = test::random_frame(fmt, w, h, 21 + static_cast<std::uint64_t>(w));
        const std::pair<int, int> targets[] = {
            {w, h}, {w * 3 / 2 + 1, h / 2 + 1}, {w / 2 + 1, h * 2}, {1, 1}, {w / 3 + 1, h + 3}};
        for (const auto& [dw, dh] : targets) {
          CHECK_ALL_ISAS_MATCH([&] { return make_frame(fmt, dw, dh); },
                               [&](Frame& out) { return scale(src, out, f); });
        }
      }
    }
  }
}
