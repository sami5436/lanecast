// filter suite. Owned by the filter unit.
#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>

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

TEST(levels_lut_known_values) {
  std::array<std::uint8_t, 256> lut;
  REQUIRE_OK(build_levels_lut(0, 256, lut));
  for (int v = 0; v < 256; ++v) CHECK(lut[static_cast<std::size_t>(v)] == v);
  REQUIRE_OK(build_levels_lut(10, 256, lut));
  CHECK(lut[0] == 10 && lut[100] == 110 && lut[250] == 255);
  REQUIRE_OK(build_levels_lut(-10, 256, lut));
  CHECK(lut[0] == 0 && lut[5] == 0 && lut[100] == 90 && lut[255] == 245);
  REQUIRE_OK(build_levels_lut(0, 0, lut));  // flat grey
  for (int v = 0; v < 256; ++v) CHECK(lut[static_cast<std::size_t>(v)] == 128);
  REQUIRE_OK(build_levels_lut(0, 512, lut));  // contrast x2 about 128
  CHECK(lut[128] == 128 && lut[138] == 148 && lut[100] == 72 && lut[0] == 0 && lut[255] == 255);
  REQUIRE_OK(build_levels_lut(0, 128, lut));  // contrast x0.5
  CHECK(lut[0] == 64 && lut[255] == 192 && lut[128] == 128);
  CHECK(build_levels_lut(256, 256, lut) == Status::invalid_argument);
  CHECK(build_levels_lut(0, 1025, lut) == Status::invalid_argument);
  CHECK(build_levels_lut(0, -1, lut) == Status::invalid_argument);
}

TEST(brightness_contrast_per_format) {
  std::array<std::uint8_t, 256> lut;
  REQUIRE_OK(build_levels_lut(20, 300, lut));
  for (const PixelFormat fmt : {PixelFormat::gray8, PixelFormat::yuv420p, PixelFormat::nv12,
                                PixelFormat::rgb24, PixelFormat::rgba, PixelFormat::bgra}) {
    const Frame src = test::random_frame(fmt, 19, 10, 9);
    Frame dst = test::random_frame(fmt, 19, 10, 10);
    REQUIRE_OK(brightness_contrast(src, dst, 20, 300));
    for (int p = 0; p < src.plane_count(); ++p) {
      const bool mapped = !pixfmt_info(fmt).is_yuv || p == 0;
      const Plane sp = src.plane(p);
      const Plane dp = dst.plane(p);
      for (int y = 0; y < sp.height; ++y) {
        for (int i = 0; i < sp.row_bytes(); ++i) {
          const bool alpha = pixfmt_info(fmt).has_alpha && i % 4 == 3;
          const int s = sp.row(y)[i];
          CHECK(dp.row(y)[i] == (mapped && !alpha ? lut[static_cast<std::size_t>(s)] : s));
        }
      }
    }
  }
}

TEST(brightness_contrast_in_place_and_slices) {
  for (const PixelFormat fmt : {PixelFormat::yuv420p, PixelFormat::rgba}) {
    const Frame src = test::random_frame(fmt, 21, 12, 1);
    Frame whole = make_frame(fmt, 21, 12);
    REQUIRE_OK(brightness_contrast(src, whole, -30, 400));
    Frame in_place;
    REQUIRE_OK(src.clone(in_place));
    REQUIRE_OK(brightness_contrast(in_place, in_place, -30, 400));
    CHECK(test::first_difference(whole, in_place).empty());
    Frame parts = make_frame(fmt, 21, 12);
    REQUIRE_OK(brightness_contrast(src, parts, -30, 400, RowRange{0, 6}));
    REQUIRE_OK(brightness_contrast(src, parts, -30, 400, RowRange{6, 12}));
    CHECK(test::first_difference(whole, parts).empty());
  }
}

TEST(brightness_contrast_rejects_bad_arguments) {
  const Frame src = make_frame(PixelFormat::yuv420p, 8, 8);
  Frame dst = make_frame(PixelFormat::yuv420p, 8, 8);
  Frame other = make_frame(PixelFormat::nv12, 8, 8);
  CHECK(brightness_contrast(src, other, 0, 256) == Status::invalid_argument);
  CHECK(brightness_contrast(src, dst, 300, 256) == Status::invalid_argument);
  CHECK(brightness_contrast(src, dst, 0, 256, RowRange{1, 8}) == Status::invalid_argument);
}

TEST(brightness_contrast_all_isas_match) {
  for (const PixelFormat fmt : {PixelFormat::gray8, PixelFormat::yuv420p, PixelFormat::nv12,
                                PixelFormat::rgb24, PixelFormat::rgba, PixelFormat::bgra}) {
    for (const auto& [w, h] : test::test_sizes()) {
      const Frame src = test::random_frame(fmt, w, h, 12);
      CHECK_ALL_ISAS_MATCH([&] { return make_frame(fmt, w, h); },
                           [&](Frame& out) { return brightness_contrast(src, out, 17, 333); });
    }
  }
}

namespace {

constexpr PixelFormat kAllFormats[] = {PixelFormat::gray8, PixelFormat::yuv420p, PixelFormat::nv12,
                                       PixelFormat::rgb24, PixelFormat::rgba,    PixelFormat::bgra};

// Straightforward reference of the documented formula, per plane channel.
Frame blur_reference(const Frame& src, int radius) {
  static const int b1[] = {1, 2, 1};
  static const int b2[] = {1, 4, 6, 4, 1};
  const int* bk = radius == 1 ? b1 : b2;
  Frame out = make_frame(src.format(), src.width(), src.height());
  for (int p = 0; p < src.plane_count(); ++p) {
    const Plane sp = src.plane(p);
    const Plane dp = out.plane(p);
    const int ch = sp.bytes_per_pixel;
    for (int y = 0; y < sp.height; ++y) {
      for (int x = 0; x < sp.width; ++x) {
        for (int c = 0; c < ch; ++c) {
          int v = 0;
          for (int ky = 0; ky <= 2 * radius; ++ky) {
            const int sy = std::clamp(y + ky - radius, 0, sp.height - 1);
            int h = 0;
            for (int kx = 0; kx <= 2 * radius; ++kx) {
              const int sx = std::clamp(x + kx - radius, 0, sp.width - 1);
              h += bk[kx] * sp.row(sy)[sx * ch + c];
            }
            v += bk[ky] * h;
          }
          const int shift = radius == 1 ? 4 : 8;
          dp.row(y)[x * ch + c] = static_cast<std::uint8_t>((v + (1 << (shift - 1))) >> shift);
        }
      }
    }
  }
  return out;
}

}  // namespace

TEST(blur_constant_plane_is_identity) {
  for (const int radius : {1, 2}) {
    for (const PixelFormat fmt : kAllFormats) {
      Frame src = make_frame(fmt, 13, 9);
      for (int p = 0; p < src.plane_count(); ++p) {
        const Plane pl = src.plane(p);
        for (int y = 0; y < pl.height; ++y) {
          std::memset(pl.row(y), 77 + p, static_cast<std::size_t>(pl.row_bytes()));
        }
      }
      Frame dst = make_frame(fmt, 13, 9);
      REQUIRE_OK(blur(src, dst, radius));
      CHECK(test::first_difference(src, dst).empty());
    }
  }
}

TEST(blur_known_values) {
  // Impulse of 255 at the center of a 5x5 plane: output is the kernel.
  Frame src = make_frame(PixelFormat::gray8, 5, 5);
  src.plane(0).row(2)[2] = 255;
  Frame dst = make_frame(PixelFormat::gray8, 5, 5);
  REQUIRE_OK(blur(src, dst, 1));
  CHECK(dst.plane(0).row(2)[2] == 64);  // 255 * 4 / 16 = 63.75
  CHECK(dst.plane(0).row(2)[1] == 32);  // 255 * 2 / 16 = 31.9
  CHECK(dst.plane(0).row(1)[1] == 16);  // 255 / 16 = 15.9
  CHECK(dst.plane(0).row(0)[0] == 0);
  REQUIRE_OK(blur(src, dst, 2));
  CHECK(dst.plane(0).row(2)[2] == 36);  // 255 * 36 / 256 = 35.9
  CHECK(dst.plane(0).row(2)[1] == 24);  // 255 * 24 / 256 = 23.9
  CHECK(dst.plane(0).row(0)[0] == 1);   // 255 / 256 = 1.0
}

TEST(blur_replicates_edges) {
  // Width 1: horizontal taps all read the one pixel. Vertical column
  // 10, 200, 50 with clamped rows.
  Frame src = make_frame(PixelFormat::gray8, 1, 3);
  src.plane(0).row(0)[0] = 10;
  src.plane(0).row(1)[0] = 200;
  src.plane(0).row(2)[0] = 50;
  Frame dst = make_frame(PixelFormat::gray8, 1, 3);
  REQUIRE_OK(blur(src, dst, 1));
  // h = 4 * value; v = sum {1 2 1} of h; out = (v + 8) >> 4.
  CHECK(dst.plane(0).row(0)[0] == ((4 * (10 + 2 * 10 + 200)) + 8) / 16);
  CHECK(dst.plane(0).row(1)[0] == ((4 * (10 + 2 * 200 + 50)) + 8) / 16);
  CHECK(dst.plane(0).row(2)[0] == ((4 * (200 + 2 * 50 + 50)) + 8) / 16);
}

TEST(blur_matches_reference) {
  for (const int radius : {1, 2}) {
    for (const PixelFormat fmt : kAllFormats) {
      for (const auto& [w, h] :
           {std::pair{1, 1}, std::pair{2, 7}, std::pair{9, 2}, std::pair{33, 17}}) {
        const Frame src = test::random_frame(fmt, w, h, 40 + static_cast<std::uint64_t>(w));
        Frame dst = make_frame(fmt, w, h);
        REQUIRE_OK(blur(src, dst, radius));
        const Frame ref = blur_reference(src, radius);
        CHECK_MSG(test::first_difference(ref, dst).empty(),
                  std::string(pixfmt_name(fmt)) + " r" + std::to_string(radius) + " " +
                      std::to_string(w) + "x" + std::to_string(h) + ": " +
                      test::first_difference(ref, dst));
      }
    }
  }
}

TEST(blur_row_ranges_equal_whole_frame) {
  for (const int radius : {1, 2}) {
    for (const PixelFormat fmt : {PixelFormat::yuv420p, PixelFormat::nv12, PixelFormat::rgba}) {
      const Frame src = test::random_frame(fmt, 41, 26, 8);
      Frame whole = make_frame(fmt, 41, 26);
      REQUIRE_OK(blur(src, whole, radius));
      Frame parts = make_frame(fmt, 41, 26);
      const int cuts[] = {0, 2, 10, 26};
      for (int i = 0; i < 3; ++i) {
        REQUIRE_OK(blur(src, parts, radius, RowRange{cuts[i], cuts[i + 1]}));
      }
      CHECK_MSG(test::first_difference(whole, parts).empty(),
                std::string(pixfmt_name(fmt)) + " slices differ");
    }
  }
}

TEST(blur_rejects_bad_arguments) {
  Frame a = make_frame(PixelFormat::yuv420p, 8, 8);
  Frame b = make_frame(PixelFormat::yuv420p, 8, 8);
  Frame c = make_frame(PixelFormat::yuv420p, 6, 8);
  Frame empty;
  CHECK(blur(a, b, 0) == Status::invalid_argument);
  CHECK(blur(a, b, 3) == Status::invalid_argument);
  CHECK(blur(a, a, 1) == Status::invalid_argument);
  CHECK(blur(a, c, 1) == Status::invalid_argument);
  CHECK(blur(empty, b, 1) == Status::invalid_argument);
  CHECK(blur(a, b, 1, RowRange{1, 8}) == Status::invalid_argument);
}

TEST(blur_all_isas_match) {
  for (const int radius : {1, 2}) {
    for (const PixelFormat fmt : kAllFormats) {
      for (const auto& [w, h] : test::test_sizes()) {
        const Frame src = test::random_frame(fmt, w, h, 60);
        CHECK_ALL_ISAS_MATCH([&] { return make_frame(fmt, w, h); },
                             [&](Frame& out) { return blur(src, out, radius); });
      }
    }
  }
}
