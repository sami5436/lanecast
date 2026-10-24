// convert suite. Owned by the convert unit.
// Required: known values (black, white, gray, primaries) per matrix and
// range, every supported pair, RowRange slices equal whole-frame output, and
// CHECK_ALL_ISAS_MATCH over test::test_sizes() for every kernel.
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "lanecast/convert.hpp"
#include "test_support.hpp"

using namespace lanecast;

TEST(coefficients_match_spec) {
  struct M {
    ColorMatrix m;
    double kr, kb;
  };
  for (const M mat : {M{ColorMatrix::bt601, 0.299, 0.114}, M{ColorMatrix::bt709, 0.2126, 0.0722}}) {
    for (const ColorRange r : {ColorRange::limited, ColorRange::full}) {
      const bool lim = r == ColorRange::limited;
      const double kg = 1.0 - mat.kr - mat.kb;
      const double ys = lim ? 255.0 / 219.0 : 1.0, cs = lim ? 255.0 / 224.0 : 1.0;
      const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(mat.m, r);
      const auto q13 = [](double v) { return static_cast<int>(std::lround(v * 8192.0)); };
      CHECK_EQ(c.y_offset, lim ? 16 : 0);
      CHECK_EQ(c.cy, q13(ys));
      CHECK_EQ(c.crv, q13(2 * (1 - mat.kr) * cs));
      CHECK_EQ(c.cbu, q13(2 * (1 - mat.kb) * cs));
      CHECK_EQ(c.cgu, q13(2 * (1 - mat.kb) * mat.kb / kg * cs));
      CHECK_EQ(c.cgv, q13(2 * (1 - mat.kr) * mat.kr / kg * cs));

      const RgbToYuvCoeffs k = rgb_to_yuv_coeffs(mat.m, r);
      const double ys2 = lim ? 219.0 / 255.0 : 1.0, cs2 = lim ? 224.0 / 255.0 : 1.0;
      const auto near_q15 = [](int got, double v) {
        return std::abs(got - static_cast<int>(std::lround(v * 32768.0))) <= 1;
      };
      CHECK_EQ(k.y_offset, lim ? 16 : 0);
      CHECK(near_q15(k.yr, mat.kr * ys2));
      CHECK(near_q15(k.yg, kg * ys2));
      CHECK(near_q15(k.yb, mat.kb * ys2));
      CHECK(near_q15(k.ur, -mat.kr / (2 * (1 - mat.kb)) * cs2));
      CHECK(near_q15(k.ub, 0.5 * cs2));
      CHECK(near_q15(k.vb, -mat.kb / (2 * (1 - mat.kr)) * cs2));
      CHECK_EQ(k.yr + k.yg + k.yb, static_cast<int>(std::lround(32768.0 * ys2)));
      CHECK_EQ(k.ur + k.ug + k.ub, 0);
      CHECK_EQ(k.vr + k.vg + k.vb, 0);
      CHECK_EQ(k.ub, k.vr);
    }
  }
}

namespace {

struct Rgb {
  int r, g, b;
};

int clamp8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// Chroma sample c (0 = U, 1 = V) at chroma column cx, row cy.
int chroma_at(const Frame& f, int c, int cx, int cy) {
  if (f.format() == PixelFormat::nv12) return f.plane(1).row(cy)[cx * 2 + c];
  return f.plane(1 + c).row(cy)[cx];
}

// Straight-from-the-spec chroma upsampling for one pixel (convert.hpp).
int upsampled(const Frame& f, int c, int x, int y, ChromaMode mode, ChromaLocation loc) {
  const int cw = (f.width() + 1) / 2, ch = (f.height() + 1) / 2;
  const int cx = x / 2, cy = y / 2;
  if (mode == ChromaMode::nearest) return chroma_at(f, c, cx, cy);
  const int cxn = std::min(cx + 1, cw - 1);
  const int cyn = std::min(std::max(cy + (y % 2 ? 1 : -1), 0), ch - 1);
  const auto vert = [&](int i) { return 3 * chroma_at(f, c, i, cy) + chroma_at(f, c, i, cyn); };
  if (loc == ChromaLocation::center) {
    const int cxo = std::min(std::max(cx + (x % 2 ? 1 : -1), 0), cw - 1);
    return (3 * vert(cx) + vert(cxo) + 8) >> 4;
  }
  const int h = (x % 2) ? vert(cx) + vert(cxn) : 2 * vert(cx);
  return (h + 4) >> 3;
}

Rgb ref_rgb(const Frame& f, int x, int y, const ConvertOptions& o) {
  const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(o.matrix, o.range);
  const int yy = f.plane(0).row(y)[x] - c.y_offset;
  const int u = upsampled(f, 0, x, y, o.chroma, o.chroma_loc) - 128;
  const int v = upsampled(f, 1, x, y, o.chroma, o.chroma_loc) - 128;
  return {clamp8((c.cy * yy + c.crv * v + 4096) >> 13),
          clamp8((c.cy * yy - c.cgu * u - c.cgv * v + 4096) >> 13),
          clamp8((c.cy * yy + c.cbu * u + 4096) >> 13)};
}

Rgb pixel(const Frame& f, int x, int y) {
  const std::uint8_t* p = f.plane(0).row(y) + x * pixfmt_info(f.format()).bytes_per_pixel[0];
  return f.format() == PixelFormat::bgra ? Rgb{p[2], p[1], p[0]} : Rgb{p[0], p[1], p[2]};
}

constexpr PixelFormat kYuvSources[] = {PixelFormat::yuv420p, PixelFormat::nv12};
constexpr PixelFormat kRgbTargets[] = {PixelFormat::rgb24, PixelFormat::rgba, PixelFormat::bgra};
constexpr ColorMatrix kMatrices[] = {ColorMatrix::bt601, ColorMatrix::bt709};
constexpr ColorRange kRanges[] = {ColorRange::limited, ColorRange::full};
constexpr ChromaMode kChromaModes[] = {ChromaMode::nearest, ChromaMode::bilinear};
constexpr ChromaLocation kLocations[] = {ChromaLocation::left, ChromaLocation::center};

// Sets every chroma sample of a 4:2:0 frame (planar or interleaved).
void set_chroma(Frame& f, int u, int v) {
  for (int p = 1; p < f.plane_count(); ++p) {
    const Plane pl = f.plane(p);
    for (int r = 0; r < pl.height; ++r) {
      for (int i = 0; i < pl.row_bytes(); ++i) {
        const bool is_v = f.format() == PixelFormat::nv12 ? (i % 2) != 0 : p == 2;
        pl.row(r)[i] = static_cast<std::uint8_t>(is_v ? v : u);
      }
    }
  }
}

// A yuv frame with constant luma and constant chroma.
Frame flat_yuv(PixelFormat fmt, int w, int h, int y, int u, int v) {
  Frame f;
  REQUIRE_OK(Frame::allocate(fmt, w, h, f));
  const Plane py = f.plane(0);
  for (int r = 0; r < py.height; ++r) {
    for (int i = 0; i < py.row_bytes(); ++i) py.row(r)[i] = static_cast<std::uint8_t>(y);
  }
  set_chroma(f, u, v);
  return f;
}

// Double-precision RGB -> YUV, used to build primaries independently.
void rgb_to_yuv_double(ColorMatrix m, ColorRange r, const Rgb& in, int& y, int& u, int& v) {
  const double kr = m == ColorMatrix::bt601 ? 0.299 : 0.2126;
  const double kb = m == ColorMatrix::bt601 ? 0.114 : 0.0722;
  const double kg = 1.0 - kr - kb;
  const bool lim = r == ColorRange::limited;
  const double yf = kr * in.r + kg * in.g + kb * in.b;
  const double uf = (in.b - yf) / (2 * (1 - kb)), vf = (in.r - yf) / (2 * (1 - kr));
  y = static_cast<int>(std::lround(lim ? 16 + yf * 219.0 / 255.0 : yf));
  u = static_cast<int>(std::lround(128 + (lim ? uf * 224.0 / 255.0 : uf)));
  v = static_cast<int>(std::lround(128 + (lim ? vf * 224.0 / 255.0 : vf)));
  u = clamp8(u);  // full-range blue and red land on 256
  v = clamp8(v);
}

}  // namespace

TEST(yuv_to_rgb_primaries_and_grays) {
  const int levels[] = {0, 128, 255};
  for (const PixelFormat src : kYuvSources) {
    for (const PixelFormat dst : kRgbTargets) {
      for (const ColorMatrix m : kMatrices) {
        for (const ColorRange r : kRanges) {
          for (const int lr : levels) {
            for (const int lg : levels) {
              for (const int lb : levels) {
                const Rgb want{lr, lg, lb};
                int y, u, v;
                rgb_to_yuv_double(m, r, want, y, u, v);
                const Frame in = flat_yuv(src, 4, 4, y, u, v);
                Frame out;
                REQUIRE_OK(Frame::allocate(dst, 4, 4, out));
                REQUIRE_OK(convert(in, out, {m, r}));
                const Rgb got = pixel(out, 1, 2);
                // Rounded 8-bit yuv input limits accuracy to a few levels.
                const std::string at = std::string(pixfmt_name(src)) + " yuv " + std::to_string(y) +
                                       "," + std::to_string(u) + "," + std::to_string(v) + " got " +
                                       std::to_string(got.r) + "," + std::to_string(got.g) + "," +
                                       std::to_string(got.b);
                CHECK_MSG(std::abs(got.r - want.r) <= 3 && std::abs(got.g - want.g) <= 3 &&
                              std::abs(got.b - want.b) <= 3,
                          at);
                if (lr == lg && lg == lb) CHECK_MSG(got.r == got.g && got.g == got.b, at);
                if (dst != PixelFormat::rgb24) CHECK_EQ(int(out.plane(0).row(2)[1 * 4 + 3]), 255);
              }
            }
          }
        }
      }
    }
  }
}

TEST(yuv_to_rgb_exact_black_white_gray) {
  for (const PixelFormat src : kYuvSources) {
    for (const ColorMatrix m : kMatrices) {
      Frame out;
      REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 3, 3, out));
      const struct {
        int y;
        ColorRange range;
        int want;
      } cases[] = {{16, ColorRange::limited, 0},
                   {235, ColorRange::limited, 255},
                   {0, ColorRange::full, 0},
                   {128, ColorRange::full, 128},
                   {255, ColorRange::full, 255}};
      for (const auto& c : cases) {
        REQUIRE_OK(convert(flat_yuv(src, 3, 3, c.y, 128, 128), out, {m, c.range}));
        const Rgb p = pixel(out, 2, 2);
        CHECK(p.r == c.want && p.g == c.want && p.b == c.want);
      }
    }
  }
}

TEST(yuv_to_rgb_matches_reference_model) {
  std::uint64_t seed = 100;
  for (const PixelFormat src : kYuvSources) {
    for (const PixelFormat dst : kRgbTargets) {
      for (const ChromaMode mode : kChromaModes) {
        for (const ChromaLocation loc : kLocations) {
          for (const auto& [w, h] : test::test_sizes()) {
            const Frame in = test::random_frame(src, w, h, ++seed);
            const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, mode, loc};
            Frame out = test::random_frame(dst, w, h, ++seed);
            REQUIRE_OK(convert(in, out, o));
            int bad = 0;
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                const Rgb a = pixel(out, x, y), b = ref_rgb(in, x, y, o);
                bad += a.r != b.r || a.g != b.g || a.b != b.b;
              }
            }
            CHECK_MSG(bad == 0, std::string(pixfmt_name(src)) + " -> " +
                                    std::string(pixfmt_name(dst)) + " " + std::to_string(w) + "x" +
                                    std::to_string(h) + ": " + std::to_string(bad) + " bad pixels");
          }
        }
      }
    }
  }
}

TEST(bilinear_matches_nearest_on_flat_chroma_and_differs_otherwise) {
  for (const PixelFormat src : kYuvSources) {
    Frame in = test::random_frame(src, 17, 9, 5);
    Frame a, b;
    REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 17, 9, a));
    REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 17, 9, b));
    const ConvertOptions nearest{ColorMatrix::bt601, ColorRange::limited, ChromaMode::nearest};
    REQUIRE_OK(convert(in, a, nearest));
    REQUIRE_OK(convert(in, b));  // default is bilinear
    CHECK(!test::first_difference(a, b).empty());
    set_chroma(in, 61, 203);
    REQUIRE_OK(convert(in, a, nearest));
    REQUIRE_OK(convert(in, b));
    CHECK_MSG(test::first_difference(a, b).empty(), test::first_difference(a, b));
  }
}

TEST(yuv_to_rgb_row_ranges_equal_whole_frame) {
  for (const PixelFormat src : kYuvSources) {
    for (const ChromaMode mode : kChromaModes) {
      const Frame in = test::random_frame(src, 33, 11, 9);
      const ConvertOptions o{ColorMatrix::bt601, ColorRange::full, mode};
      Frame whole, parts;
      REQUIRE_OK(Frame::allocate(PixelFormat::bgra, 33, 11, whole));
      REQUIRE_OK(Frame::allocate(PixelFormat::bgra, 33, 11, parts));
      REQUIRE_OK(convert(in, whole, o));
      REQUIRE_OK(convert(in, parts, o, {0, 4}));
      REQUIRE_OK(convert(in, parts, o, {4, 10}));
      REQUIRE_OK(convert(in, parts, o, {10, 11}));
      CHECK_MSG(test::first_difference(whole, parts).empty(), test::first_difference(whole, parts));
      CHECK_EQ(convert(in, parts, o, {3, 5}), Status::invalid_argument);
    }
  }
}

TEST(yuv_to_rgb_all_isas_match) {
  for (const PixelFormat src : kYuvSources) {
    for (const PixelFormat dst : kRgbTargets) {
      for (const ChromaMode mode : kChromaModes) {
        for (const ChromaLocation loc : kLocations) {
          for (const auto& [w, h] : test::test_sizes()) {
            const Frame in = test::random_frame(src, w, h, 77 + w * 31 + h);
            const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, mode, loc};
            CHECK_ALL_ISAS_MATCH([&] { return test::random_frame(dst, w, h, 3); },
                                 [&](Frame& out) { return convert(in, out, o); });
          }
        }
      }
    }
  }
}

TEST(convert_supported_and_errors) {
  for (int s = 0; s < kPixelFormatCount; ++s) {
    for (int d = 0; d < kPixelFormatCount; ++d) {
      const auto sf = static_cast<PixelFormat>(s), df = static_cast<PixelFormat>(d);
      CHECK(convert_supported(sf, df));
      Frame in, out;
      REQUIRE_OK(Frame::allocate(sf, 4, 4, in));
      REQUIRE_OK(Frame::allocate(df, 4, 4, out));
      CHECK_EQ(convert(in, out), Status::ok);
    }
  }
  Frame in, out;
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 4, 4, in));
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 6, 4, out));
  CHECK_EQ(convert(in, out), Status::invalid_argument);
  REQUIRE_OK(Frame::allocate(PixelFormat::gray8, 4, 4, in));
  REQUIRE_OK(Frame::allocate(PixelFormat::nv12, 4, 4, out));
  CHECK_EQ(convert(in, out, {}, {1, 4}), Status::invalid_argument);  // dst needs even rows
}

namespace {

constexpr PixelFormat kRgbSources[] = {PixelFormat::rgb24, PixelFormat::rgba, PixelFormat::bgra};
constexpr PixelFormat kYuvTargets[] = {PixelFormat::yuv420p, PixelFormat::nv12};

void set_pixel(Frame& f, int x, int y, const Rgb& c) {
  const int bpp = pixfmt_info(f.format()).bytes_per_pixel[0];
  std::uint8_t* p = f.plane(0).row(y) + x * bpp;
  const bool bgr = f.format() == PixelFormat::bgra;
  p[bgr ? 2 : 0] = static_cast<std::uint8_t>(c.r);
  p[1] = static_cast<std::uint8_t>(c.g);
  p[bgr ? 0 : 2] = static_cast<std::uint8_t>(c.b);
  if (bpp == 4) p[3] = 255;
}

Frame flat_rgb(PixelFormat fmt, int w, int h, const Rgb& c) {
  Frame f;
  REQUIRE_OK(Frame::allocate(fmt, w, h, f));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) set_pixel(f, x, y, c);
  }
  return f;
}

// Y of pixel (x, y) and the chroma pair of a 4:2:0 frame sample (cx, cy).
int y_at(const Frame& f, int x, int y) { return f.plane(0).row(y)[x]; }

// Straight-from-the-spec rgb -> yuv (convert.hpp) for the luma of one pixel.
int ref_luma(const Rgb& p, const RgbToYuvCoeffs& k) {
  return clamp8((k.yr * p.r + k.yg * p.g + k.yb * p.b + (k.y_offset << 15) + 16384) >> 15);
}

// Chroma (c = 0 for U, 1 for V) of sample (cx, cy), straight from convert.hpp.
int ref_chroma(const Frame& rgb, int c, int cx, int cy, const RgbToYuvCoeffs& k,
               ChromaLocation loc) {
  const int w = rgb.width(), x0 = 2 * cx;
  int sr = 0, sg = 0, sb = 0;
  const auto add = [&](int x, int y, int weight) {
    const Rgb p = pixel(rgb, std::min(std::max(x, 0), w - 1), std::min(y, rgb.height() - 1));
    sr += weight * p.r;
    sg += weight * p.g;
    sb += weight * p.b;
  };
  for (int j = 0; j < 2; ++j) {
    if (loc == ChromaLocation::center) {
      add(x0, 2 * cy + j, 1);
      add(x0 + 1, 2 * cy + j, 1);
    } else {
      add(x0 - 1, 2 * cy + j, 1);
      add(x0, 2 * cy + j, 2);
      add(x0 + 1, 2 * cy + j, 1);
    }
  }
  const int shift = loc == ChromaLocation::center ? 17 : 18;
  const int v = c == 0 ? k.ur * sr + k.ug * sg + k.ub * sb : k.vr * sr + k.vg * sg + k.vb * sb;
  return clamp8((v + (128 << shift) + (1 << (shift - 1))) >> shift);
}

}  // namespace

TEST(rgb_to_yuv_exact_black_white_gray) {
  for (const PixelFormat src : kRgbSources) {
    for (const PixelFormat dst : kYuvTargets) {
      for (const ColorMatrix m : kMatrices) {
        for (const ColorRange r : kRanges) {
          const bool lim = r == ColorRange::limited;
          Frame out;
          REQUIRE_OK(Frame::allocate(dst, 5, 3, out));
          const struct {
            int level, want_y;
          } cases[] = {{0, lim ? 16 : 0}, {255, lim ? 235 : 255}};
          for (const auto& c : cases) {
            REQUIRE_OK(convert(flat_rgb(src, 5, 3, {c.level, c.level, c.level}), out, {m, r}));
            CHECK_EQ(y_at(out, 4, 2), c.want_y);
            CHECK_EQ(chroma_at(out, 0, 2, 1), 128);
            CHECK_EQ(chroma_at(out, 1, 2, 1), 128);
          }
          // Every gray maps to U = V = 128 exactly.
          Frame small;
          REQUIRE_OK(Frame::allocate(dst, 2, 2, small));
          for (int g = 0; g < 256; g += 5) {
            REQUIRE_OK(convert(flat_rgb(src, 2, 2, {g, g, g}), small, {m, r}));
            CHECK_EQ(chroma_at(small, 0, 0, 0), 128);
            CHECK_EQ(chroma_at(small, 1, 0, 0), 128);
          }
        }
      }
    }
  }
}

TEST(rgb_to_yuv_primaries_match_double_reference) {
  const int levels[] = {0, 255};
  for (const PixelFormat src : kRgbSources) {
    for (const PixelFormat dst : kYuvTargets) {
      for (const ColorMatrix m : kMatrices) {
        for (const ColorRange r : kRanges) {
          for (const int lr : levels) {
            for (const int lg : levels) {
              for (const int lb : levels) {
                int y, u, v;
                rgb_to_yuv_double(m, r, {lr, lg, lb}, y, u, v);
                Frame out;
                REQUIRE_OK(Frame::allocate(dst, 4, 4, out));
                REQUIRE_OK(convert(flat_rgb(src, 4, 4, {lr, lg, lb}), out, {m, r}));
                CHECK(std::abs(y_at(out, 1, 1) - y) <= 1);
                CHECK(std::abs(chroma_at(out, 0, 1, 1) - u) <= 1);
                CHECK(std::abs(chroma_at(out, 1, 1, 1) - v) <= 1);
              }
            }
          }
        }
      }
    }
  }
}

TEST(rgb_to_yuv_matches_reference_model) {
  std::uint64_t seed = 400;
  for (const PixelFormat src : kRgbSources) {
    for (const PixelFormat dst : kYuvTargets) {
      for (const ChromaLocation loc : kLocations) {
        for (const auto& [w, h] : test::test_sizes()) {
          const Frame in = test::random_frame(src, w, h, ++seed);
          const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, ChromaMode::bilinear, loc};
          const RgbToYuvCoeffs k = rgb_to_yuv_coeffs(o.matrix, o.range);
          Frame out = test::random_frame(dst, w, h, ++seed);
          REQUIRE_OK(convert(in, out, o));
          int bad = 0;
          for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) bad += y_at(out, x, y) != ref_luma(pixel(in, x, y), k);
          }
          for (int cy = 0; cy < (h + 1) / 2; ++cy) {
            for (int cx = 0; cx < (w + 1) / 2; ++cx) {
              bad += chroma_at(out, 0, cx, cy) != ref_chroma(in, 0, cx, cy, k, loc);
              bad += chroma_at(out, 1, cx, cy) != ref_chroma(in, 1, cx, cy, k, loc);
            }
          }
          CHECK_MSG(bad == 0, std::string(pixfmt_name(src)) + " -> " +
                                  std::string(pixfmt_name(dst)) + " " + std::to_string(w) + "x" +
                                  std::to_string(h) + ": " + std::to_string(bad) + " bad samples");
        }
      }
    }
  }
}

TEST(round_trip_error_bounds) {
  // 2x2-aligned random color blocks: chroma is exact per block, so the error
  // is quantization alone (nearest). A smooth ramp bounds bilinear.
  for (const ColorMatrix m : kMatrices) {
    for (const ColorRange r : kRanges) {
      const int bound = r == ColorRange::limited ? 4 : 2;
      test::Rng rng(11);
      Frame rgb;
      REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 32, 16, rgb));
      for (int by = 0; by < 8; ++by) {
        for (int bx = 0; bx < 16; ++bx) {
          const Rgb c{rng.byte(), rng.byte(), rng.byte()};
          for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) set_pixel(rgb, 2 * bx + i, 2 * by + j, c);
          }
        }
      }
      Frame ramp;
      REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 64, 32, ramp));
      for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 64; ++x) set_pixel(ramp, x, y, {60 + x * 2, 200 - y * 3, 40 + x + y});
      }
      for (const PixelFormat mid : kYuvTargets) {
        for (const ChromaMode mode : kChromaModes) {
          const Frame& img = mode == ChromaMode::nearest ? rgb : ramp;
          Frame yuv, back;
          REQUIRE_OK(Frame::allocate(mid, img.width(), img.height(), yuv));
          REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, img.width(), img.height(), back));
          // Nearest decoding pairs with the box encoder; bilinear with the left filter.
          const ChromaLocation loc =
              mode == ChromaMode::nearest ? ChromaLocation::center : ChromaLocation::left;
          REQUIRE_OK(convert(img, yuv, {m, r, mode, loc}));
          REQUIRE_OK(convert(yuv, back, {m, r, mode, loc}));
          int worst = 0;
          for (int y = 0; y < img.height(); ++y) {
            for (int x = 0; x < img.width(); ++x) {
              const Rgb a = pixel(img, x, y), b = pixel(back, x, y);
              worst = std::max({worst, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
            }
          }
          CHECK_MSG(worst <= (mode == ChromaMode::nearest ? bound : bound + 2),
                    std::string(mode == ChromaMode::nearest ? "nearest" : "bilinear") + " round trip error " + std::to_string(worst));
        }
      }
    }
  }
}

TEST(swizzle_gray_and_copy_paths) {
  const Frame rgb = test::random_frame(PixelFormat::rgb24, 19, 7, 21);
  const Frame rgba = test::random_frame(PixelFormat::rgba, 19, 7, 22);
  Frame bgra, back, rgb_back;
  REQUIRE_OK(Frame::allocate(PixelFormat::bgra, 19, 7, bgra));
  REQUIRE_OK(Frame::allocate(PixelFormat::rgba, 19, 7, back));
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 19, 7, rgb_back));
  REQUIRE_OK(convert(rgba, bgra));
  REQUIRE_OK(convert(bgra, back));
  CHECK_MSG(test::first_difference(rgba, back).empty(), test::first_difference(rgba, back));
  REQUIRE_OK(convert(rgba, rgb_back));
  REQUIRE_OK(convert(rgb, back));
  for (int y = 0; y < 7; ++y) {
    for (int x = 0; x < 19; ++x) {
      const Rgb a = pixel(rgba, x, y), b = pixel(rgb_back, x, y), c = pixel(back, x, y),
                d = pixel(rgb, x, y);
      CHECK(a.r == b.r && a.g == b.g && a.b == b.b);
      CHECK(c.r == d.r && c.g == d.g && c.b == d.b && back.plane(0).row(y)[x * 4 + 3] == 255);
    }
  }
  // gray8 <-> packed and yuv.
  const Frame gray = test::random_frame(PixelFormat::gray8, 19, 7, 23);
  for (const PixelFormat p : kRgbTargets) {
    Frame out, g2;
    REQUIRE_OK(Frame::allocate(p, 19, 7, out));
    REQUIRE_OK(Frame::allocate(PixelFormat::gray8, 19, 7, g2));
    REQUIRE_OK(convert(gray, out));
    for (int x = 0; x < 19; ++x) {
      const Rgb c = pixel(out, x, 3);
      CHECK(c.r == gray.plane(0).row(3)[x] && c.g == c.r && c.b == c.r);
    }
    REQUIRE_OK(convert(out, g2, {ColorMatrix::bt709, ColorRange::limited}));
    CHECK_MSG(test::first_difference(gray, g2).empty(), test::first_difference(gray, g2));
  }
  for (const PixelFormat p : kYuvTargets) {
    Frame out, g2;
    REQUIRE_OK(Frame::allocate(p, 19, 7, out));
    REQUIRE_OK(Frame::allocate(PixelFormat::gray8, 19, 7, g2));
    REQUIRE_OK(convert(gray, out));
    CHECK_EQ(chroma_at(out, 0, 9, 3), 128);
    CHECK_EQ(chroma_at(out, 1, 0, 0), 128);
    REQUIRE_OK(convert(out, g2));
    CHECK_MSG(test::first_difference(gray, g2).empty(), test::first_difference(gray, g2));
  }
  // yuv420p <-> nv12 round trip is exact, and same-format is a copy.
  const Frame i420 = test::random_frame(PixelFormat::yuv420p, 19, 7, 24);
  Frame nv, i420b, copy;
  REQUIRE_OK(Frame::allocate(PixelFormat::nv12, 19, 7, nv));
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 19, 7, i420b));
  REQUIRE_OK(convert(i420, nv));
  CHECK_EQ(int(nv.plane(1).row(2)[7 * 2]), int(i420.plane(1).row(2)[7]));
  CHECK_EQ(int(nv.plane(1).row(2)[7 * 2 + 1]), int(i420.plane(2).row(2)[7]));
  REQUIRE_OK(convert(nv, i420b));
  CHECK_MSG(test::first_difference(i420, i420b).empty(), test::first_difference(i420, i420b));
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 19, 7, copy));
  REQUIRE_OK(convert(i420, copy));
  CHECK_MSG(test::first_difference(i420, copy).empty(), test::first_difference(i420, copy));
}

TEST(every_pair_row_ranges_and_isas_match) {
  for (int s = 0; s < kPixelFormatCount; ++s) {
    for (int d = 0; d < kPixelFormatCount; ++d) {
      const auto sf = static_cast<PixelFormat>(s), df = static_cast<PixelFormat>(d);
      for (const auto& [w, h] : test::test_sizes()) {
        const Frame in = test::random_frame(sf, w, h, 900 + s * 7 + d + w);
        const ChromaLocation loc = (w + s) % 2 ? ChromaLocation::left : ChromaLocation::center;
        const ConvertOptions o{ColorMatrix::bt601, ColorRange::limited, ChromaMode::bilinear, loc};
        CHECK_ALL_ISAS_MATCH([&] { return test::random_frame(df, w, h, 3); },
                             [&](Frame& out) { return convert(in, out, o); });
        if (h < 4) continue;
        Frame whole = test::random_frame(df, w, h, 4), parts = test::random_frame(df, w, h, 5);
        REQUIRE_OK(convert(in, whole, o));
        REQUIRE_OK(convert(in, parts, o, {0, 2}));
        REQUIRE_OK(convert(in, parts, o, {2, h}));
        const std::string diff = test::first_difference(whole, parts);
        CHECK_MSG(diff.empty(), std::string(pixfmt_name(sf)) + " -> " +
                                    std::string(pixfmt_name(df)) + " " + std::to_string(w) + "x" +
                                    std::to_string(h) + ": " + diff);
      }
    }
  }
}

namespace {

// PSNR of rgb24 -> yuv420p -> rgb24 with one chroma mode and siting.
double round_trip_psnr(const Frame& img, ChromaMode mode, ChromaLocation loc) {
  Frame yuv, back;
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, img.width(), img.height(), yuv));
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, img.width(), img.height(), back));
  const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, mode, loc};
  REQUIRE_OK(convert(img, yuv, o));
  REQUIRE_OK(convert(yuv, back, o));
  double sse = 0;
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      const Rgb a = pixel(img, x, y), b = pixel(back, x, y);
      sse += (a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b);
    }
  }
  return 10.0 * std::log10(255.0 * 255.0 / std::max(sse / (3.0 * img.width() * img.height()), 1e-9));
}

}  // namespace

TEST(bilinear_round_trip_beats_nearest_on_sharp_edges) {
  // Edges on odd coordinates, so no chroma sample sits on a clean boundary,
  // plus 5-pixel color checkerboards.
  Frame edges, checks;
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 96, 64, edges));
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 96, 64, checks));
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 96; ++x) {
      Rgb c = x < 41 ? Rgb{220, 40, 30} : Rgb{20, 60, 230};
      if (y >= 29) c = {c.g, c.b, c.r};
      set_pixel(edges, x, y, c);
      set_pixel(checks, x, y, ((x / 5 + y / 5) & 1) ? Rgb{240, 200, 20} : Rgb{30, 40, 200});
    }
  }
  for (const ChromaLocation loc : kLocations) {
    for (const Frame* img : {&edges, &checks}) {
      const double nearest = round_trip_psnr(*img, ChromaMode::nearest, loc);
      const double bilinear = round_trip_psnr(*img, ChromaMode::bilinear, loc);
      CHECK_MSG(bilinear >= nearest, "bilinear " + std::to_string(bilinear) + " dB < nearest " +
                                         std::to_string(nearest) + " dB, loc " +
                                         std::to_string(static_cast<int>(loc)));
    }
  }
}

namespace {

int rmul(int a, int k) { return (a * k + 16384) >> 15; }

// Straight-from-the-spec fast precision pixel (convert.hpp, "Fast precision").
Rgb ref_rgb_fast(const Frame& f, int x, int y, const ConvertOptions& o) {
  const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(o.matrix, o.range);
  const int yp = f.plane(0).row(y)[x] << 7;
  const int up = chroma_at(f, 0, x / 2, y / 2) << 7, vp = chroma_at(f, 1, x / 2, y / 2) << 7;
  const int ky = rmul(c.y_offset << 7, c.cy), m = 128 << 7;
  const int ly = rmul(yp, c.cy);
  const int r5 = ly + rmul(vp, c.crv) + 16 - ky - rmul(m, c.crv);
  const int g5 =
      ly + rmul(up, -c.cgu) + rmul(vp, -c.cgv) + 16 - ky + rmul(m, c.cgu) + rmul(m, c.cgv);
  const int b5 = ly + rmul(up, c.cbu) + 16 - ky - rmul(m, c.cbu);
  return {clamp8(r5 >> 5), clamp8(g5 >> 5), clamp8(b5 >> 5)};
}

}  // namespace

TEST(fast_precision_matches_contract_and_stays_close_to_exact) {
  std::uint64_t seed = 400;
  for (const PixelFormat src : kYuvSources) {
    for (const PixelFormat dst : kRgbTargets) {
      for (const ColorMatrix m : kMatrices) {
        for (const ColorRange r : kRanges) {
          for (const auto& [w, h] : test::test_sizes()) {
            const Frame in = test::random_frame(src, w, h, ++seed);
            ConvertOptions o{m, r, ChromaMode::nearest};
            Frame fast = test::random_frame(dst, w, h, ++seed);
            Frame exact = test::random_frame(dst, w, h, ++seed);
            REQUIRE_OK(convert(in, exact, o));
            o.precision = Precision::fast;
            REQUIRE_OK(convert(in, fast, o));
            int bad = 0, far = 0;
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                const Rgb a = pixel(fast, x, y), b = ref_rgb_fast(in, x, y, o);
                const Rgb e = pixel(exact, x, y);
                bad += a.r != b.r || a.g != b.g || a.b != b.b;
                far += std::abs(a.r - e.r) > 1 || std::abs(a.g - e.g) > 1 || std::abs(a.b - e.b) > 1;
              }
            }
            const std::string at = std::string(pixfmt_name(src)) + " -> " +
                                   std::string(pixfmt_name(dst)) + " " + std::to_string(w) + "x" +
                                   std::to_string(h);
            CHECK_MSG(bad == 0, at + ": " + std::to_string(bad) + " pixels differ from the contract");
            CHECK_MSG(far == 0, at + ": " + std::to_string(far) + " pixels more than 1 from exact");
          }
        }
      }
    }
  }
}

TEST(fast_precision_all_isas_match) {
  for (const PixelFormat src : kYuvSources) {
    for (const PixelFormat dst : kRgbTargets) {
      for (const auto& [w, h] : test::test_sizes()) {
        const Frame in = test::random_frame(src, w, h, 91 + w * 17 + h);
        const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, ChromaMode::nearest,
                               ChromaLocation::left, Precision::fast};
        CHECK_ALL_ISAS_MATCH([&] { return test::random_frame(dst, w, h, 3); },
                             [&](Frame& out) { return convert(in, out, o); });
      }
    }
  }
}
