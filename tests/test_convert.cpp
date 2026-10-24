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
int upsampled(const Frame& f, int c, int x, int y, ChromaMode mode) {
  const int cw = (f.width() + 1) / 2, ch = (f.height() + 1) / 2;
  const int cx = x / 2, cy = y / 2;
  if (mode == ChromaMode::nearest) return chroma_at(f, c, cx, cy);
  const int cxn = std::min(cx + 1, cw - 1);
  const int cyn = std::min(std::max(cy + (y % 2 ? 1 : -1), 0), ch - 1);
  const auto vert = [&](int i) { return 3 * chroma_at(f, c, i, cy) + chroma_at(f, c, i, cyn); };
  const int h = (x % 2) ? vert(cx) + vert(cxn) : 2 * vert(cx);
  return (h + 4) >> 3;
}

Rgb ref_rgb(const Frame& f, int x, int y, const ConvertOptions& o) {
  const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(o.matrix, o.range);
  const int yy = f.plane(0).row(y)[x] - c.y_offset;
  const int u = upsampled(f, 0, x, y, o.chroma) - 128;
  const int v = upsampled(f, 1, x, y, o.chroma) - 128;
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
        for (const auto& [w, h] : test::test_sizes()) {
          const Frame in = test::random_frame(src, w, h, ++seed);
          const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, mode};
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
        for (const auto& [w, h] : test::test_sizes()) {
          const Frame in = test::random_frame(src, w, h, 77 + w * 31 + h);
          const ConvertOptions o{ColorMatrix::bt709, ColorRange::limited, mode};
          CHECK_ALL_ISAS_MATCH([&] { return test::random_frame(dst, w, h, 3); },
                               [&](Frame& out) { return convert(in, out, o); });
        }
      }
    }
  }
}

TEST(convert_supported_and_errors) {
  for (int s = 0; s < kPixelFormatCount; ++s) {
    for (int d = 0; d < kPixelFormatCount; ++d) {
      const auto sf = static_cast<PixelFormat>(s), df = static_cast<PixelFormat>(d);
      const bool want = (sf == PixelFormat::yuv420p || sf == PixelFormat::nv12) &&
                        (df == PixelFormat::rgb24 || df == PixelFormat::rgba ||
                         df == PixelFormat::bgra);
      CHECK_EQ(convert_supported(sf, df), want);
      Frame in, out;
      REQUIRE_OK(Frame::allocate(sf, 4, 4, in));
      REQUIRE_OK(Frame::allocate(df, 4, 4, out));
      CHECK_EQ(convert(in, out), want ? Status::ok : Status::not_implemented);
    }
  }
  Frame in, out;
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 4, 4, in));
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 6, 4, out));
  CHECK_EQ(convert(in, out), Status::invalid_argument);
}
