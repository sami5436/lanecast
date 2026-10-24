// fused suite: ScaleConvert equals scale() then convert() bit for bit, for
// sliced and threaded runs and under every supported ISA tier.
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kernels/convert/kernels.hpp"
#include "kernels/scale/kernels.hpp"
#include "lanecast/pipeline.hpp"
#include "test_support.hpp"

namespace lanecast {
namespace {

struct Size {
  int sw, sh, dw, dh;
};

// Down, up, mixed, odd, tiny, and heights that span several 32 row strips.
const Size kSizes[] = {{1, 1, 1, 1},     {1, 1, 5, 3},     {2, 2, 1, 1},     {7, 5, 3, 3},
                       {17, 9, 33, 21},  {33, 33, 17, 17}, {64, 40, 21, 13}, {65, 17, 129, 35},
                       {96, 70, 61, 45}, {47, 130, 31, 75}, {200, 90, 127, 66}};

std::unique_ptr<ScaleConvert> make(PixelFormat sf, const Size& z, PixelFormat df,
                                   ScaleFilter filter, const ConvertOptions& opts) {
  std::unique_ptr<ScaleConvert> sc;
  REQUIRE_OK(ScaleConvert::create(sf, z.sw, z.sh, df, z.dw, z.dh, filter, opts, sc));
  return sc;
}

// The contract: scale at the destination size, then convert.
Frame unfused(const Frame& src, const Size& z, PixelFormat df, ScaleFilter filter,
              const ConvertOptions& opts) {
  Frame mid;
  REQUIRE_OK(Frame::allocate(src.format(), z.dw, z.dh, mid));
  REQUIRE_OK(scale(src, mid, filter, {}, opts.chroma_loc));
  Frame out;
  REQUIRE_OK(Frame::allocate(df, z.dw, z.dh, out));
  REQUIRE_OK(convert(mid, out, opts));
  return out;
}

Frame alloc(PixelFormat f, int w, int h) {
  Frame out;
  REQUIRE_OK(Frame::allocate(f, w, h, out));
  test::poison_padding(out, 7);
  return out;
}

std::string describe(PixelFormat sf, PixelFormat df, const Size& z) {
  return std::string(pixfmt_name(sf)) + " -> " + std::string(pixfmt_name(df)) + " " +
         std::to_string(z.sw) + "x" + std::to_string(z.sh) + " -> " + std::to_string(z.dw) + "x" +
         std::to_string(z.dh);
}

// The Q7 vertical kernel against its formula, on every tier. Rows carry the
// padding that kernels may touch.
TEST(VerticalQ7KernelMatchesTheFormula) {
  constexpr std::size_t kPad = 64;
  test::Rng rng(7);
  for (const int taps : {1, 2, 3, 4, 5, 8}) {
    for (const int count : {1, 15, 16, 17, 31, 32, 33, 70, 128}) {
      const auto n = static_cast<std::size_t>(count);
      std::vector<std::vector<std::int16_t>> rows(static_cast<std::size_t>(taps),
                                                  std::vector<std::int16_t>(n + kPad));
      std::vector<const std::int16_t*> ptrs;
      for (auto& r : rows) {
        for (std::size_t i = 0; i < n; ++i) {
          // Q7 samples, with the extremes 0 and 255 * 128 mixed in.
          const int pick = rng.range(0, 7);
          r[i] = static_cast<std::int16_t>(pick == 0   ? 0
                                           : pick == 1 ? 255 * 128
                                                       : rng.range(0, 255 * 128));
        }
        ptrs.push_back(r.data());
      }
      // Non-negative Q14 weights that sum to 16384.
      std::vector<std::int16_t> w(static_cast<std::size_t>(taps), 0);
      for (int left = 16384, k = 0; k < taps; ++k) {
        auto& wk = w[static_cast<std::size_t>(k)];
        wk = static_cast<std::int16_t>(k + 1 == taps ? left : rng.range(0, left));
        left -= wk;
      }
      std::vector<std::int16_t> want(n);
      for (std::size_t i = 0; i < n; ++i) {
        int sum = 0;
        for (std::size_t k = 0; k < rows.size(); ++k) sum += w[k] * rows[k][i];
        want[i] = static_cast<std::int16_t>((sum + (1 << 13)) >> 14);
      }
      for (const Isa isa : supported_isas()) {
        std::vector<std::int16_t> got(n + kPad, -1);
        kernels::scale::table(isa).vscale_q7(ptrs.data(), w.data(), taps, got.data(), count);
        CHECK_MSG(std::equal(want.begin(), want.end(), got.begin()),
                  test::show(isa) + " taps " + std::to_string(taps) + " count " +
                      std::to_string(count));
      }
    }
  }
}

// One pixel of the Q7 4:4:4 formula (kernels/convert/kernels.hpp), written to
// a packed layout (kernels::convert::Packed index p).
void model_pixel(std::uint8_t* d, int p, int y, int u, int v, const YuvToRgbCoeffs& c) {
  const auto clamp = [](int t) {
    return static_cast<std::uint8_t>(t < 0 ? 0 : t > 255 ? 255 : t);
  };
  const int ly = c.cy * (y - (c.y_offset << 7)) + (1 << 19), cu = u - 16384, cv = v - 16384;
  const int ri = p == kernels::convert::kBgra ? 2 : 0;
  d[ri] = clamp((ly + c.crv * cv) >> 20);
  d[1] = clamp((ly - c.cgu * cu - c.cgv * cv) >> 20);
  d[2 - ri] = clamp((ly + c.cbu * cu) >> 20);
  if (p != kernels::convert::kRgb24) d[3] = 255;
}

// The 4:4:4 row kernel against its formula, on every tier. Buffers carry the
// row padding that kernels may touch.
TEST(Yuv444RowKernelsMatchTheFormula) {
  namespace kc = kernels::convert;
  constexpr std::size_t kPad = 64;
  const YuvToRgbCoeffs coeffs[] = {yuv_to_rgb_coeffs(ColorMatrix::bt601, ColorRange::limited),
                                   yuv_to_rgb_coeffs(ColorMatrix::bt709, ColorRange::full),
                                   yuv_to_rgb_coeffs(ColorMatrix::bt709, ColorRange::limited)};
  test::Rng rng(444);
  const auto sample = [&](std::size_t i) {
    // Q7 samples; every fifth is an extreme to hit the clamps.
    return static_cast<std::int16_t>(i % 5 == 0 ? (rng.byte() & 1 ? 255 * 128 : 0)
                                                : rng.range(0, 255 * 128));
  };
  for (const int width : {1, 2, 15, 16, 17, 31, 33, 64, 70, 127, 128, 255}) {
    const auto w = static_cast<std::size_t>(width);
    std::vector<std::int16_t> y(w + kPad), u(w + kPad), v(w + kPad);
    for (std::size_t i = 0; i < w; ++i) {
      y[i] = sample(i);
      u[i] = sample(i);
      v[i] = sample(i);
    }
    for (const YuvToRgbCoeffs& c : coeffs) {
      for (const int p : {kc::kRgb24, kc::kRgba, kc::kBgra}) {
        const int bpp = p == kc::kRgb24 ? 3 : 4;
        std::vector<std::uint8_t> want(w * 4, 255);
        for (std::size_t i = 0; i < w; ++i) {
          model_pixel(&want[i * static_cast<std::size_t>(bpp)], p, y[i], u[i], v[i], c);
        }
        for (const Isa isa : supported_isas()) {
          std::vector<std::uint8_t> got(w * 4 + kPad, 255);
          kc::table(isa).yuv444_q7_to_packed[p](y.data(), u.data(), v.data(), got.data(), width,
                                                c);
          CHECK_MSG(std::memcmp(want.data(), got.data(), w * static_cast<std::size_t>(bpp)) == 0,
                    test::show(isa) + " width " + std::to_string(width) + " packed " +
                        std::to_string(p));
        }
      }
    }
  }
}

TEST(ScaleConvertEqualsScaleThenConvertOnEveryIsa) {
  int combo = 0;
  for (const PixelFormat sf : {PixelFormat::yuv420p, PixelFormat::nv12}) {
    for (const PixelFormat df : {PixelFormat::rgb24, PixelFormat::rgba, PixelFormat::bgra}) {
      for (const ScaleFilter filter : {ScaleFilter::bilinear, ScaleFilter::area}) {
        for (const ChromaMode mode : {ChromaMode::nearest, ChromaMode::bilinear}) {
          for (const ChromaLocation loc : {ChromaLocation::left, ChromaLocation::center}) {
            ConvertOptions opts;
            opts.matrix = (combo & 1) != 0 ? ColorMatrix::bt709 : ColorMatrix::bt601;
            opts.range = (combo & 2) != 0 ? ColorRange::full : ColorRange::limited;
            opts.chroma = mode;
            opts.chroma_loc = loc;
            ++combo;
            for (const Size& z : kSizes) {
              const Frame src = test::random_frame(sf, z.sw, z.sh, 1000 + combo);
              const auto sc = make(sf, z, df, filter, opts);
              for (const Isa isa : supported_isas()) {
                test::ScopedIsa scope(isa);
                Frame out = alloc(df, z.dw, z.dh);
                REQUIRE_OK(sc->run(src, out));
                const std::string d =
                    test::first_difference(unfused(src, z, df, filter, opts), out);
                CHECK_MSG(d.empty(), describe(sf, df, z) + " " + test::show(isa) + ": " + d);
              }
              CHECK_ALL_ISAS_MATCH([&] { return alloc(df, z.dw, z.dh); },
                                   [&](Frame& out) { return sc->run(src, out); });
            }
          }
        }
      }
    }
  }
}

TEST(ScaleConvertSlicedAndThreadedEqualWholeFrame) {
  const Size z{150, 120, 97, 83};
  ConvertOptions opts;
  opts.matrix = ColorMatrix::bt709;
  for (const PixelFormat sf : {PixelFormat::yuv420p, PixelFormat::nv12}) {
    const Frame src = test::random_frame(sf, z.sw, z.sh, 31);
    const auto sc = make(sf, z, PixelFormat::bgra, ScaleFilter::bilinear, opts);
    Frame whole = alloc(PixelFormat::bgra, z.dw, z.dh);
    REQUIRE_OK(sc->run(src, whole));
    CHECK(test::first_difference(unfused(src, z, PixelFormat::bgra, ScaleFilter::bilinear, opts),
                                 whole)
              .empty());

    for (const auto& [begin, end] : {std::pair{0, 2}, {2, 40}, {4, 36}, {30, 83}, {82, 83}}) {
      Frame part = alloc(PixelFormat::bgra, z.dw, z.dh);
      REQUIRE_OK(sc->run(src, part, RowRange{begin, end}));
      for (int y = begin; y < end; ++y) {
        CHECK(std::memcmp(whole.plane(0).row(y), part.plane(0).row(y),
                          static_cast<std::size_t>(whole.plane(0).row_bytes())) == 0);
      }
    }
    for (const int threads : {1, 2, 3, 8}) {
      ThreadPool pool(threads);
      Frame out = alloc(PixelFormat::bgra, z.dw, z.dh);
      for (int run = 0; run < 3; ++run) {
        REQUIRE_OK(sc->run(src, out, &pool));
        const std::string d = test::first_difference(whole, out);
        CHECK_MSG(d.empty(), "threads " + std::to_string(threads) + ": " + d);
      }
    }
    Frame inline_out = alloc(PixelFormat::bgra, z.dw, z.dh);
    REQUIRE_OK(sc->run(src, inline_out, static_cast<ThreadPool*>(nullptr)));
    CHECK(test::first_difference(whole, inline_out).empty());
  }
}

TEST(ScaleConvertRejectsBadArguments) {
  std::unique_ptr<ScaleConvert> sc;
  const ConvertOptions opts;
  CHECK(ScaleConvert::create(PixelFormat::rgb24, 8, 8, PixelFormat::rgb24, 4, 4,
                             ScaleFilter::bilinear, opts, sc) == Status::unsupported_format);
  CHECK(ScaleConvert::create(PixelFormat::yuv420p, 8, 8, PixelFormat::nv12, 4, 4,
                             ScaleFilter::bilinear, opts, sc) == Status::unsupported_format);
  CHECK(ScaleConvert::create(PixelFormat::yuv420p, 0, 8, PixelFormat::rgb24, 4, 4,
                             ScaleFilter::bilinear, opts, sc) == Status::invalid_argument);

  const Size z{16, 16, 9, 9};
  sc = make(PixelFormat::yuv420p, z, PixelFormat::rgb24, ScaleFilter::area, opts);
  const Frame src = test::random_frame(PixelFormat::yuv420p, 16, 16, 1);
  Frame dst = alloc(PixelFormat::rgb24, 9, 9);
  CHECK(sc->run(src, dst, RowRange{1, 5}) == Status::invalid_argument);  // odd begin
  CHECK(sc->run(src, dst, RowRange{0, 5}) == Status::invalid_argument);  // odd end
  CHECK(sc->run(src, dst, RowRange{0, 10}) == Status::invalid_argument);
  CHECK(sc->run(src, dst, RowRange{0, 9}) == Status::ok);  // odd end at the height is fine
  Frame wrong_size = alloc(PixelFormat::rgb24, 10, 9);
  CHECK(sc->run(src, wrong_size) == Status::invalid_argument);
  Frame wrong_format = alloc(PixelFormat::rgba, 9, 9);
  CHECK(sc->run(src, wrong_format) == Status::invalid_argument);
  const Frame wrong_src = test::random_frame(PixelFormat::nv12, 16, 16, 1);
  CHECK(sc->run(wrong_src, dst) == Status::invalid_argument);
}

}  // namespace
}  // namespace lanecast
