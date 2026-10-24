// fused suite: ScaleConvert equals scale() then convert() bit for bit, for
// sliced and threaded runs and under every supported ISA tier.
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
