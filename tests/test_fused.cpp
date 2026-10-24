// fused suite: ScaleConvert equals scale() then convert() bit for bit, for
// sliced and threaded runs and under every supported ISA tier.
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

}  // namespace
}  // namespace lanecast
