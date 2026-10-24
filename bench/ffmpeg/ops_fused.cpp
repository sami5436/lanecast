// Combined scale and convert, yuv420p 1080p -> rgb24 720p. swscale does both
// in one context; lanecast is measured as the two-pass Scaler then convert
// and, when the build has it, as the fused ScaleConvert.
#include <memory>

extern "C" {
#include <libswscale/swscale.h>
}

#include "compare.hpp"
#include "lanecast/pipeline.hpp"

namespace lanecast::bench::ff {
namespace {

constexpr int kSrcWidth = 1920, kSrcHeight = 1080, kDstWidth = 1280, kDstHeight = 720;

ConvertOptions convert_options() {
  ConvertOptions o;
  o.chroma = ChromaMode::bilinear;
  o.chroma_loc = ChromaLocation::left;
  return o;
}

LanecastOp two_pass(const Frame& src, Frame& dst) {
  std::unique_ptr<Scaler> made;
  if (Scaler::create(PixelFormat::yuv420p, kSrcWidth, kSrcHeight, kDstWidth, kDstHeight,
                     ScaleFilter::bilinear, made, ChromaLocation::left) != Status::ok) {
    return {};
  }
  std::shared_ptr<Scaler> scaler = std::move(made);
  auto tmp = std::make_shared<Frame>();
  if (Frame::allocate(PixelFormat::yuv420p, kDstWidth, kDstHeight, *tmp) != Status::ok) return {};
  return [scaler, tmp, &src, &dst](ThreadPool* pool) {
    const Status st = run_sliced(pool, kDstHeight, 2, [&](RowRange r) {
      return scaler->run(src, *tmp, r);
    });
    if (st != Status::ok) return st;
    return run_sliced(pool, kDstHeight, 2, [&](RowRange r) {
      return convert(*tmp, dst, convert_options(), r);
    });
  };
}

LanecastOp fused(const Frame& src, Frame& dst) {
  std::unique_ptr<ScaleConvert> made;
  if (ScaleConvert::create(PixelFormat::yuv420p, kSrcWidth, kSrcHeight, PixelFormat::rgb24,
                           kDstWidth, kDstHeight, ScaleFilter::bilinear, convert_options(),
                           made) != Status::ok) {
    return {};
  }
  std::shared_ptr<ScaleConvert> sc = std::move(made);
  return [sc, &src, &dst](ThreadPool* pool) { return sc->run(src, dst, pool); };
}

void add(std::vector<Comparison>& out, const char* name, LanecastBinder binder) {
  Comparison c;
  c.op = std::string("yuv420p 1080p>rgb24 720p ") + name;
  c.src_format = PixelFormat::yuv420p;
  c.dst_format = PixelFormat::rgb24;
  c.src_width = kSrcWidth;
  c.src_height = kSrcHeight;
  c.dst_width = kDstWidth;
  c.dst_height = kDstHeight;
  c.sws.flags = SWS_BILINEAR;
  c.sws.loc = ChromaLocation::left;
  c.lanecast = std::move(binder);
  out.push_back(std::move(c));
}

}  // namespace

void add_fused_comparisons(std::vector<Comparison>& out) {
  add(out, "Scaler+convert", two_pass);
  add(out, "ScaleConvert", fused);
}

}  // namespace lanecast::bench::ff
