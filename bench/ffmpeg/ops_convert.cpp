// Same-size colorspace conversion comparisons (BT.601, limited-range YUV,
// full-range RGB). The reference for each uses the same chroma siting.
#include <string>

extern "C" {
#include <libswscale/swscale.h>
}

#include "compare.hpp"
#include "lanecast/convert.hpp"

namespace lanecast::bench::ff {
namespace {

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;

LanecastBinder convert_binder(const ConvertOptions& opts) {
  return [opts](const Frame& src, Frame& dst) {
    return [&src, &dst, opts](RowRange rows) { return convert(src, dst, opts, rows); };
  };
}

void add(std::vector<Comparison>& out, const std::string& name, PixelFormat from, PixelFormat to,
         unsigned flags, ChromaMode mode, ChromaLocation loc) {
  Comparison c;
  c.op = std::string(pixfmt_name(from)) + ">" + std::string(pixfmt_name(to)) + " " + name;
  c.src_format = from;
  c.dst_format = to;
  c.src_width = c.dst_width = kWidth;
  c.src_height = c.dst_height = kHeight;
  c.sws.flags = flags;
  c.sws.loc = loc;
  ConvertOptions o;
  o.chroma = mode;
  o.chroma_loc = loc;
  c.lanecast = convert_binder(o);
  out.push_back(std::move(c));
}

}  // namespace

void add_convert_comparisons(std::vector<Comparison>& out) {
  using F = PixelFormat;
  constexpr unsigned kBilinear = SWS_BILINEAR | SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND;
  add(out, "bilinear left", F::yuv420p, F::rgb24, kBilinear, ChromaMode::bilinear,
      ChromaLocation::left);
  add(out, "bilinear center", F::yuv420p, F::rgb24, kBilinear, ChromaMode::bilinear,
      ChromaLocation::center);
}

}  // namespace lanecast::bench::ff
