// yuv420p scaling comparisons with swscale's matching filter flag. Chroma is
// left sited on both sides, as in H.264 video.
#include <memory>
#include <string>

extern "C" {
#include <libswscale/swscale.h>
}

#include "compare.hpp"
#include "lanecast/scale.hpp"

namespace lanecast::bench::ff {
namespace {

LanecastBinder scaler_binder(ScaleFilter filter) {
  return [filter](const Frame& src, Frame& dst) -> LanecastOp {
    std::unique_ptr<Scaler> made;
    if (Scaler::create(src.format(), src.width(), src.height(), dst.width(), dst.height(),
                       filter, made, ChromaLocation::left) != Status::ok) {
      return {};
    }
    std::shared_ptr<Scaler> scaler = std::move(made);
    return sliced_op(dst, pixfmt_row_alignment(src.format()), [scaler, &src, &dst](RowRange rows) {
      return scaler->run(src, dst, rows);
    });
  };
}

void add(std::vector<Comparison>& out, const std::string& name, int sw, int sh, int dw, int dh,
         ScaleFilter filter, unsigned flags) {
  Comparison c;
  c.op = "scale yuv420p " + name;
  c.src_width = sw;
  c.src_height = sh;
  c.dst_width = dw;
  c.dst_height = dh;
  c.src_format = c.dst_format = PixelFormat::yuv420p;
  c.sws.flags = flags;
  c.sws.loc = ChromaLocation::left;
  c.lanecast = scaler_binder(filter);
  out.push_back(std::move(c));
}

}  // namespace

void add_scale_comparisons(std::vector<Comparison>& out) {
  add(out, "1080p>720p bilinear", 1920, 1080, 1280, 720, ScaleFilter::bilinear, SWS_BILINEAR);
  add(out, "1080p>360p bilinear", 1920, 1080, 640, 360, ScaleFilter::bilinear, SWS_BILINEAR);
  add(out, "4K>1080p area", 3840, 2160, 1920, 1080, ScaleFilter::area, SWS_AREA);
  add(out, "720p>1080p bilinear", 1280, 720, 1920, 1080, ScaleFilter::bilinear, SWS_BILINEAR);
}

}  // namespace lanecast::bench::ff
