// Scaler cases: half-size and 1.5x resize, bilinear and area.
#include "bench.hpp"
#include "lanecast/scale.hpp"

namespace lanecast::bench {
namespace {

// Even output sizes keep 4:2:0 row ranges aligned.
int even(int v) { return v < 2 ? 2 : v & ~1; }

Case scale_case(const char* label, PixelFormat fmt, ScaleFilter filter, int num, int den) {
  Case c;
  c.name = std::string("scale ") + label + " " + std::string(pixfmt_name(fmt)) + " " +
           std::to_string(num) + "/" + std::to_string(den);
  c.make = [=](Size s) {
    const int dw = even(s.width * num / den);
    const int dh = even(s.height * num / den);
    auto src = make_frame(fmt, s.width, s.height, 2);
    auto dst = std::make_shared<Frame>();
    (void)Frame::allocate(fmt, dw, dh, *dst);
    std::unique_ptr<Scaler> made;
    std::shared_ptr<Scaler> scaler;
    if (Scaler::create(fmt, s.width, s.height, dw, dh, filter, made) == Status::ok) {
      scaler = std::move(made);
    }
    Instance in;
    in.bytes = frame_bytes(*src) + frame_bytes(*dst);
    in.run = sliced(dh, pixfmt_row_alignment(fmt), [src, dst, scaler](RowRange rows) {
      return scaler ? scaler->run(*src, *dst, rows) : Status::not_implemented;
    });
    return in;
  };
  return c;
}

}  // namespace

void register_scale(std::vector<Case>& out) {
  using F = PixelFormat;
  out.push_back(scale_case("bilinear", F::yuv420p, ScaleFilter::bilinear, 1, 2));
  out.push_back(scale_case("area", F::yuv420p, ScaleFilter::area, 1, 2));
  out.push_back(scale_case("bilinear", F::rgba, ScaleFilter::bilinear, 1, 2));
  out.push_back(scale_case("area", F::rgba, ScaleFilter::area, 1, 2));
  out.push_back(scale_case("bilinear", F::yuv420p, ScaleFilter::bilinear, 3, 2));
  out.push_back(scale_case("bilinear", F::rgba, ScaleFilter::bilinear, 3, 2));
}

}  // namespace lanecast::bench
