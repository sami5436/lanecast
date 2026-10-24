// Pixel format conversion cases.
#include "bench.hpp"
#include "lanecast/convert.hpp"

namespace lanecast::bench {
namespace {

Case convert_case(PixelFormat from, PixelFormat to) {
  Case c;
  c.name = "convert " + std::string(pixfmt_name(from)) + ">" + std::string(pixfmt_name(to));
  c.make = [from, to](Size s) {
    auto src = make_frame(from, s.width, s.height, 1);
    auto dst = std::make_shared<Frame>();
    (void)Frame::allocate(to, s.width, s.height, *dst);
    Instance in;
    in.bytes = frame_bytes(*src) + frame_bytes(*dst);
    in.run = sliced(s.height, row_alignment(from, to), [src, dst](RowRange rows) {
      return convert(*src, *dst, ConvertOptions{}, rows);
    });
    return in;
  };
  return c;
}

}  // namespace

void register_convert(std::vector<Case>& out) {
  using F = PixelFormat;
  out.push_back(convert_case(F::yuv420p, F::rgb24));
  out.push_back(convert_case(F::yuv420p, F::rgba));
  out.push_back(convert_case(F::nv12, F::rgba));
  out.push_back(convert_case(F::rgb24, F::yuv420p));
  out.push_back(convert_case(F::rgba, F::nv12));
  out.push_back(convert_case(F::yuv420p, F::nv12));
  out.push_back(convert_case(F::rgba, F::bgra));
}

}  // namespace lanecast::bench
