// Filter cases: blur, brightness and contrast, alpha blend.
#include "bench.hpp"
#include "lanecast/filter.hpp"

namespace lanecast::bench {
namespace {

Case blur_case(PixelFormat fmt, int radius) {
  Case c;
  c.name = "blur r" + std::to_string(radius) + " " + std::string(pixfmt_name(fmt));
  c.make = [=](Size s) {
    auto src = make_frame(fmt, s.width, s.height, 3);
    auto dst = std::make_shared<Frame>();
    (void)Frame::allocate(fmt, s.width, s.height, *dst);
    Instance in;
    in.bytes = frame_bytes(*src) + frame_bytes(*dst);
    in.run = sliced(s.height, pixfmt_row_alignment(fmt), [src, dst, radius](RowRange rows) {
      return blur(*src, *dst, radius, rows);
    });
    return in;
  };
  return c;
}

Case levels_case(PixelFormat fmt) {
  Case c;
  c.name = "brightness_contrast " + std::string(pixfmt_name(fmt));
  c.make = [=](Size s) {
    auto src = make_frame(fmt, s.width, s.height, 4);
    auto dst = std::make_shared<Frame>();
    (void)Frame::allocate(fmt, s.width, s.height, *dst);
    Instance in;
    in.bytes = frame_bytes(*src) + frame_bytes(*dst);
    in.run = sliced(s.height, pixfmt_row_alignment(fmt), [src, dst](RowRange rows) {
      return brightness_contrast(*src, *dst, 10, 288, rows);
    });
    return in;
  };
  return c;
}

Case blend_case(PixelFormat fmt) {
  Case c;
  c.name = "alpha_blend " + std::string(pixfmt_name(fmt));
  c.make = [=](Size s) {
    auto overlay = make_frame(fmt, s.width, s.height, 5);
    auto base = make_frame(fmt, s.width, s.height, 6);
    Instance in;
    // Reads overlay and base, writes base.
    in.bytes = 2 * frame_bytes(*base) + frame_bytes(*overlay);
    in.run = sliced(s.height, pixfmt_row_alignment(fmt), [overlay, base](RowRange rows) {
      return alpha_blend(*overlay, *base, rows);
    });
    return in;
  };
  return c;
}

}  // namespace

void register_filter(std::vector<Case>& out) {
  using F = PixelFormat;
  out.push_back(blur_case(F::rgba, 1));
  out.push_back(blur_case(F::rgba, 2));
  out.push_back(blur_case(F::gray8, 1));
  out.push_back(levels_case(F::rgb24));
  out.push_back(levels_case(F::gray8));
  out.push_back(blend_case(F::rgba));
}

}  // namespace lanecast::bench
