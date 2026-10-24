// Front end: validation, LUT build, edge extension, row loop.
#include "lanecast/filter.hpp"

#include "kernels/filter/kernels.hpp"

namespace lanecast {

Status alpha_blend(const Frame& overlay, Frame& base, RowRange rows) {
  (void)overlay;
  (void)base;
  (void)rows;
  return Status::not_implemented;
}

Status blur(const Frame& src, Frame& dst, int radius, RowRange rows) {
  (void)src;
  (void)dst;
  (void)radius;
  (void)rows;
  return Status::not_implemented;
}

Status build_levels_lut(int brightness, int contrast_q8, std::array<std::uint8_t, 256>& lut) {
  (void)brightness;
  (void)contrast_q8;
  (void)lut;
  return Status::not_implemented;
}

Status brightness_contrast(const Frame& src, Frame& dst, int brightness, int contrast_q8,
                           RowRange rows) {
  (void)src;
  (void)dst;
  (void)brightness;
  (void)contrast_q8;
  (void)rows;
  return Status::not_implemented;
}

}  // namespace lanecast
