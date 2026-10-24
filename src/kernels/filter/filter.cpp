// Front end: validation, LUT build, edge extension, row loop.
#include "lanecast/filter.hpp"

#include "kernels/filter/kernels.hpp"

namespace lanecast {

Status alpha_blend(const Frame& overlay, Frame& base, RowRange rows) {
  if (overlay.empty() || base.empty() || !overlay.same_shape(base)) return Status::invalid_argument;
  if (!pixfmt_info(base.format()).has_alpha) return Status::unsupported_format;
  RowRange r;
  if (Status s = resolve_rows(base, rows, r); s != Status::ok) return s;
  const auto blend = kernels::filter::active_table().blend;
  const Plane op = overlay.plane(0);
  const Plane bp = base.plane(0);
  for (int y = r.begin; y < r.end; ++y) blend(op.row(y), bp.row(y), base.width());
  return Status::ok;
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
