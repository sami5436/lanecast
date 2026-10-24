// Front end: validation, LUT build, edge extension, row loop.
#include "lanecast/filter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>

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
  if (brightness < -255 || brightness > 255 || contrast_q8 < 0 || contrast_q8 > 1024) {
    return Status::invalid_argument;
  }
  for (int v = 0; v < 256; ++v) {
    const int t = ((v - 128) * contrast_q8 + 128) >> 8;
    lut[static_cast<std::size_t>(v)] = static_cast<std::uint8_t>(std::clamp(128 + t + brightness, 0, 255));
  }
  return Status::ok;
}

Status brightness_contrast(const Frame& src, Frame& dst, int brightness, int contrast_q8,
                           RowRange rows) {
  if (src.empty() || dst.empty() || !src.same_shape(dst)) return Status::invalid_argument;
  std::array<std::uint8_t, 256> lut;
  if (Status s = build_levels_lut(brightness, contrast_q8, lut); s != Status::ok) return s;
  RowRange r;
  if (Status s = resolve_rows(dst, rows, r); s != Status::ok) return s;

  const PixelFormat fmt = src.format();
  const auto& t = kernels::filter::active_table();
  const bool in_place = &src == &dst;
  // Planes that go through the LUT: luma only for YUV, the single packed
  // plane otherwise. Other planes (chroma) are copied.
  const int lut_planes = pixfmt_info(fmt).is_yuv ? 1 : src.plane_count();
  for (int p = 0; p < src.plane_count(); ++p) {
    const Plane sp = src.plane(p);
    const Plane dp = dst.plane(p);
    const int shift = pixfmt_info(fmt).shift_y[static_cast<std::size_t>(p)];
    const int y0 = r.begin >> shift;
    const int y1 = (r.end + (1 << shift) - 1) >> shift;
    for (int y = y0; y < y1; ++y) {
      if (p >= lut_planes) {
        if (!in_place) std::memcpy(dp.row(y), sp.row(y), static_cast<std::size_t>(sp.row_bytes()));
      } else if (pixfmt_info(fmt).has_alpha) {
        t.lut_skip_alpha(sp.row(y), dp.row(y), sp.width, lut.data());
      } else {
        t.lut(sp.row(y), dp.row(y), sp.row_bytes(), lut.data());
      }
    }
  }
  return Status::ok;
}

}  // namespace lanecast
