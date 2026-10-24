// Front end: validation, LUT build, edge extension, row loop.
#include "lanecast/filter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "core/scratch.hpp"
#include "kernels/filter/kernels.hpp"

namespace lanecast {
namespace {

std::size_t align64(std::size_t n) { return (n + 63) & ~static_cast<std::size_t>(63); }

}  // namespace

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
  if (src.empty() || dst.empty() || !src.same_shape(dst) || &src == &dst ||
      (radius != 1 && radius != 2)) {
    return Status::invalid_argument;
  }
  RowRange r;
  if (Status s = resolve_rows(dst, rows, r); s != Status::ok) return s;

  const PixelFormat fmt = src.format();
  const int taps = 2 * radius + 1;
  // One thread-local scratch block per call, sized for the widest plane: the
  // edge-extended source row, a ring of horizontal sums (one slot per tap)
  // and the row pointers handed to the vertical kernel.
  std::size_t ext_bytes = 0;
  std::size_t mid_stride = 0;  // uint16 elements per ring row, incl. slack
  for (int p = 0; p < src.plane_count(); ++p) {
    const auto ch = static_cast<std::size_t>(src.plane(p).bytes_per_pixel);
    const auto bytes = static_cast<std::size_t>(src.plane(p).row_bytes());
    ext_bytes = std::max(ext_bytes, bytes + 2 * static_cast<std::size_t>(radius) * ch);
    mid_stride = std::max(mid_stride, align64(bytes * 2 + kRowPadding) / 2);
  }
  const std::size_t ext_span = align64(ext_bytes + kRowPadding);
  const std::size_t ring_span = static_cast<std::size_t>(taps) * mid_stride * 2;
  std::uint8_t* scratch =
      detail::thread_scratch(3, ext_span + ring_span + static_cast<std::size_t>(taps) * sizeof(void*));
  if (scratch == nullptr) return Status::out_of_memory;
  std::uint8_t* ext = scratch;
  auto* ring = reinterpret_cast<std::uint16_t*>(scratch + ext_span);
  auto** ptrs = reinterpret_cast<const std::uint16_t**>(scratch + ext_span + ring_span);

  const auto& t = kernels::filter::active_table();
  for (int p = 0; p < src.plane_count(); ++p) {
    const int shift = pixfmt_info(fmt).shift_y[static_cast<std::size_t>(p)];
    const Plane sp = src.plane(p);
    const Plane dp = dst.plane(p);
    const int ch = sp.bytes_per_pixel;
    const int bytes = sp.row_bytes();
    const int pad = radius * ch;
    const int y0 = r.begin >> shift;
    const int y1 = (r.end + (1 << shift) - 1) >> shift;
    // Clamped source rows are non-decreasing, so rows below have_end are
    // already in the ring (slot = row % taps) and each row is summed once.
    int have_end = std::max(y0 - radius, 0);
    for (int y = y0; y < y1; ++y) {
      for (int k = 0; k < taps; ++k) {
        const int sy = std::clamp(y + k - radius, 0, sp.height - 1);
        std::uint16_t* slot = ring + static_cast<std::size_t>(sy % taps) * mid_stride;
        if (sy >= have_end) {
          const std::uint8_t* row = sp.row(sy);
          std::memcpy(ext + pad, row, static_cast<std::size_t>(bytes));
          for (int i = 0; i < radius; ++i) {
            std::memcpy(ext + i * ch, row, static_cast<std::size_t>(ch));
            std::memcpy(ext + pad + bytes + i * ch, row + bytes - ch, static_cast<std::size_t>(ch));
          }
          t.blur_h(ext, slot, bytes, ch, radius);
          have_end = sy + 1;
        }
        ptrs[k] = slot;
      }
      t.blur_v(ptrs, dp.row(y), bytes, radius);
    }
  }
  return Status::ok;
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
