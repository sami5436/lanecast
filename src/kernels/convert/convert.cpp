// Front end: validation, format routing, row loop. Calls kernels through
// active_table() only.
#include "lanecast/convert.hpp"

#include <algorithm>

#include "kernels/convert/kernels.hpp"

namespace lanecast {
namespace {

using namespace kernels::convert;

// Index into the per-layout tables, or -1 for non-packed formats.
int packed_index(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::rgb24: return kRgb24;
    case PixelFormat::rgba: return kRgba;
    case PixelFormat::bgra: return kBgra;
    default: return -1;
  }
}

bool is_yuv420_family(PixelFormat f) noexcept {
  return f == PixelFormat::yuv420p || f == PixelFormat::nv12;
}

Status yuv_to_packed(const Frame& src, Frame& dst, const ConvertOptions& opts, RowRange r,
                     int packed) {
  const Table& t = active_table();
  const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(opts.matrix, opts.range);
  const bool nv12 = src.format() == PixelFormat::nv12;
  const bool bilinear = opts.chroma == ChromaMode::bilinear;
  const Plane py = src.plane(0), pu = src.plane(1), pd = dst.plane(0);
  const Plane pv = nv12 ? src.plane(1) : src.plane(2);
  const int w = src.width(), ch = pu.height;
  for (int y = r.begin; y < r.end; ++y) {
    const int cy = y >> 1;
    // Vertical neighbor for bilinear: the other chroma row nearest this luma row.
    const int cyn = std::clamp(cy + ((y & 1) ? 1 : -1), 0, ch - 1);
    if (nv12) {
      if (bilinear) {
        const auto fn = t.nv12_bilinear_to_packed[packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(py.row(y), pu.row(cy), pu.row(cyn), pd.row(y), w, c);
      } else {
        const auto fn = t.nv12_to_packed[packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(py.row(y), pu.row(cy), pd.row(y), w, c);
      }
    } else if (bilinear) {
      const auto fn = t.yuv420_bilinear_to_packed[packed];
      if (fn == nullptr) return Status::not_implemented;
      fn(py.row(y), pu.row(cy), pu.row(cyn), pv.row(cy), pv.row(cyn), pd.row(y), w, c);
    } else {
      const auto fn = t.yuv420_to_packed[packed];
      if (fn == nullptr) return Status::not_implemented;
      fn(py.row(y), pu.row(cy), pv.row(cy), pd.row(y), w, c);
    }
  }
  return Status::ok;
}

}  // namespace

bool convert_supported(PixelFormat src, PixelFormat dst) noexcept {
  return is_yuv420_family(src) && packed_index(dst) >= 0;
}

Status convert(const Frame& src, Frame& dst, const ConvertOptions& opts, RowRange rows) {
  if (src.empty() || dst.empty() || src.width() != dst.width() || src.height() != dst.height()) {
    return Status::invalid_argument;
  }
  if (!convert_supported(src.format(), dst.format())) return Status::not_implemented;
  RowRange r;
  if (const Status s = resolve_rows(src, rows, r); s != Status::ok) return s;
  return yuv_to_packed(src, dst, opts, r, packed_index(dst.format()));
}

}  // namespace lanecast
