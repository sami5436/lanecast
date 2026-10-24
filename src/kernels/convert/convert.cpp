// Front end: validation, format routing, row loop. Calls kernels through
// active_table() only.
#include "lanecast/convert.hpp"

#include <algorithm>
#include <cstring>

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
  const bool fast = !bilinear && opts.precision == Precision::fast;
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
        fn(py.row(y), pu.row(cy), pu.row(cyn), pd.row(y), w, c, opts.chroma_loc);
      } else {
        const auto fn = fast ? t.nv12_fast_to_packed[packed] : t.nv12_to_packed[packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(py.row(y), pu.row(cy), pd.row(y), w, c);
      }
    } else if (bilinear) {
      const auto fn = t.yuv420_bilinear_to_packed[packed];
      if (fn == nullptr) return Status::not_implemented;
      fn(py.row(y), pu.row(cy), pu.row(cyn), pv.row(cy), pv.row(cyn), pd.row(y), w, c,
         opts.chroma_loc);
    } else {
      const auto fn = fast ? t.yuv420_fast_to_packed[packed] : t.yuv420_to_packed[packed];
      if (fn == nullptr) return Status::not_implemented;
      fn(py.row(y), pu.row(cy), pv.row(cy), pd.row(y), w, c);
    }
  }
  return Status::ok;
}

void copy_rows(const Plane& s, const Plane& d, int begin, int end) {
  for (int y = begin; y < end; ++y) {
    std::memcpy(d.row(y), s.row(y), static_cast<std::size_t>(s.row_bytes()));
  }
}

// Chroma plane rows covered by luma rows [r.begin, r.end).
int chroma_begin(RowRange r) noexcept { return r.begin >> 1; }
int chroma_end(RowRange r) noexcept { return (r.end + 1) >> 1; }

Status packed_to_yuv(const Frame& src, Frame& dst, const ConvertOptions& opts, RowRange r,
                     int packed) {
  const Table& t = active_table();
  const RgbToYuvCoeffs k = rgb_to_yuv_coeffs(opts.matrix, opts.range);
  const bool nv12 = dst.format() == PixelFormat::nv12;
  const auto fn420 = t.packed_to_yuv420[packed];
  const auto fn12 = t.packed_to_nv12[packed];
  if (nv12 ? fn12 == nullptr : fn420 == nullptr) return Status::not_implemented;
  const Plane ps = src.plane(0), py = dst.plane(0), pu = dst.plane(1);
  const Plane pv = nv12 ? dst.plane(1) : dst.plane(2);
  const int last = src.height() - 1;
  for (int y = r.begin; y < r.end; y += 2) {
    const int y1 = std::min(y + 1, last);  // odd height: the last row pairs with itself
    if (nv12) {
      fn12(ps.row(y), ps.row(y1), py.row(y), py.row(y1), pu.row(y >> 1), src.width(), k,
           opts.chroma_loc);
    } else {
      fn420(ps.row(y), ps.row(y1), py.row(y), py.row(y1), pu.row(y >> 1), pv.row(y >> 1),
            src.width(), k, opts.chroma_loc);
    }
  }
  return Status::ok;
}

Status packed_to_packed(const Frame& src, Frame& dst, RowRange r, int sp, int dp) {
  const auto fn = active_table().swizzle[sp][dp];
  if (fn == nullptr) return Status::not_implemented;
  for (int y = r.begin; y < r.end; ++y) fn(src.plane(0).row(y), dst.plane(0).row(y), src.width());
  return Status::ok;
}

Status yuv420_nv12(const Frame& src, Frame& dst, RowRange r) {
  const Table& t = active_table();
  copy_rows(src.plane(0), dst.plane(0), r.begin, r.end);
  const int cw = plane_width(src.format(), 1, src.width());
  if (src.format() == PixelFormat::nv12) {
    if (t.deinterleave_uv == nullptr) return Status::not_implemented;
    for (int y = chroma_begin(r); y < chroma_end(r); ++y) {
      t.deinterleave_uv(src.plane(1).row(y), dst.plane(1).row(y), dst.plane(2).row(y), cw);
    }
  } else {
    if (t.interleave_uv == nullptr) return Status::not_implemented;
    for (int y = chroma_begin(r); y < chroma_end(r); ++y) {
      t.interleave_uv(src.plane(1).row(y), src.plane(2).row(y), dst.plane(1).row(y), cw);
    }
  }
  return Status::ok;
}

// gray8 -> yuv420p / nv12: Y copied, chroma set to 128.
void gray_to_yuv(const Frame& src, Frame& dst, RowRange r) {
  copy_rows(src.plane(0), dst.plane(0), r.begin, r.end);
  for (int p = 1; p < dst.plane_count(); ++p) {
    const Plane pl = dst.plane(p);
    for (int y = chroma_begin(r); y < chroma_end(r); ++y) {
      std::memset(pl.row(y), 128, static_cast<std::size_t>(pl.row_bytes()));
    }
  }
}

Status gray_to_packed(const Frame& src, Frame& dst, RowRange r, int packed) {
  const auto fn = active_table().gray_to_packed[packed];
  if (fn == nullptr) return Status::not_implemented;
  for (int y = r.begin; y < r.end; ++y) fn(src.plane(0).row(y), dst.plane(0).row(y), src.width());
  return Status::ok;
}

Status packed_to_gray(const Frame& src, Frame& dst, const ConvertOptions& opts, RowRange r,
                      int packed) {
  const auto fn = active_table().packed_to_gray[packed];
  if (fn == nullptr) return Status::not_implemented;
  const RgbToYuvCoeffs k = rgb_to_yuv_coeffs(opts.matrix, ColorRange::full);
  for (int y = r.begin; y < r.end; ++y) {
    fn(src.plane(0).row(y), dst.plane(0).row(y), src.width(), k);
  }
  return Status::ok;
}

}  // namespace

// Every pair is covered: see the table in convert.hpp.
bool convert_supported(PixelFormat, PixelFormat) noexcept { return true; }

Status convert(const Frame& src, Frame& dst, const ConvertOptions& opts, RowRange rows) {
  if (src.empty() || dst.empty() || src.width() != dst.width() || src.height() != dst.height()) {
    return Status::invalid_argument;
  }
  // Rows must satisfy both formats' alignment (2 for 4:2:0, 1 otherwise).
  RowRange r, checked;
  if (const Status s = resolve_rows(src, rows, r); s != Status::ok) return s;
  if (const Status s = resolve_rows(dst, r, checked); s != Status::ok) return s;

  const PixelFormat sf = src.format(), df = dst.format();
  const int sp = packed_index(sf), dp = packed_index(df);
  if (sf == df) {
    for (int p = 0; p < src.plane_count(); ++p) {
      const bool sub = pixfmt_info(sf).shift_y[static_cast<std::size_t>(p)] != 0;
      copy_rows(src.plane(p), dst.plane(p), sub ? chroma_begin(r) : r.begin,
                sub ? chroma_end(r) : r.end);
    }
    return Status::ok;
  }
  if (is_yuv420_family(sf) && dp >= 0) return yuv_to_packed(src, dst, opts, r, dp);
  if (sp >= 0 && is_yuv420_family(df)) return packed_to_yuv(src, dst, opts, r, sp);
  if (sp >= 0 && dp >= 0) return packed_to_packed(src, dst, r, sp, dp);
  if (is_yuv420_family(sf) && is_yuv420_family(df)) return yuv420_nv12(src, dst, r);
  if (sf == PixelFormat::gray8 && is_yuv420_family(df)) {
    gray_to_yuv(src, dst, r);
    return Status::ok;
  }
  if (is_yuv420_family(sf) && df == PixelFormat::gray8) {
    copy_rows(src.plane(0), dst.plane(0), r.begin, r.end);
    return Status::ok;
  }
  if (sf == PixelFormat::gray8 && dp >= 0) return gray_to_packed(src, dst, r, dp);
  if (sp >= 0 && df == PixelFormat::gray8) return packed_to_gray(src, dst, opts, r, sp);
  return Status::not_implemented;
}

}  // namespace lanecast
