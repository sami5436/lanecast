// Fused yuv -> scale -> rgb row-strip pass. Uses the scale and convert row
// kernels directly (kernels/scale/kernels.hpp, kernels/convert/kernels.hpp)
// with strip buffers from core/scratch.hpp slots 1 and 2.
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/scratch.hpp"
#include "kernels/convert/kernels.hpp"
#include "kernels/scale/kernels.hpp"
#include "lanecast/pipeline.hpp"

namespace lanecast {
namespace {

constexpr int kStripRows = 32;   // destination luma rows per strip
constexpr int kCenterPos = 128;  // sample centered in its pixel, see scale.hpp
constexpr int kWeightPad = 16;   // zero weights after the last row, see FilterView

std::size_t align64(std::size_t n) { return (n + 63) & ~static_cast<std::size_t>(63); }

// Filters and geometry of one scaled plane. Same tables and padding as the
// Scaler builds, so the row kernels see identical inputs.
struct PlaneJob {
  int channels = 0;
  int dst_w = 0;
  ScaleFilterTable hx, hy;
  std::vector<std::int16_t> hx_weights;
  kernels::scale::FilterView hview{};
  std::size_t mid_stride = 0;  // int16 elements per intermediate row, incl. slack
  std::size_t out_stride = 0;  // bytes per strip row, incl. slack
};

Status build_job(PixelFormat format, int plane, int src_w, int src_h, int dst_w, int dst_h,
                 ScaleFilter filter, ChromaLocation loc, PlaneJob& j) {
  const auto pi = static_cast<std::size_t>(plane);
  j.channels = pixfmt_info(format).bytes_per_pixel[pi];
  j.dst_w = plane_width(format, plane, dst_w);
  const int shift_x = pixfmt_info(format).shift_x[pi];
  const int pos_x =
      shift_x == 0 ? kCenterPos : ((loc == ChromaLocation::left ? 0 : kCenterPos) + 128) >> shift_x;
  if (Status s = build_scale_filter(filter, plane_width(format, plane, src_w), j.dst_w, j.hx,
                                    pos_x, pos_x);
      s != Status::ok) {
    return s;
  }
  if (Status s = build_scale_filter(filter, plane_height(format, plane, src_h),
                                    plane_height(format, plane, dst_h), j.hy);
      s != Status::ok) {
    return s;
  }
  const int taps = j.hx.taps;
  const int stride = taps <= 2 ? taps : (taps + 3) & ~3;
  j.hx_weights.assign(
      static_cast<std::size_t>(j.dst_w) * static_cast<std::size_t>(stride) + kWeightPad, 0);
  for (int x = 0; x < j.dst_w; ++x) {
    std::copy_n(j.hx.weights.data() + static_cast<std::size_t>(x) * taps, taps,
                j.hx_weights.data() + static_cast<std::size_t>(x) * stride);
  }
  j.hview = {taps, stride, j.hx.start.data(), j.hx_weights.data()};
  const auto row_bytes = static_cast<std::size_t>(j.dst_w) * j.channels;
  j.mid_stride = align64(row_bytes * 2 + 64) / 2;
  j.out_stride = align64(row_bytes + kRowPadding);
  return Status::ok;
}

PixelFormat packed_format(int packed) noexcept {
  return packed == kernels::convert::kRgb24  ? PixelFormat::rgb24
         : packed == kernels::convert::kRgba ? PixelFormat::rgba
                                             : PixelFormat::bgra;
}

int packed_index(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::rgb24: return kernels::convert::kRgb24;
    case PixelFormat::rgba: return kernels::convert::kRgba;
    case PixelFormat::bgra: return kernels::convert::kBgra;
    default: return -1;
  }
}

// Scales destination rows [o0, o1) of one plane into `out`, rows out_stride
// apart, with the Scaler's ring of horizontally scaled rows in `scratch`.
void scale_rows(const PlaneJob& j, const kernels::scale::Table& t, const Plane& sp, int o0,
                int o1, std::uint8_t* out, std::uint8_t* scratch) {
  const int taps = j.hy.taps;
  auto** ptrs = reinterpret_cast<const std::int16_t**>(scratch);
  auto* ring = reinterpret_cast<std::int16_t*>(
      scratch + align64(static_cast<std::size_t>(taps) * sizeof(void*)));
  const auto hscale = t.hscale[j.channels];
  const int count = j.dst_w * j.channels;
  int have_end = 0;  // source rows below this are already in the ring
  for (int o = o0; o < o1; ++o) {
    const int s0 = j.hy.start[static_cast<std::size_t>(o)];
    for (int sr = std::max(s0, have_end); sr < s0 + taps; ++sr) {
      hscale(sp.row(sr), ring + static_cast<std::size_t>(sr % taps) * j.mid_stride, j.dst_w,
             j.hview);
    }
    have_end = s0 + taps;
    for (int k = 0; k < taps; ++k) {
      ptrs[k] = ring + static_cast<std::size_t>((s0 + k) % taps) * j.mid_stride;
    }
    t.vscale(ptrs, j.hy.weights.data() + static_cast<std::size_t>(o) * taps, taps,
             out + static_cast<std::size_t>(o - o0) * j.out_stride, count);
  }
}

}  // namespace

struct ScaleConvert::Impl {
  PixelFormat src_format;
  int src_w, src_h, dst_w, dst_h;
  ConvertOptions opts;
  int packed = 0;
  PlaneJob luma, chroma;
  std::size_t ring_bytes = 0;   // scale intermediate rows (scratch slot 0)
  std::size_t chroma_rows = 0;  // chroma rows a strip holds
};

ScaleConvert::ScaleConvert(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ScaleConvert::~ScaleConvert() = default;

Status ScaleConvert::create(PixelFormat src_format, int src_width, int src_height,
                            PixelFormat dst_format, int dst_width, int dst_height,
                            ScaleFilter filter, const ConvertOptions& opts,
                            std::unique_ptr<ScaleConvert>& out) {
  const int packed = packed_index(dst_format);
  if ((src_format != PixelFormat::yuv420p && src_format != PixelFormat::nv12) || packed < 0) {
    return Status::unsupported_format;
  }
  for (const int d : {src_width, src_height, dst_width, dst_height}) {
    if (d < 1 || d > kMaxDimension) return Status::invalid_argument;
  }
  auto impl = std::make_unique<Impl>();
  impl->src_format = src_format;
  impl->src_w = src_width;
  impl->src_h = src_height;
  impl->dst_w = dst_width;
  impl->dst_h = dst_height;
  impl->opts = opts;
  impl->packed = packed;
  for (int p = 0; p < 2; ++p) {
    PlaneJob& j = p == 0 ? impl->luma : impl->chroma;
    if (Status s = build_job(src_format, p, src_width, src_height, dst_width, dst_height, filter,
                             opts.chroma_loc, j);
        s != Status::ok) {
      return s;
    }
    const auto taps = static_cast<std::size_t>(j.hy.taps);
    impl->ring_bytes =
        std::max(impl->ring_bytes, align64(taps * sizeof(void*)) + taps * j.mid_stride * 2);
  }
  impl->chroma_rows = kStripRows / 2 + 2;
  out.reset(new ScaleConvert(std::move(impl)));
  return Status::ok;
}

Status ScaleConvert::run(const Frame& src, Frame& dst, RowRange rows) const {
  const Impl& m = *impl_;
  if (src.empty() || dst.empty() || src.format() != m.src_format ||
      dst.format() != packed_format(m.packed) || src.width() != m.src_w ||
      src.height() != m.src_h || dst.width() != m.dst_w || dst.height() != m.dst_h) {
    return Status::invalid_argument;
  }
  // Same alignment as convert on the intermediate 4:2:0 frame.
  const int end = rows.end < 0 ? m.dst_h : rows.end;
  if (rows.begin < 0 || rows.begin > end || end > m.dst_h || (rows.begin & 1) != 0 ||
      ((end & 1) != 0 && end != m.dst_h)) {
    return Status::invalid_argument;
  }

  const bool nv12 = m.src_format == PixelFormat::nv12;
  const bool bilinear = m.opts.chroma == ChromaMode::bilinear;
  const bool fast = !bilinear && m.opts.precision == Precision::fast;
  const auto& st = kernels::scale::active_table();
  const auto& ct = kernels::convert::active_table();
  const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(m.opts.matrix, m.opts.range);
  const int ch = plane_height(m.src_format, 1, m.dst_h);
  const std::size_t luma_bytes = static_cast<std::size_t>(kStripRows) * m.luma.out_stride;
  const std::size_t plane_bytes = m.chroma_rows * m.chroma.out_stride;
  std::uint8_t* ring = detail::thread_scratch(0, m.ring_bytes);
  std::uint8_t* ybuf = detail::thread_scratch(1, luma_bytes);
  std::uint8_t* ubuf = detail::thread_scratch(2, nv12 ? plane_bytes : 2 * plane_bytes);
  if (ring == nullptr || ybuf == nullptr || ubuf == nullptr) return Status::out_of_memory;
  std::uint8_t* vbuf = ubuf + plane_bytes;

  const Plane dp = dst.plane(0);
  for (int a = rows.begin; a < end; a += kStripRows) {
    const int b = std::min(a + kStripRows, end);
    // Chroma rows this strip's luma rows read (one more each side for bilinear).
    const int c0 = std::max((a >> 1) - (bilinear ? 1 : 0), 0);
    const int c1 = std::min(((b + 1) >> 1) + (bilinear ? 1 : 0), ch);
    scale_rows(m.luma, st, src.plane(0), a, b, ybuf, ring);
    scale_rows(m.chroma, st, src.plane(1), c0, c1, ubuf, ring);
    if (!nv12) scale_rows(m.chroma, st, src.plane(2), c0, c1, vbuf, ring);
    const auto crow = [&](std::uint8_t* base, int row) {
      return base + static_cast<std::size_t>(row - c0) * m.chroma.out_stride;
    };
    for (int y = a; y < b; ++y) {
      const int cy = y >> 1;
      const int cyn = std::clamp(cy + ((y & 1) ? 1 : -1), 0, ch - 1);
      const std::uint8_t* yr = ybuf + static_cast<std::size_t>(y - a) * m.luma.out_stride;
      std::uint8_t* out = dp.row(y);
      if (nv12 && bilinear) {
        const auto fn = ct.nv12_bilinear_to_packed[m.packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(yr, crow(ubuf, cy), crow(ubuf, cyn), out, m.dst_w, c, m.opts.chroma_loc);
      } else if (nv12) {
        const auto fn = fast ? ct.nv12_fast_to_packed[m.packed] : ct.nv12_to_packed[m.packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(yr, crow(ubuf, cy), out, m.dst_w, c);
      } else if (bilinear) {
        const auto fn = ct.yuv420_bilinear_to_packed[m.packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(yr, crow(ubuf, cy), crow(ubuf, cyn), crow(vbuf, cy), crow(vbuf, cyn), out, m.dst_w, c,
           m.opts.chroma_loc);
      } else {
        const auto fn =
            fast ? ct.yuv420_fast_to_packed[m.packed] : ct.yuv420_to_packed[m.packed];
        if (fn == nullptr) return Status::not_implemented;
        fn(yr, crow(ubuf, cy), crow(vbuf, cy), out, m.dst_w, c);
      }
    }
  }
  return Status::ok;
}

Status ScaleConvert::run(const Frame& src, Frame& dst, ThreadPool* pool) const {
  return run_sliced(
      pool, impl_->dst_h, 2, [&](RowRange r) { return run(src, dst, r); }, kStripRows);
}

}  // namespace lanecast
