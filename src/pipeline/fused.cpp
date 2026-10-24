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

int packed_index(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::rgb24: return kernels::convert::kRgb24;
    case PixelFormat::rgba: return kernels::convert::kRgba;
    case PixelFormat::bgra: return kernels::convert::kBgra;
    default: return -1;
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
  (void)src;
  (void)dst;
  (void)rows;
  return Status::not_implemented;
}

Status ScaleConvert::run(const Frame& src, Frame& dst, ThreadPool* pool) const {
  (void)src;
  (void)dst;
  (void)pool;
  return Status::not_implemented;
}

}  // namespace lanecast
