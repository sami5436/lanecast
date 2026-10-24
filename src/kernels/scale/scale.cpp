// Front end: filter table generation (shared by every ISA), Scaler, row
// loop over thread-local intermediate rows (core/scratch.hpp slot 0).
#include "lanecast/scale.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/scratch.hpp"
#include "kernels/scale/kernels.hpp"

namespace lanecast {
namespace {

constexpr std::int64_t kOne = 1 << 14;

// Adds weight w at source index idx (clamped into [0, src_size)) to a
// packed row whose first tap is source index `start`.
struct RowBuilder {
  int src_size;
  int start;
  std::int16_t* row;
  void add(std::int64_t idx, std::int32_t w) const {
    const int i = static_cast<int>(std::clamp<std::int64_t>(idx, 0, src_size - 1));
    row[i - start] = static_cast<std::int16_t>(row[i - start] + w);
  }
};

}  // namespace

namespace {

constexpr int kWeightPad = 16;  // zero weights after the last row, see FilterView

std::size_t align64(std::size_t n) { return (n + 63) & ~static_cast<std::size_t>(63); }

// Per-plane geometry and filters. Vertical weights are used as built;
// horizontal weights get a padded copy for the kernels.
struct PlaneJob {
  int channels = 0;
  int dst_w = 0;
  int shift_y = 0;
  ScaleFilterTable hx, hy;
  std::vector<std::int16_t> hx_weights;
  kernels::scale::FilterView hview{};
  std::size_t mid_stride = 0;  // int16 elements per intermediate row, incl. slack
};

}  // namespace

struct Scaler::Impl {
  PixelFormat format;
  int src_w, src_h, dst_w, dst_h;
  int plane_count;
  PlaneJob planes[kMaxPlanes];
};

Scaler::Scaler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Scaler::~Scaler() = default;

Status build_scale_filter(ScaleFilter filter, int src_size, int dst_size, ScaleFilterTable& out) {
  if (src_size < 1 || dst_size < 1 || src_size > kMaxDimension || dst_size > kMaxDimension) {
    return Status::invalid_argument;
  }
  const std::int64_t src = src_size;
  const std::int64_t dst = dst_size;

  int taps = 0;
  if (filter == ScaleFilter::bilinear) {
    taps = src_size < 2 ? 1 : 2;
  } else {
    for (std::int64_t o = 0; o < dst; ++o) {
      const std::int64_t first = o * src / dst;
      const std::int64_t last = ((o + 1) * src - 1) / dst;
      taps = std::max(taps, static_cast<int>(last - first + 1));
    }
  }
  taps = std::min(taps, src_size);

  out.taps = taps;
  out.start.assign(static_cast<std::size_t>(dst_size), 0);
  out.weights.assign(static_cast<std::size_t>(dst_size) * static_cast<std::size_t>(taps), 0);

  std::vector<std::pair<std::int64_t, std::int64_t>> frac;  // (-fraction, index)
  for (std::int64_t o = 0; o < dst; ++o) {
    std::int16_t* row = out.weights.data() + static_cast<std::size_t>(o) * taps;
    if (filter == ScaleFilter::bilinear) {
      const std::int64_t num = (2 * o + 1) * src - dst;
      const std::int64_t p = num <= 0 ? 0 : (num << 14) / (2 * dst);
      std::int64_t i0 = p >> 14;
      std::int64_t f = p & (kOne - 1);
      if (i0 >= src - 1) {
        i0 = src - 1;
        f = 0;
      }
      const int start = static_cast<int>(std::min<std::int64_t>(i0, src - taps));
      out.start[static_cast<std::size_t>(o)] = start;
      const RowBuilder rb{src_size, start, row};
      rb.add(i0, static_cast<std::int32_t>(kOne - f));
      rb.add(i0 + 1, static_cast<std::int32_t>(f));
    } else {
      const std::int64_t first = o * src / dst;
      const std::int64_t last = ((o + 1) * src - 1) / dst;
      const int start = static_cast<int>(std::min<std::int64_t>(first, src - taps));
      out.start[static_cast<std::size_t>(o)] = start;
      const RowBuilder rb{src_size, start, row};
      // Largest remainder: floor every weight, then hand the leftover units
      // one each to the largest fractional parts (lowest i on ties).
      std::int64_t sum = 0;
      frac.clear();
      for (std::int64_t i = first; i <= last; ++i) {
        const std::int64_t overlap =
            std::min((o + 1) * src, (i + 1) * dst) - std::max(o * src, i * dst);
        const std::int64_t scaled = overlap * kOne;
        rb.add(i, static_cast<std::int32_t>(scaled / src));
        sum += scaled / src;
        frac.emplace_back(-(scaled % src), i);
      }
      const auto leftover = static_cast<std::size_t>(kOne - sum);
      std::partial_sort(frac.begin(), frac.begin() + static_cast<std::ptrdiff_t>(leftover),
                        frac.end());
      for (std::size_t k = 0; k < leftover; ++k) rb.add(frac[k].second, 1);
    }
  }
  return Status::ok;
}

Status Scaler::create(PixelFormat format, int src_width, int src_height, int dst_width,
                      int dst_height, ScaleFilter filter, std::unique_ptr<Scaler>& out) {
  if (static_cast<int>(format) >= kPixelFormatCount) return Status::unsupported_format;
  for (const int d : {src_width, src_height, dst_width, dst_height}) {
    if (d < 1 || d > kMaxDimension) return Status::invalid_argument;
  }
  auto impl = std::make_unique<Impl>();
  impl->format = format;
  impl->src_w = src_width;
  impl->src_h = src_height;
  impl->dst_w = dst_width;
  impl->dst_h = dst_height;
  impl->plane_count = pixfmt_info(format).plane_count;
  for (int p = 0; p < impl->plane_count; ++p) {
    PlaneJob& j = impl->planes[p];
    const auto pi = static_cast<std::size_t>(p);
    j.channels = pixfmt_info(format).bytes_per_pixel[pi];
    j.dst_w = plane_width(format, p, dst_width);
    j.shift_y = pixfmt_info(format).shift_y[pi];
    if (Status s = build_scale_filter(filter, plane_width(format, p, src_width), j.dst_w, j.hx);
        s != Status::ok) {
      return s;
    }
    if (Status s = build_scale_filter(filter, plane_height(format, p, src_height),
                                      plane_height(format, p, dst_height), j.hy);
        s != Status::ok) {
      return s;
    }
    const int taps = j.hx.taps;
    const int stride = taps <= 2 ? taps : (taps + 3) & ~3;
    j.hx_weights.assign(static_cast<std::size_t>(j.dst_w) * static_cast<std::size_t>(stride) +
                            kWeightPad,
                        0);
    for (int x = 0; x < j.dst_w; ++x) {
      std::copy_n(j.hx.weights.data() + static_cast<std::size_t>(x) * taps, taps,
                  j.hx_weights.data() + static_cast<std::size_t>(x) * stride);
    }
    j.hview = {taps, stride, j.hx.start.data(), j.hx_weights.data()};
    j.mid_stride = align64(static_cast<std::size_t>(j.dst_w) * j.channels * 2 + 64) / 2;
  }
  out.reset(new Scaler(std::move(impl)));
  return Status::ok;
}

Status Scaler::run(const Frame& src, Frame& dst, RowRange rows) const {
  const Impl& m = *impl_;
  if (src.empty() || dst.empty() || src.format() != m.format || dst.format() != m.format ||
      src.width() != m.src_w || src.height() != m.src_h || dst.width() != m.dst_w ||
      dst.height() != m.dst_h) {
    return Status::invalid_argument;
  }
  RowRange r;
  if (Status s = resolve_rows(dst, rows, r); s != Status::ok) return s;

  // One scratch block per call: row pointers, then a ring of intermediate
  // rows (one slot per vertical tap), sized for the largest plane.
  std::size_t need = 0;
  for (int p = 0; p < m.plane_count; ++p) {
    const PlaneJob& j = m.planes[p];
    const auto taps = static_cast<std::size_t>(j.hy.taps);
    need = std::max(need, align64(taps * sizeof(void*)) + taps * j.mid_stride * 2);
  }
  std::uint8_t* scratch = detail::thread_scratch(0, need);
  if (scratch == nullptr) return Status::out_of_memory;

  const kernels::scale::Table& t = kernels::scale::active_table();
  for (int p = 0; p < m.plane_count; ++p) {
    const PlaneJob& j = m.planes[p];
    const int taps = j.hy.taps;
    const int row_begin = r.begin >> j.shift_y;
    const int row_end = (r.end + (1 << j.shift_y) - 1) >> j.shift_y;
    const Plane sp = src.plane(p);
    const Plane dp = dst.plane(p);
    auto** ptrs = reinterpret_cast<const std::int16_t**>(scratch);
    auto* ring = reinterpret_cast<std::int16_t*>(
        scratch + align64(static_cast<std::size_t>(taps) * sizeof(void*)));
    const auto hscale = t.hscale[j.channels];
    const int count = j.dst_w * j.channels;
    int have_end = 0;  // source rows below this are already in the ring
    for (int o = row_begin; o < row_end; ++o) {
      const int s0 = j.hy.start[static_cast<std::size_t>(o)];
      for (int sr = std::max(s0, have_end); sr < s0 + taps; ++sr) {
        hscale(sp.row(sr), ring + static_cast<std::size_t>(sr % taps) * j.mid_stride, j.dst_w,
               j.hview);
      }
      have_end = s0 + taps;
      for (int k = 0; k < taps; ++k) {
        ptrs[k] = ring + static_cast<std::size_t>((s0 + k) % taps) * j.mid_stride;
      }
      t.vscale(ptrs, j.hy.weights.data() + static_cast<std::size_t>(o) * taps, taps, dp.row(o),
               count);
    }
  }
  return Status::ok;
}

Status scale(const Frame& src, Frame& dst, ScaleFilter filter, RowRange rows) {
  std::unique_ptr<Scaler> scaler;
  if (Status s = Scaler::create(src.format(), src.width(), src.height(), dst.width(),
                                dst.height(), filter, scaler);
      s != Status::ok) {
    return s;
  }
  return scaler->run(src, dst, rows);
}

}  // namespace lanecast
