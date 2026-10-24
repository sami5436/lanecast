// Front end: filter table generation (shared by every ISA), Scaler, row
// loop over thread-local intermediate rows (core/scratch.hpp slot 0).
#include "lanecast/scale.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

#include "core/scratch.hpp"
#include "kernels/scale/kernels.hpp"

namespace lanecast {
namespace {

constexpr std::int64_t kOne = 1 << 14;
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

namespace {

// ROUNDED_DIV of swscale: round to nearest, ties away from zero.
std::int64_t rounded_div(std::int64_t a, std::int64_t b) {
  return (a >= 0 ? a + (b >> 1) : a - (b >> 1)) / b;
}

}  // namespace

Status build_scale_filter(ScaleFilter filter, int src_size, int dst_size, ScaleFilterTable& out,
                          int src_pos, int dst_pos) {
  if (src_size < 1 || dst_size < 1 || src_size > kMaxDimension || dst_size > kMaxDimension ||
      src_pos < 0 || src_pos > 256 || dst_pos < 0 || dst_pos > 256) {
    return Status::invalid_argument;
  }
  const std::int64_t src = src_size;
  const std::int64_t dst = dst_size;

  // Ideal coefficients, as swscale's initFilter computes them. Distances are
  // in 2^-17 source pixels, coefficients in units of fone (1.0 == fone).
  const std::int64_t inc = ((src << 16) + (dst >> 1)) / dst;
  const bool down = inc > (1 << 16);
  const std::int64_t factor = (filter == ScaleFilter::area && down) ? 1 : 2;
  const bool box = factor == 1;
  const int width = static_cast<int>(
      std::clamp<std::int64_t>(down ? 1 + (factor * src + dst - 1) / dst : 1 + factor, 1, src));
  int log2_ratio = 0;
  while (log2_ratio < 8 && (src / dst) >> (log2_ratio + 1) != 0) ++log2_ratio;
  const std::int64_t fone = std::int64_t{1} << (54 - log2_ratio);
  const std::int64_t cutoff = fone / 500;

  const auto n = static_cast<std::size_t>(dst_size);
  const auto w = static_cast<std::size_t>(width);
  std::vector<std::int64_t> coef(n * w);
  std::vector<std::int64_t> pos(n);
  std::int64_t center = ((dst_pos * inc) >> 7) - ((std::int64_t{src_pos} * 0x10000) >> 7);
  for (std::size_t o = 0; o < n; ++o) {
    std::int64_t xx = (center - (width - 2) * (std::int64_t{1} << 16)) / (1 << 17);
    pos[o] = xx;
    for (std::size_t j = 0; j < w; ++j, ++xx) {
      std::int64_t d = std::abs(xx * (1 << 17) - center) << 13;
      if (down) d = d * dst / src;
      std::int64_t c;
      if (box) {
        const std::int64_t t = (d - (std::int64_t{1} << 29)) * inc;
        if (t < -(std::int64_t{1} << 45)) {
          c = std::int64_t{1} << 46;
        } else if (t < (std::int64_t{1} << 45)) {
          c = (std::int64_t{1} << 45) - t;
        } else {
          c = 0;
        }
        c *= fone >> 46;
      } else {
        c = std::max<std::int64_t>((std::int64_t{1} << 30) - d, 0) * (fone >> 30);
      }
      coef[o * w + j] = c;
    }
    center += 2 * inc;
  }

  // Shift tiny leading taps out and find the common width that keeps every
  // row's mass except at most 0.2 percent per side.
  std::size_t kept = 1;
  for (std::size_t o = 0; o < n; ++o) {
    std::int64_t* row = coef.data() + o * w;
    std::int64_t cut = 0;
    for (std::size_t j = 0; j < w; ++j) {
      cut += std::abs(row[0]);
      if (cut > cutoff) break;
      std::copy(row + 1, row + w, row);
      row[w - 1] = 0;
      ++pos[o];
    }
    cut = 0;
    std::size_t used = w;
    for (std::size_t j = w - 1; j > 0; --j) {
      cut += std::abs(row[j]);
      if (cut > cutoff) break;
      --used;
    }
    kept = std::max(kept, used);
  }

  // Fold taps outside the source into the edge pixel, normalize to Q14 by
  // error diffusion, then drop columns that are zero in every row.
  std::vector<std::int32_t> first(n);
  std::vector<std::int16_t> dense(n * kept);
  std::vector<std::int64_t> acc(kept);
  int taps = 1;
  std::vector<std::pair<int, int>> span(n);
  for (std::size_t o = 0; o < n; ++o) {
    const int s0 = static_cast<int>(
        std::clamp<std::int64_t>(pos[o], 0, src - static_cast<std::int64_t>(kept)));
    first[o] = s0;
    std::fill(acc.begin(), acc.end(), 0);
    for (std::size_t j = 0; j < kept; ++j) {
      const auto idx = std::clamp<std::int64_t>(pos[o] + static_cast<std::int64_t>(j), 0, src - 1);
      acc[static_cast<std::size_t>(idx - s0)] += coef[o * w + j];
    }
    std::int64_t total = 0;
    for (const std::int64_t c : acc) total += c;
    const std::int64_t norm = std::max<std::int64_t>((total + kOne / 2) / kOne, 1);
    std::int16_t* row = dense.data() + o * kept;
    std::int64_t err = 0;
    int sum = 0;
    std::size_t big = 0;
    for (std::size_t j = 0; j < kept; ++j) {
      const std::int64_t v = acc[j] + err;
      const std::int64_t iv = rounded_div(v, norm);
      err = v - iv * norm;
      row[j] = static_cast<std::int16_t>(iv);
      sum += static_cast<int>(iv);
      if (row[j] > row[big]) big = j;
    }
    row[big] = static_cast<std::int16_t>(row[big] + (kOne - sum));  // exact Q14 sum
    int lo = 0;
    int hi = static_cast<int>(kept) - 1;
    while (row[lo] == 0) ++lo;
    while (row[hi] == 0) --hi;
    span[o] = {lo, hi};
    taps = std::max(taps, hi - lo + 1);
  }
  taps = std::min(taps, src_size);

  out.taps = taps;
  out.start.assign(n, 0);
  out.weights.assign(n * static_cast<std::size_t>(taps), 0);
  for (std::size_t o = 0; o < n; ++o) {
    const int lo = first[o] + span[o].first;
    const int start = std::min(lo, src_size - taps);
    out.start[o] = start;
    const std::int16_t* row = dense.data() + o * kept;
    for (int j = span[o].first; j <= span[o].second; ++j) {
      out.weights[o * static_cast<std::size_t>(taps) +
                  static_cast<std::size_t>(first[o] + j - start)] = row[j];
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
