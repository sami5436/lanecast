// Front end: filter table generation (shared by every ISA), Scaler, row
// loop over thread-local intermediate rows (core/scratch.hpp slot 0).
#include "lanecast/scale.hpp"

#include "kernels/scale/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

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

struct Scaler::Impl {};

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
  (void)format;
  (void)src_width;
  (void)src_height;
  (void)dst_width;
  (void)dst_height;
  (void)filter;
  (void)out;
  return Status::not_implemented;
}

Status Scaler::run(const Frame& src, Frame& dst, RowRange rows) const {
  (void)src;
  (void)dst;
  (void)rows;
  return Status::not_implemented;
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
