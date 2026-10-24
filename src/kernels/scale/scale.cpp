// Front end: filter table generation (shared by every ISA), Scaler, row
// loop over thread-local intermediate rows (core/scratch.hpp slot 0).
#include "lanecast/scale.hpp"

#include "kernels/scale/kernels.hpp"

namespace lanecast {

struct Scaler::Impl {};

Scaler::Scaler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Scaler::~Scaler() = default;

Status build_scale_filter(ScaleFilter filter, int src_size, int dst_size, ScaleFilterTable& out) {
  (void)filter;
  (void)src_size;
  (void)dst_size;
  (void)out;
  return Status::not_implemented;
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
