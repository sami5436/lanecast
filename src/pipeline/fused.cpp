// Fused yuv -> scale -> rgb row-strip pass. Uses the scale and convert row
// kernels directly (kernels/scale/kernels.hpp, kernels/convert/kernels.hpp)
// with strip buffers from core/scratch.hpp slots 1 and 2.
#include "lanecast/pipeline.hpp"

namespace lanecast {

struct ScaleConvert::Impl {};

ScaleConvert::ScaleConvert(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ScaleConvert::~ScaleConvert() = default;

Status ScaleConvert::create(PixelFormat src_format, int src_width, int src_height,
                            PixelFormat dst_format, int dst_width, int dst_height,
                            ScaleFilter filter, const ConvertOptions& opts,
                            std::unique_ptr<ScaleConvert>& out) {
  (void)src_format;
  (void)src_width;
  (void)src_height;
  (void)dst_format;
  (void)dst_width;
  (void)dst_height;
  (void)filter;
  (void)opts;
  (void)out;
  return Status::not_implemented;
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
