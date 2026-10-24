#include "sws_runner.hpp"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace lanecast::bench::ff {
namespace {

AVPixelFormat av_format(PixelFormat f) {
  switch (f) {
    case PixelFormat::gray8: return AV_PIX_FMT_GRAY8;
    case PixelFormat::yuv420p: return AV_PIX_FMT_YUV420P;
    case PixelFormat::nv12: return AV_PIX_FMT_NV12;
    case PixelFormat::rgb24: return AV_PIX_FMT_RGB24;
    case PixelFormat::rgba: return AV_PIX_FMT_RGBA;
    case PixelFormat::bgra: return AV_PIX_FMT_BGRA;
  }
  return AV_PIX_FMT_NONE;
}

// Describes `f` to swscale, borrowing its buffers.
void describe(AVFrame* av, const Frame& f, const SwsSpec& spec) {
  av->width = f.width();
  av->height = f.height();
  av->format = av_format(f.format());
  for (int p = 0; p < f.plane_count(); ++p) {
    av->data[p] = f.plane(p).data;
    av->linesize[p] = static_cast<int>(f.plane(p).stride);
  }
  if (pixfmt_info(f.format()).is_yuv) {
    av->color_range = spec.yuv_range == ColorRange::full ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    av->colorspace = spec.matrix == ColorMatrix::bt709 ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;
    av->chroma_location =
        spec.loc == ChromaLocation::left ? AVCHROMA_LOC_LEFT : AVCHROMA_LOC_CENTER;
  } else {
    av->color_range = AVCOL_RANGE_JPEG;
    av->colorspace = AVCOL_SPC_RGB;
  }
}

}  // namespace

struct SwsRunner::Impl {
  SwsContext* ctx = nullptr;
  AVFrame* src = nullptr;
  AVFrame* dst = nullptr;
  ~Impl() {
    av_frame_free(&src);
    av_frame_free(&dst);
    sws_free_context(&ctx);
  }
};

SwsRunner::SwsRunner(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SwsRunner::~SwsRunner() = default;

std::unique_ptr<SwsRunner> SwsRunner::create(const SwsSpec& spec, const Frame& src, Frame& dst,
                                             int threads, std::string& error) {
  auto impl = std::make_unique<Impl>();
  impl->ctx = sws_alloc_context();
  impl->src = av_frame_alloc();
  impl->dst = av_frame_alloc();
  if (!impl->ctx || !impl->src || !impl->dst) {
    error = "out of memory";
    return nullptr;
  }
  impl->ctx->flags = spec.flags;
  impl->ctx->threads = threads;
  describe(impl->src, src, spec);
  describe(impl->dst, dst, spec);
  if (sws_frame_setup(impl->ctx, impl->dst, impl->src) < 0) {
    error = "sws_frame_setup failed";
    return nullptr;
  }
  return std::unique_ptr<SwsRunner>(new SwsRunner(std::move(impl)));
}

bool SwsRunner::run() { return sws_scale_frame(impl_->ctx, impl_->dst, impl_->src) >= 0; }

int SwsRunner::threads() const { return impl_->ctx->threads; }

}  // namespace lanecast::bench::ff
