// The staged, slice-threaded Pipeline.
#include "lanecast/pipeline.hpp"

namespace lanecast {

struct Pipeline::Impl {
  VideoInfo output;
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::~Pipeline() = default;

Status Pipeline::create(std::string_view chain, const VideoInfo& input,
                        const PipelineOptions& opts, std::unique_ptr<Pipeline>& out,
                        std::string* error) {
  (void)chain;
  (void)input;
  (void)opts;
  (void)out;
  (void)error;
  return Status::not_implemented;
}

const VideoInfo& Pipeline::output_info() const noexcept { return impl_->output; }

Status Pipeline::process(const Frame& in, Frame& out) {
  (void)in;
  (void)out;
  return Status::not_implemented;
}

}  // namespace lanecast
