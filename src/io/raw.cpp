// Raw planar/packed files: frames back to back, rows tightly packed.
#include "io/io_internal.hpp"

namespace lanecast::io_detail {
namespace {

class RawReader final : public FrameReader {
 public:
  explicit RawReader(const VideoInfo& info) : info_(info) {}
  Status open(const std::string& path) { return file_.open(path, false); }

  const VideoInfo& info() const override { return info_; }

  Status read(Frame& frame) override {
    if (Status s = frame.ensure(info_.format, info_.width, info_.height); s != Status::ok) {
      return s;
    }
    return read_planes(file_, frame, true);
  }

 private:
  VideoInfo info_;
  File file_;
};

class RawWriter final : public FrameWriter {
 public:
  explicit RawWriter(const VideoInfo& info) : info_(info) {}
  Status open(const std::string& path) { return file_.open(path, true); }

  Status write(const Frame& frame) override {
    if (!file_.is_open() || !frame_matches(frame, info_)) return Status::invalid_argument;
    return write_planes(file_, frame);
  }
  Status finish() override { return file_.close(); }

 private:
  VideoInfo info_;
  File file_;
};

}  // namespace

Status open_raw_reader(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameReader>& out) {
  if (!valid_info(info)) return Status::invalid_argument;
  auto reader = std::make_unique<RawReader>(info);
  if (Status s = reader->open(path); s != Status::ok) return s;
  out = std::move(reader);
  return Status::ok;
}

Status open_raw_writer(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameWriter>& out) {
  if (!valid_info(info)) return Status::invalid_argument;
  auto writer = std::make_unique<RawWriter>(info);
  if (Status s = writer->open(path); s != Status::ok) return s;
  out = std::move(writer);
  return Status::ok;
}

}  // namespace lanecast::io_detail
