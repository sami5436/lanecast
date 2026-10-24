// Container routing: open_reader / open_writer pick the y4m, raw or pnm
// implementation (y4m.cpp, raw.cpp, pnm.cpp, added by the io unit).
#include "lanecast/io.hpp"

namespace lanecast {

std::optional<Container> container_from_path(std::string_view path) noexcept {
  (void)path;
  return std::nullopt;
}

Status open_reader(const std::string& path, const ReaderOptions& opts,
                   std::unique_ptr<FrameReader>& out) {
  (void)path;
  (void)opts;
  (void)out;
  return Status::not_implemented;
}

Status open_writer(const std::string& path, const VideoInfo& info,
                   std::optional<Container> container, std::unique_ptr<FrameWriter>& out) {
  (void)path;
  (void)info;
  (void)container;
  (void)out;
  return Status::not_implemented;
}

}  // namespace lanecast
