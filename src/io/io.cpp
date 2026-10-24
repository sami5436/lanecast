// Container routing and shared file helpers. The formats live in y4m.cpp,
// raw.cpp and pnm.cpp.
#include <algorithm>
#include <cctype>

#include "io/io_internal.hpp"
#include "lanecast/io.hpp"

namespace lanecast {
namespace io_detail {
namespace {
constexpr std::size_t kStdioBufferBytes = 256 * 1024;
}  // namespace

File::~File() { (void)close(); }

Status File::open(const std::string& path, bool write) {
  if (path == "-") {
    fp_ = write ? stdout : stdin;
    owned_ = false;
    return Status::ok;
  }
  fp_ = std::fopen(path.c_str(), write ? "wb" : "rb");
  if (fp_ == nullptr) return Status::io_error;
  owned_ = true;
  buffer_ = std::make_unique<char[]>(kStdioBufferBytes);
  std::setvbuf(fp_, buffer_.get(), _IOFBF, kStdioBufferBytes);
  return Status::ok;
}

Status File::close() {
  if (fp_ == nullptr) return Status::ok;
  std::FILE* fp = fp_;
  fp_ = nullptr;
  bool ok = std::fflush(fp) == 0 || !owned_;
  if (owned_) ok = (std::fclose(fp) == 0) && ok;
  buffer_.reset();
  return ok ? Status::ok : Status::io_error;
}

Status read_planes(File& file, Frame& frame, bool eof_ok) {
  bool first = true;
  for (int p = 0; p < frame.plane_count(); ++p) {
    const Plane pl = frame.plane(p);
    const auto bytes = static_cast<std::size_t>(pl.row_bytes());
    for (int y = 0; y < pl.height; ++y) {
      const std::size_t n = file.read(pl.row(y), bytes);
      if (n != bytes) {
        if (file.failed()) return Status::io_error;
        return (first && eof_ok && n == 0) ? Status::end_of_stream : Status::parse_error;
      }
      first = false;
    }
  }
  return Status::ok;
}

Status write_planes(File& file, const Frame& frame) {
  for (int p = 0; p < frame.plane_count(); ++p) {
    const Plane pl = frame.plane(p);
    const auto bytes = static_cast<std::size_t>(pl.row_bytes());
    for (int y = 0; y < pl.height; ++y) {
      if (!file.write(pl.row(y), bytes)) return Status::io_error;
    }
  }
  return Status::ok;
}

bool valid_info(const VideoInfo& info) {
  return info.width >= 1 && info.height >= 1 && info.width <= kMaxDimension &&
         info.height <= kMaxDimension;
}

bool frame_matches(const Frame& frame, const VideoInfo& info) {
  return !frame.empty() && frame.format() == info.format && frame.width() == info.width &&
         frame.height() == info.height;
}

}  // namespace io_detail

namespace {

std::optional<Container> resolve(const std::optional<Container>& explicit_container,
                                 const std::string& path) {
  if (explicit_container) return explicit_container;
  return container_from_path(path);
}

}  // namespace

std::optional<Container> container_from_path(std::string_view path) noexcept {
  const auto dot = path.find_last_of('.');
  const auto slash = path.find_last_of("/\\");
  if (dot == std::string_view::npos || (slash != std::string_view::npos && slash > dot)) {
    return std::nullopt;
  }
  std::string ext(path.substr(dot + 1));
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (ext == "y4m") return Container::y4m;
  if (ext == "yuv" || ext == "raw" || ext == "rgb" || ext == "nv12") return Container::raw;
  if (ext == "ppm" || ext == "pgm" || ext == "pnm") return Container::pnm;
  return std::nullopt;
}

Status open_reader(const std::string& path, const ReaderOptions& opts,
                   std::unique_ptr<FrameReader>& out) {
  const auto container = resolve(opts.container, path);
  if (!container) return Status::invalid_argument;
  switch (*container) {
    case Container::y4m: return Status::not_implemented;
    case Container::raw: return io_detail::open_raw_reader(path, opts.raw, out);
    case Container::pnm: return Status::not_implemented;
  }
  return Status::invalid_argument;
}

Status open_writer(const std::string& path, const VideoInfo& info,
                   std::optional<Container> container, std::unique_ptr<FrameWriter>& out) {
  const auto resolved = resolve(container, path);
  if (!resolved) return Status::invalid_argument;
  switch (*resolved) {
    case Container::y4m: return Status::not_implemented;
    case Container::raw: return io_detail::open_raw_writer(path, info, out);
    case Container::pnm: return Status::not_implemented;
  }
  return Status::invalid_argument;
}

}  // namespace lanecast
