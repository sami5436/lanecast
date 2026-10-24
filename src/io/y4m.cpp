// YUV4MPEG2 reader and writer.
//
// Reader: W and H are required. F, A, I, C and XCOLORRANGE are interpreted;
// other tags are skipped. C420, C420jpeg, C420mpeg2, C420paldv and no C tag
// give yuv420p, Cmono gives gray8; every other chroma layout (C444, C422,
// high bit depth) is unsupported_format. Interlaced streams (It, Ib, Im) are
// unsupported_format. C420 tags describe chroma siting only: C420jpeg is
// center, C420, C420mpeg2 and C420paldv are left. Range is limited unless
// XCOLORRANGE=FULL. FRAME lines may carry parameters, which are ignored.
#include <charconv>
#include <string_view>
#include <vector>

#include "io/io_internal.hpp"

namespace lanecast::io_detail {
namespace {

constexpr std::string_view kMagic = "YUV4MPEG2";
constexpr std::string_view kFrameMarker = "FRAME";
constexpr std::size_t kMaxLine = 4096;

// Appends bytes up to and excluding '\n' to `line`. parse_error when the line
// is longer than kMaxLine or the file ends before the newline.
Status read_line(File& file, std::string& line) {
  for (;;) {
    const int c = file.get();
    if (c == '\n') return Status::ok;
    if (c == EOF) return file.failed() ? Status::io_error : Status::parse_error;
    if (line.size() >= kMaxLine) return Status::parse_error;
    line.push_back(static_cast<char>(c));
  }
}

bool parse_int(std::string_view s, int& out) {
  if (s.empty()) return false;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && end == s.data() + s.size() && out >= 0;
}

// "n:d" with d != 0, or the "unknown" form 0:0 which returns false.
bool parse_ratio(std::string_view s, int& num, int& den) {
  const auto colon = s.find(':');
  if (colon == std::string_view::npos) return false;
  return parse_int(s.substr(0, colon), num) && parse_int(s.substr(colon + 1), den);
}

Status parse_header(const std::string& line, VideoInfo& info) {
  if (line.compare(0, kMagic.size(), kMagic) != 0) return Status::parse_error;
  info = VideoInfo{};
  info.format = PixelFormat::yuv420p;
  info.range = ColorRange::limited;
  bool have_w = false;
  bool have_h = false;

  std::size_t pos = kMagic.size();
  while (pos < line.size()) {
    if (line[pos] != ' ') return Status::parse_error;
    const std::size_t start = ++pos;
    while (pos < line.size() && line[pos] != ' ') ++pos;
    const std::string_view tok(line.data() + start, pos - start);
    if (tok.empty()) continue;  // repeated spaces
    const std::string_view val = tok.substr(1);
    switch (tok[0]) {
      case 'W':
        if (!parse_int(val, info.width)) return Status::parse_error;
        have_w = true;
        break;
      case 'H':
        if (!parse_int(val, info.height)) return Status::parse_error;
        have_h = true;
        break;
      case 'F': {
        int n = 0;
        int d = 0;
        if (!parse_ratio(val, n, d)) return Status::parse_error;
        if (d == 0 && n != 0) return Status::parse_error;
        if (d != 0) {
          info.fps_num = n;
          info.fps_den = d;
        }
        break;
      }
      case 'A': {
        int n = 0;
        int d = 0;
        if (!parse_ratio(val, n, d)) return Status::parse_error;
        if (d == 0 && n != 0) return Status::parse_error;
        if (n != 0 && d != 0) {
          info.sar_num = n;
          info.sar_den = d;
        }
        break;
      }
      case 'I':
        if (val != "p" && val != "?") return Status::unsupported_format;
        break;
      case 'C':
        if (val == "420" || val == "420mpeg2" || val == "420paldv") {
          info.format = PixelFormat::yuv420p;
          info.chroma_loc = ChromaLocation::left;
        } else if (val == "420jpeg") {
          info.format = PixelFormat::yuv420p;
          info.chroma_loc = ChromaLocation::center;
        } else if (val == "mono") {
          info.format = PixelFormat::gray8;
        } else {
          return Status::unsupported_format;
        }
        break;
      case 'X':
        if (val == "COLORRANGE=FULL") {
          info.range = ColorRange::full;
        } else if (val == "COLORRANGE=LIMITED") {
          info.range = ColorRange::limited;
        }
        break;
      default: break;
    }
  }
  if (!have_w || !have_h) return Status::parse_error;
  if (info.width < 1 || info.height < 1) return Status::parse_error;
  if (info.width > kMaxDimension || info.height > kMaxDimension) return Status::unsupported_format;
  return Status::ok;
}

class Y4mReader final : public FrameReader {
 public:
  Status open(const std::string& path) {
    if (Status s = file_.open(path, false); s != Status::ok) return s;
    std::string line;
    if (Status s = read_line(file_, line); s != Status::ok) return s;
    return parse_header(line, info_);
  }

  const VideoInfo& info() const override { return info_; }

  Status read(Frame& frame) override {
    const int first = file_.get();
    if (first == EOF) return file_.failed() ? Status::io_error : Status::end_of_stream;
    if (first == '\n') return Status::parse_error;
    line_.assign(1, static_cast<char>(first));
    if (Status s = read_line(file_, line_); s != Status::ok) return s;
    if (line_.compare(0, kFrameMarker.size(), kFrameMarker) != 0 ||
        (line_.size() > kFrameMarker.size() && line_[kFrameMarker.size()] != ' ')) {
      return Status::parse_error;
    }
    if (Status s = frame.ensure(info_.format, info_.width, info_.height); s != Status::ok) {
      return s;
    }
    const Status s = read_planes(file_, frame, false);
    return s == Status::end_of_stream ? Status::parse_error : s;
  }

 private:
  VideoInfo info_;
  File file_;
  std::string line_;
};

class Y4mWriter final : public FrameWriter {
 public:
  explicit Y4mWriter(const VideoInfo& info) : info_(info) {}

  Status open(const std::string& path) {
    if (Status s = file_.open(path, true); s != Status::ok) return s;
    const bool full = info_.range == ColorRange::full;
    const bool known_sar = info_.sar_num > 0 && info_.sar_den > 0;
    std::string header = "YUV4MPEG2 W" + std::to_string(info_.width) + " H" +
                         std::to_string(info_.height) + " F" + std::to_string(info_.fps_num) +
                         ":" + std::to_string(info_.fps_den) + " Ip A" +
                         (known_sar ? std::to_string(info_.sar_num) + ":" + std::to_string(info_.sar_den)
                                    : std::string("0:0"));
    if (info_.format == PixelFormat::gray8) {
      header += " Cmono";
    } else {
      header += info_.chroma_loc == ChromaLocation::center ? " C420jpeg" : " C420mpeg2";
    }
    header += full ? " XCOLORRANGE=FULL\n" : " XCOLORRANGE=LIMITED\n";
    return file_.write(header.data(), header.size()) ? Status::ok : Status::io_error;
  }

  Status write(const Frame& frame) override {
    if (!file_.is_open() || !frame_matches(frame, info_)) return Status::invalid_argument;
    if (!file_.write("FRAME\n", kFrameMarker.size() + 1)) return Status::io_error;
    return write_planes(file_, frame);
  }
  Status finish() override { return file_.close(); }

 private:
  VideoInfo info_;
  File file_;
};

}  // namespace

Status open_y4m_reader(const std::string& path, std::unique_ptr<FrameReader>& out) {
  auto reader = std::make_unique<Y4mReader>();
  if (Status s = reader->open(path); s != Status::ok) return s;
  out = std::move(reader);
  return Status::ok;
}

Status open_y4m_writer(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameWriter>& out) {
  if (!valid_info(info)) return Status::invalid_argument;
  if (info.format != PixelFormat::yuv420p && info.format != PixelFormat::gray8) {
    return Status::unsupported_format;
  }
  if (info.fps_num < 0 || info.fps_den < 1 || info.sar_num < 0 || info.sar_den < 0) {
    return Status::invalid_argument;
  }
  auto writer = std::make_unique<Y4mWriter>(info);
  if (Status s = writer->open(path); s != Status::ok) return s;
  out = std::move(writer);
  return Status::ok;
}

}  // namespace lanecast::io_detail
