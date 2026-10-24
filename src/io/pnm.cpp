// PNM reader and writer: binary P5 (gray8) and P6 (rgb24), maxval 255.
// Several images may follow each other in one file. P1-P4 and P7 and other
// maxvals are unsupported_format; they are real PNM, just not ours.
#include <charconv>
#include <string>
#include <string_view>

#include "io/io_internal.hpp"

namespace lanecast::io_detail {
namespace {

constexpr std::size_t kMaxWord = 16;

enum class Word { ok, eof, error, io_error };

bool is_space(int c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Reads one whitespace-delimited word, skipping whitespace and '#' comments
// before it, and consumes the single whitespace byte after it. `eof` means
// the file ended before the word started.
Word read_word(File& file, std::string& word) {
  word.clear();
  int c = file.get();
  for (;;) {
    while (is_space(c)) c = file.get();
    if (c != '#') break;
    while (c != '\n' && c != EOF) c = file.get();
  }
  if (c == EOF) return file.failed() ? Word::io_error : Word::eof;
  while (c != EOF && !is_space(c)) {
    if (word.size() >= kMaxWord) return Word::error;
    word.push_back(static_cast<char>(c));
    c = file.get();
  }
  if (c == EOF && file.failed()) return Word::io_error;
  return Word::ok;
}

Status word_status(Word w) {
  return w == Word::io_error ? Status::io_error : Status::parse_error;
}

bool parse_uint(const std::string& s, int& out) {
  if (s.empty()) return false;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && end == s.data() + s.size() && out >= 0;
}

// Reads "P5|P6 width height maxval" and the one whitespace byte after it.
// end_of_stream when the file ends cleanly before the magic.
Status read_header(File& file, PixelFormat& format, int& width, int& height) {
  std::string word;
  if (const Word w = read_word(file, word); w == Word::eof) {
    return Status::end_of_stream;
  } else if (w != Word::ok) {
    return word_status(w);
  }
  if (word == "P5") {
    format = PixelFormat::gray8;
  } else if (word == "P6") {
    format = PixelFormat::rgb24;
  } else if (word.size() == 2 && word[0] == 'P' && word[1] >= '1' && word[1] <= '7') {
    return Status::unsupported_format;
  } else {
    return Status::parse_error;
  }
  int maxval = 0;
  int* const fields[] = {&width, &height, &maxval};
  for (int* field : fields) {
    if (const Word w = read_word(file, word); w != Word::ok) return word_status(w);
    if (!parse_uint(word, *field)) return Status::parse_error;
  }
  if (width < 1 || height < 1 || maxval < 1 || maxval > 65535) return Status::parse_error;
  if (maxval != 255) return Status::unsupported_format;
  if (width > kMaxDimension || height > kMaxDimension) return Status::unsupported_format;
  return Status::ok;
}

class PnmReader final : public FrameReader {
 public:
  Status open(const std::string& path) {
    if (Status s = file_.open(path, false); s != Status::ok) return s;
    Status s = read_header(file_, info_.format, info_.width, info_.height);
    if (s == Status::end_of_stream) return Status::parse_error;  // empty file
    pending_ = s == Status::ok;
    info_.range = ColorRange::full;
    return s;
  }

  const VideoInfo& info() const override { return info_; }

  Status read(Frame& frame) override {
    if (!pending_) {
      PixelFormat format = PixelFormat::gray8;
      int w = 0;
      int h = 0;
      if (Status s = read_header(file_, format, w, h); s != Status::ok) return s;
      // The stream shape was fixed by the first image.
      if (format != info_.format || w != info_.width || h != info_.height) {
        return Status::parse_error;
      }
    }
    pending_ = false;
    if (Status s = frame.ensure(info_.format, info_.width, info_.height); s != Status::ok) {
      return s;
    }
    const Status s = read_planes(file_, frame, false);
    return s == Status::end_of_stream ? Status::parse_error : s;
  }

 private:
  VideoInfo info_;
  File file_;
  bool pending_ = false;
};

class PnmWriter final : public FrameWriter {
 public:
  explicit PnmWriter(const VideoInfo& info) : info_(info) {}
  Status open(const std::string& path) { return file_.open(path, true); }

  Status write(const Frame& frame) override {
    if (!file_.is_open() || !frame_matches(frame, info_)) return Status::invalid_argument;
    const std::string header = std::string(info_.format == PixelFormat::gray8 ? "P5" : "P6") +
                               "\n" + std::to_string(info_.width) + " " +
                               std::to_string(info_.height) + "\n255\n";
    if (!file_.write(header.data(), header.size())) return Status::io_error;
    return write_planes(file_, frame);
  }
  Status finish() override { return file_.close(); }

 private:
  VideoInfo info_;
  File file_;
};

}  // namespace

Status open_pnm_reader(const std::string& path, std::unique_ptr<FrameReader>& out) {
  auto reader = std::make_unique<PnmReader>();
  if (Status s = reader->open(path); s != Status::ok) return s;
  out = std::move(reader);
  return Status::ok;
}

Status open_pnm_writer(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameWriter>& out) {
  if (!valid_info(info)) return Status::invalid_argument;
  if (info.format != PixelFormat::gray8 && info.format != PixelFormat::rgb24) {
    return Status::unsupported_format;
  }
  auto writer = std::make_unique<PnmWriter>(info);
  if (Status s = writer->open(path); s != Status::ok) return s;
  out = std::move(writer);
  return Status::ok;
}

}  // namespace lanecast::io_detail
