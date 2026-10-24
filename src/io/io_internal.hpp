// Shared by io.cpp, y4m.cpp, raw.cpp and pnm.cpp. Not part of the public API.
#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>

#include "lanecast/frame.hpp"
#include "lanecast/io.hpp"

namespace lanecast::io_detail {

// FILE* owner. Path "-" maps to stdin or stdout, which are never closed.
class File {
 public:
  File() = default;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File();

  Status open(const std::string& path, bool write);
  // Flushes and closes; io_error if either fails. Safe to call twice.
  Status close();

  std::size_t read(void* dst, std::size_t n) { return std::fread(dst, 1, n, fp_); }
  bool write(const void* src, std::size_t n) { return std::fwrite(src, 1, n, fp_) == n; }
  int get() { return std::getc(fp_); }
  bool failed() const { return std::ferror(fp_) != 0; }
  bool is_open() const { return fp_ != nullptr; }

 private:
  std::FILE* fp_ = nullptr;
  bool owned_ = false;
  std::unique_ptr<char[]> buffer_;
};

// Reads every plane of `frame` (already shaped) with one fread per row, so
// bytes land straight in the padded buffer. A short read is parse_error
// (truncated), except zero bytes at the very first row when `eof_ok` is set,
// which is end_of_stream.
Status read_planes(File& file, Frame& frame, bool eof_ok);
Status write_planes(File& file, const Frame& frame);

bool valid_info(const VideoInfo& info);
bool frame_matches(const Frame& frame, const VideoInfo& info);

Status open_y4m_reader(const std::string& path, std::unique_ptr<FrameReader>& out);
Status open_raw_reader(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameReader>& out);
Status open_pnm_reader(const std::string& path, std::unique_ptr<FrameReader>& out);
Status open_y4m_writer(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameWriter>& out);
Status open_raw_writer(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameWriter>& out);
Status open_pnm_writer(const std::string& path, const VideoInfo& info,
                       std::unique_ptr<FrameWriter>& out);

}  // namespace lanecast::io_detail
