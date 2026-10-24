// Shared by io.cpp, y4m.cpp, raw.cpp and pnm.cpp. Not part of the public API.
#pragma once

#include <cstddef>
#include <sys/uio.h>

#include <cstdio>
#include <memory>
#include <string>

#include "lanecast/frame.hpp"
#include "lanecast/io.hpp"

namespace lanecast::io_detail {

// File descriptor owner with a small read buffer for header lines and a small
// write buffer for headers, so a frame goes out as one writev. Frame payloads
// bypass both buffers: reads land straight in the padded rows via readv.
// Path "-" maps to stdin or stdout, which are never closed.
class File {
 public:
  File() = default;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File();

  Status open(const std::string& path, bool write);
  // Flushes and closes; io_error if either fails. Safe to call twice.
  Status close();

  // Fills dst[0, n) from the buffer and the descriptor; fewer bytes only at
  // end of file or on error (see failed()).
  std::size_t read(void* dst, std::size_t n);
  // Like read, scattered over iov. Entries are consumed in place.
  std::size_t readv(iovec* iov, int count);
  int get();  // next byte or EOF

  // Appends to the write buffer; flushes to the descriptor when it fills.
  bool write(const void* src, std::size_t n);
  // Writes the pending buffer followed by iov[1, count) in as few syscalls as
  // possible; iov[0] is reserved for the buffer and overwritten. Entries are
  // consumed in place.
  bool writev(iovec* iov, int count);
  bool flush();

  bool failed() const { return failed_; }
  bool is_open() const { return fd_ >= 0; }

 private:
  bool fill();
  bool write_all(iovec* iov, int count);

  int fd_ = -1;
  bool owned_ = false;
  bool failed_ = false;
  std::unique_ptr<char[]> rbuf_;
  std::size_t rpos_ = 0;
  std::size_t rend_ = 0;
  std::unique_ptr<char[]> wbuf_;
  std::size_t wlen_ = 0;
};

// Reads every plane of `frame` (already shaped) with readv straight into the
// padded rows, so bytes are copied once. A short read is parse_error
// (truncated), except zero bytes at the very start when `eof_ok` is set,
// which is end_of_stream.
Status read_planes(File& file, Frame& frame, bool eof_ok);
// Writes any buffered header bytes and every plane with writev.
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
