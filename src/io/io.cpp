// Container routing and shared file helpers. The formats live in y4m.cpp,
// raw.cpp and pnm.cpp.
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <vector>

#include "io/io_internal.hpp"
#include "lanecast/io.hpp"

namespace lanecast {
namespace io_detail {
namespace {
constexpr std::size_t kReadChunk = 512;  // header lines only; payload skips the buffer
constexpr std::size_t kWriteBufferBytes = 64 * 1024;
const int kIovMax = [] {
  const long v = sysconf(_SC_IOV_MAX);
  return static_cast<int>(v > 0 && v < 1024 ? v : 1024);
}();

// Drops `n` consumed bytes from the front of the iovec array.
void advance(iovec*& iov, int& count, std::size_t n) {
  while (count > 0 && n >= iov->iov_len) {
    n -= iov->iov_len;
    ++iov;
    --count;
  }
  if (count > 0 && n > 0) {
    iov->iov_base = static_cast<char*>(iov->iov_base) + n;
    iov->iov_len -= n;
  }
}
}  // namespace

File::~File() { (void)close(); }

Status File::open(const std::string& path, bool write) {
  if (path == "-") {
    fd_ = write ? STDOUT_FILENO : STDIN_FILENO;
    owned_ = false;
  } else {
    int flags = write ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    fd_ = ::open(path.c_str(), flags, 0666);
    if (fd_ < 0) return Status::io_error;
    owned_ = true;
  }
  if (write) {
    wbuf_ = std::make_unique<char[]>(kWriteBufferBytes);
  } else {
    rbuf_ = std::make_unique<char[]>(kReadChunk);
  }
  return Status::ok;
}

Status File::close() {
  if (fd_ < 0) return Status::ok;
  bool ok = flush();
  if (owned_ && ::close(fd_) != 0) ok = false;
  fd_ = -1;
  rbuf_.reset();
  wbuf_.reset();
  return ok ? Status::ok : Status::io_error;
}

bool File::fill() {
  for (;;) {
    const ssize_t n = ::read(fd_, rbuf_.get(), kReadChunk);
    if (n > 0) {
      rpos_ = 0;
      rend_ = static_cast<std::size_t>(n);
      return true;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) failed_ = true;
    return false;
  }
}

int File::get() {
  if (rpos_ == rend_ && !fill()) return EOF;
  return static_cast<unsigned char>(rbuf_[rpos_++]);
}

std::size_t File::read(void* dst, std::size_t n) {
  iovec v{dst, n};
  return readv(&v, 1);
}

std::size_t File::readv(iovec* iov, int count) {
  std::size_t total = 0;
  if (rpos_ < rend_) {
    // Bytes already pulled in while parsing a header come first.
    while (count > 0 && rpos_ < rend_) {
      const std::size_t take = std::min(iov->iov_len, rend_ - rpos_);
      std::memcpy(iov->iov_base, rbuf_.get() + rpos_, take);
      rpos_ += take;
      total += take;
      advance(iov, count, take);
    }
  }
  advance(iov, count, 0);  // skip empty leading entries
  while (count > 0) {
    const ssize_t n = ::readv(fd_, iov, std::min(count, kIovMax));
    if (n < 0) {
      if (errno == EINTR) continue;
      failed_ = true;
      break;
    }
    if (n == 0) break;
    total += static_cast<std::size_t>(n);
    advance(iov, count, static_cast<std::size_t>(n));
  }
  return total;
}

bool File::write_all(iovec* iov, int count) {
  while (count > 0) {
    if (iov->iov_len == 0) {
      ++iov;
      --count;
      continue;
    }
    const ssize_t n = ::writev(fd_, iov, std::min(count, kIovMax));
    if (n < 0) {
      if (errno == EINTR) continue;
      failed_ = true;
      return false;
    }
    advance(iov, count, static_cast<std::size_t>(n));
  }
  return true;
}

bool File::flush() {
  if (wlen_ == 0) return !failed_;
  iovec v{wbuf_.get(), wlen_};
  wlen_ = 0;
  return write_all(&v, 1);
}

bool File::write(const void* src, std::size_t n) {
  if (n > kWriteBufferBytes - wlen_) {
    if (!flush()) return false;
    if (n > kWriteBufferBytes) {
      iovec v{const_cast<void*>(src), n};
      return write_all(&v, 1);
    }
  }
  std::memcpy(wbuf_.get() + wlen_, src, n);
  wlen_ += n;
  return true;
}

bool File::writev(iovec* iov, int count) {
  if (wlen_ == 0) return write_all(iov + 1, count - 1);
  iov[0] = iovec{wbuf_.get(), wlen_};
  wlen_ = 0;
  return write_all(iov, count);
}

namespace {
// Builds one iovec per visible row of every plane into `iov`, after the
// reserved slot File::writev fills with buffered header bytes.
void plane_iovecs(const Frame& frame, std::vector<iovec>& iov) {
  iov.assign(1, iovec{nullptr, 0});
  for (int p = 0; p < frame.plane_count(); ++p) {
    const Plane pl = frame.plane(p);
    const auto bytes = static_cast<std::size_t>(pl.row_bytes());
    for (int y = 0; y < pl.height; ++y) iov.push_back(iovec{pl.row(y), bytes});
  }
}
}  // namespace

Status read_planes(File& file, Frame& frame, bool eof_ok) {
  thread_local std::vector<iovec> iov;
  plane_iovecs(frame, iov);
  std::size_t want = 0;
  for (const iovec& v : iov) want += v.iov_len;
  const std::size_t got = file.readv(iov.data() + 1, static_cast<int>(iov.size()) - 1);
  if (got == want) return Status::ok;
  if (file.failed()) return Status::io_error;
  return (eof_ok && got == 0) ? Status::end_of_stream : Status::parse_error;
}

Status write_planes(File& file, const Frame& frame) {
  thread_local std::vector<iovec> iov;
  plane_iovecs(frame, iov);
  return file.writev(iov.data(), static_cast<int>(iov.size())) ? Status::ok : Status::io_error;
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
    case Container::y4m: return io_detail::open_y4m_reader(path, out);
    case Container::raw: return io_detail::open_raw_reader(path, opts.raw, out);
    case Container::pnm: return io_detail::open_pnm_reader(path, out);
  }
  return Status::invalid_argument;
}

Status open_writer(const std::string& path, const VideoInfo& info,
                   std::optional<Container> container, std::unique_ptr<FrameWriter>& out) {
  const auto resolved = resolve(container, path);
  if (!resolved) return Status::invalid_argument;
  switch (*resolved) {
    case Container::y4m: return io_detail::open_y4m_writer(path, info, out);
    case Container::raw: return io_detail::open_raw_writer(path, info, out);
    case Container::pnm: return io_detail::open_pnm_writer(path, info, out);
  }
  return Status::invalid_argument;
}

}  // namespace lanecast
