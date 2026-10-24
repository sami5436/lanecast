// Owning, aligned, padded frames.
//
// Memory guarantee (every Frame allocated by lanecast, relied on by all
// SIMD kernels):
//   1. Every plane's data pointer is 64-byte aligned (kFrameAlignment).
//   2. Every plane's stride is a multiple of 64 and at least
//      row_bytes + kRowPadding, where row_bytes = plane_row_bytes(...).
//   3. Therefore each row has at least 64 bytes of addressable slack after
//      its last visible byte, inside that same row. A kernel may READ and
//      WRITE any byte in [row, row + row_bytes + kRowPadding) of a row it
//      owns. This lets a vector loop finish the tail with one full-width
//      load and store (up to 64 bytes past the end) instead of a scalar tail.
//   4. Padding contents are unspecified. No result may depend on them; the
//      tests poison padding with random bytes to enforce this.
//   5. Kernels never write outside the rows they were asked for, so slices
//      on different threads never write the same bytes. They may read
//      neighbor source rows (filter taps, bilinear chroma), which is safe
//      because sources are read-only during a call.
// Freshly allocated frames are zero-filled.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "lanecast/pixfmt.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

inline constexpr std::size_t kFrameAlignment = 64;
inline constexpr std::size_t kRowPadding = 64;
inline constexpr int kMaxDimension = 16384;  // width and height limit

// Rows [begin, end) of a frame, in luma (full-resolution) rows.
// end < 0 means "to the last row". For 4:2:0 formats begin must be even and
// end must be even or equal to the height; chroma rows [begin/2, ceil(end/2))
// are processed. Ops that take a RowRange are safe to call concurrently on
// disjoint ranges of the same destination.
struct RowRange {
  int begin = 0;
  int end = -1;
};

// Shape of a video stream, shared by io and pipeline.
struct VideoInfo {
  int width = 0;
  int height = 0;
  PixelFormat format = PixelFormat::yuv420p;
  ColorRange range = ColorRange::limited;
  ColorMatrix matrix = ColorMatrix::bt601;
  ChromaLocation chroma_loc = ChromaLocation::left;
  int fps_num = 25;
  int fps_den = 1;
  int sar_num = 1;
  int sar_den = 1;
};

// Non-owning view of one plane.
struct Plane {
  std::uint8_t* data = nullptr;
  std::ptrdiff_t stride = 0;  // bytes between rows, multiple of 64
  int width = 0;              // pixels
  int height = 0;             // rows
  int bytes_per_pixel = 0;
  int row_bytes() const noexcept { return width * bytes_per_pixel; }
  std::uint8_t* row(int y) const noexcept { return data + static_cast<std::ptrdiff_t>(y) * stride; }
};

class Frame {
 public:
  Frame() = default;
  Frame(Frame&&) noexcept = default;
  Frame& operator=(Frame&&) noexcept = default;
  Frame(const Frame&) = delete;
  Frame& operator=(const Frame&) = delete;
  ~Frame() = default;

  // Allocates a zero-filled frame. width and height in [1, kMaxDimension].
  static Status allocate(PixelFormat format, int width, int height, Frame& out);

  // Reuses the existing buffer when format and size already match, so
  // steady-state pipelines never allocate per frame.
  Status ensure(PixelFormat format, int width, int height);

  Status clone(Frame& out) const;  // deep copy of visible bytes

  bool empty() const noexcept { return buffer_ == nullptr; }
  PixelFormat format() const noexcept { return format_; }
  int width() const noexcept { return width_; }
  int height() const noexcept { return height_; }
  int plane_count() const noexcept { return pixfmt_info(format_).plane_count; }
  Plane plane(int i) const noexcept { return planes_[static_cast<std::size_t>(i)]; }
  bool same_shape(const Frame& other) const noexcept {
    return format_ == other.format_ && width_ == other.width_ && height_ == other.height_;
  }

 private:
  struct AlignedFree {
    void operator()(std::uint8_t* p) const noexcept;
  };
  std::unique_ptr<std::uint8_t[], AlignedFree> buffer_;
  std::size_t capacity_ = 0;
  PixelFormat format_ = PixelFormat::gray8;
  int width_ = 0;
  int height_ = 0;
  Plane planes_[kMaxPlanes] = {};
};

// Resolves end < 0 and validates alignment and bounds for this frame.
Status resolve_rows(const Frame& f, RowRange in, RowRange& out) noexcept;

}  // namespace lanecast
