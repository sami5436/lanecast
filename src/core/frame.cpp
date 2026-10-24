#include "lanecast/frame.hpp"

#include <cstring>
#include <new>

namespace lanecast {
namespace {

constexpr std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) & ~(a - 1); }

std::size_t plane_stride(PixelFormat f, int plane, int width) {
  return align_up(static_cast<std::size_t>(plane_row_bytes(f, plane, width)) + kRowPadding,
                  kFrameAlignment);
}

std::size_t frame_bytes(PixelFormat f, int width, int height) {
  std::size_t total = 0;
  for (int p = 0; p < pixfmt_info(f).plane_count; ++p) {
    total += plane_stride(f, p, width) * static_cast<std::size_t>(plane_height(f, p, height));
  }
  return total;
}

bool valid_dims(int width, int height) {
  return width >= 1 && height >= 1 && width <= kMaxDimension && height <= kMaxDimension;
}

}  // namespace

void Frame::AlignedFree::operator()(std::uint8_t* p) const noexcept {
  ::operator delete[](p, std::align_val_t{kFrameAlignment});
}

Status Frame::allocate(PixelFormat format, int width, int height, Frame& out) {
  if (!valid_dims(width, height)) return Status::invalid_argument;
  const std::size_t bytes = frame_bytes(format, width, height);
  auto* raw = static_cast<std::uint8_t*>(
      ::operator new[](bytes, std::align_val_t{kFrameAlignment}, std::nothrow));
  if (raw == nullptr) return Status::out_of_memory;
  std::memset(raw, 0, bytes);

  Frame f;
  f.buffer_.reset(raw);
  f.capacity_ = bytes;
  f.format_ = format;
  f.width_ = width;
  f.height_ = height;
  std::uint8_t* cursor = raw;
  for (int p = 0; p < pixfmt_info(format).plane_count; ++p) {
    Plane& pl = f.planes_[p];
    pl.data = cursor;
    pl.stride = static_cast<std::ptrdiff_t>(plane_stride(format, p, width));
    pl.width = plane_width(format, p, width);
    pl.height = plane_height(format, p, height);
    pl.bytes_per_pixel = pixfmt_info(format).bytes_per_pixel[static_cast<std::size_t>(p)];
    cursor += static_cast<std::size_t>(pl.stride) * static_cast<std::size_t>(pl.height);
  }
  out = std::move(f);
  return Status::ok;
}

Status Frame::ensure(PixelFormat format, int width, int height) {
  if (!empty() && format_ == format && width_ == width && height_ == height) return Status::ok;
  return allocate(format, width, height, *this);
}

Status Frame::clone(Frame& out) const {
  if (empty()) return Status::invalid_argument;
  Frame copy;
  if (Status s = allocate(format_, width_, height_, copy); s != Status::ok) return s;
  for (int p = 0; p < plane_count(); ++p) {
    const Plane src = planes_[p];
    const Plane dst = copy.planes_[p];
    for (int y = 0; y < src.height; ++y) {
      std::memcpy(dst.row(y), src.row(y), static_cast<std::size_t>(src.row_bytes()));
    }
  }
  out = std::move(copy);
  return Status::ok;
}

Status resolve_rows(const Frame& f, RowRange in, RowRange& out) noexcept {
  if (f.empty()) return Status::invalid_argument;
  const int end = in.end < 0 ? f.height() : in.end;
  const int align = pixfmt_row_alignment(f.format());
  if (in.begin < 0 || in.begin > end || end > f.height()) return Status::invalid_argument;
  if (in.begin % align != 0) return Status::invalid_argument;
  if (end % align != 0 && end != f.height()) return Status::invalid_argument;
  out = RowRange{in.begin, end};
  return Status::ok;
}

}  // namespace lanecast
