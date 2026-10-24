// Pixel formats and color descriptions.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace lanecast {

// Names match ffmpeg's -pix_fmt names. All formats are 8 bits per sample.
enum class PixelFormat : std::uint8_t {
  gray8,    // 1 plane: Y
  yuv420p,  // 3 planes: Y, U (w/2 x h/2, rounded up), V (same)
  nv12,     // 2 planes: Y, UV interleaved (U first) at w/2 x h/2, rounded up
  rgb24,    // 1 plane: R G B
  rgba,     // 1 plane: R G B A
  bgra,     // 1 plane: B G R A
};
inline constexpr int kPixelFormatCount = 6;
inline constexpr int kMaxPlanes = 4;

enum class ColorMatrix : std::uint8_t { bt601, bt709 };
enum class ColorRange : std::uint8_t {
  limited,  // Y in [16, 235], U and V in [16, 240] ("tv", "mpeg")
  full,     // all samples in [0, 255] ("pc", "jpeg")
};

struct PixelFormatInfo {
  std::string_view name;
  int plane_count;
  // Bytes per pixel of each plane (nv12 plane 1 is 2: one U and one V).
  std::array<int, kMaxPlanes> bytes_per_pixel;
  // Plane size is ceil(width / 2^shift_x) by ceil(height / 2^shift_y).
  std::array<int, kMaxPlanes> shift_x;
  std::array<int, kMaxPlanes> shift_y;
  bool is_yuv;     // gray8 counts as yuv (luma only)
  bool has_alpha;  // alpha is byte 3 of each pixel
};

inline constexpr std::array<PixelFormatInfo, kPixelFormatCount> kPixelFormats = {{
    {"gray8", 1, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, true, false},
    {"yuv420p", 3, {1, 1, 1, 0}, {0, 1, 1, 0}, {0, 1, 1, 0}, true, false},
    {"nv12", 2, {1, 2, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, true, false},
    {"rgb24", 1, {3, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, false, false},
    {"rgba", 1, {4, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, false, true},
    {"bgra", 1, {4, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, false, true},
}};

constexpr const PixelFormatInfo& pixfmt_info(PixelFormat f) noexcept {
  return kPixelFormats[static_cast<std::size_t>(f)];
}
constexpr std::string_view pixfmt_name(PixelFormat f) noexcept { return pixfmt_info(f).name; }

// Plane width in pixels, height in rows, and visible bytes per row.
constexpr int plane_width(PixelFormat f, int plane, int width) noexcept {
  const int s = pixfmt_info(f).shift_x[static_cast<std::size_t>(plane)];
  return (width + (1 << s) - 1) >> s;
}
constexpr int plane_height(PixelFormat f, int plane, int height) noexcept {
  const int s = pixfmt_info(f).shift_y[static_cast<std::size_t>(plane)];
  return (height + (1 << s) - 1) >> s;
}
constexpr int plane_row_bytes(PixelFormat f, int plane, int width) noexcept {
  return plane_width(f, plane, width) *
         pixfmt_info(f).bytes_per_pixel[static_cast<std::size_t>(plane)];
}
// Row ranges passed to ops must start on a multiple of this (2 for 4:2:0).
constexpr int pixfmt_row_alignment(PixelFormat f) noexcept {
  return 1 << pixfmt_info(f).shift_y[1 < pixfmt_info(f).plane_count ? 1 : 0];
}

// Accepts ffmpeg names ("gray" is accepted as gray8).
std::optional<PixelFormat> parse_pixfmt(std::string_view name) noexcept;
std::optional<ColorMatrix> parse_color_matrix(std::string_view name) noexcept;  // bt601, bt709
std::optional<ColorRange> parse_color_range(std::string_view name) noexcept;    // limited|tv, full|pc

}  // namespace lanecast
