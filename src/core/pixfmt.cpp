#include "lanecast/pixfmt.hpp"

namespace lanecast {

std::optional<PixelFormat> parse_pixfmt(std::string_view name) noexcept {
  if (name == "gray") return PixelFormat::gray8;
  for (int i = 0; i < kPixelFormatCount; ++i) {
    const auto f = static_cast<PixelFormat>(i);
    if (pixfmt_name(f) == name) return f;
  }
  return std::nullopt;
}

std::optional<ColorMatrix> parse_color_matrix(std::string_view name) noexcept {
  if (name == "bt601") return ColorMatrix::bt601;
  if (name == "bt709") return ColorMatrix::bt709;
  return std::nullopt;
}

std::optional<ColorRange> parse_color_range(std::string_view name) noexcept {
  if (name == "limited" || name == "tv") return ColorRange::limited;
  if (name == "full" || name == "pc") return ColorRange::full;
  return std::nullopt;
}

}  // namespace lanecast
