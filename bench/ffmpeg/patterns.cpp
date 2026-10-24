#include "patterns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "lanecast/convert.hpp"

namespace lanecast::bench::ff {
namespace {

void put(std::uint8_t* px, int r, int g, int b) {
  px[0] = static_cast<std::uint8_t>(std::clamp(r, 0, 255));
  px[1] = static_cast<std::uint8_t>(std::clamp(g, 0, 255));
  px[2] = static_cast<std::uint8_t>(std::clamp(b, 0, 255));
  px[3] = 255;
}

void render_testsrc(Frame& f) {
  static const int bars[8][3] = {{255, 255, 255}, {255, 255, 0}, {0, 255, 255}, {0, 255, 0},
                                 {255, 0, 255},   {255, 0, 0},   {0, 0, 255},   {0, 0, 0}};
  const Plane p = f.plane(0);
  for (int y = 0; y < p.height; ++y) {
    std::uint8_t* row = p.row(y);
    for (int x = 0; x < p.width; ++x) {
      std::uint8_t* px = row + 4 * x;
      if (y < p.height * 2 / 3) {
        const int* c = bars[x * 8 / p.width];
        put(px, c[0], c[1], c[2]);
      } else if (y < p.height * 5 / 6) {
        const int v = x * 255 / (p.width - 1);
        put(px, v, v, v);
      } else {
        const bool on = (((x / 8) + (y / 8)) & 1) != 0;
        put(px, on ? 235 : 20, on ? 235 : 20, on ? 20 : 235);
      }
    }
  }
}

void render_noise(Frame& f) {
  const Plane p = f.plane(0);
  std::uint64_t s = 0x9E3779B97F4A7C15ULL;
  for (int y = 0; y < p.height; ++y) {
    std::uint8_t* row = p.row(y);
    for (int x = 0; x < p.width; ++x) {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      const double u = x * 0.011, v = y * 0.017;
      const int n = static_cast<int>((s >> 40) % 25) - 12;
      int r = static_cast<int>(128 + 90 * std::sin(u + 0.5 * v));
      int g = static_cast<int>(128 + 90 * std::sin(0.7 * u - v + 2.0));
      int b = static_cast<int>(128 + 90 * std::sin(1.3 * v + 0.2 * u + 4.0));
      if (((x / 96) + (y / 64)) % 5 == 0) r = 255 - r;  // hard edges
      put(row + 4 * x, r + n, g + n, b + n);
    }
  }
}

}  // namespace

const char* pattern_name(Pattern p) noexcept {
  return p == Pattern::testsrc ? "testsrc" : "noise";
}

Status make_pattern_frame(Pattern pattern, PixelFormat format, int width, int height, Frame& out) {
  Frame rgba;
  if (const Status st = Frame::allocate(PixelFormat::rgba, width, height, rgba); st != Status::ok) {
    return st;
  }
  if (pattern == Pattern::testsrc) {
    render_testsrc(rgba);
  } else {
    render_noise(rgba);
  }
  if (const Status st = Frame::allocate(format, width, height, out); st != Status::ok) return st;
  return convert(rgba, out, ConvertOptions{});
}

}  // namespace lanecast::bench::ff
