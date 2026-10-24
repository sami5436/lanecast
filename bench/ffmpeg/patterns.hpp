// Synthetic input frames for bench_vs_ffmpeg, generated in code.
#pragma once

#include "lanecast/frame.hpp"

namespace lanecast::bench::ff {

enum class Pattern { testsrc, noise };

const char* pattern_name(Pattern p) noexcept;

// testsrc: color bars, a luma ramp, hard edges and a checkerboard (like
// ffmpeg's testsrc). noise: smooth low-frequency color fields with
// per-pixel noise and a few edges (natural-looking, stresses filters).
// The pattern is rendered as rgba and converted to `format` with lanecast's
// own converter (left chroma siting, BT.601, limited range for yuv), so both
// libraries read the exact same bytes.
Status make_pattern_frame(Pattern pattern, PixelFormat format, int width, int height, Frame& out);

}  // namespace lanecast::bench::ff
