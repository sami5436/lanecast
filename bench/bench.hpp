// Shared types for lanecast_bench: a Case builds a ready-to-run Instance for
// a frame size; main.cpp times every instance under every supported ISA tier
// and thread count. Cases use only the public API, so the benchmark measures
// what users call, including the dispatch.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lanecast/frame.hpp"
#include "lanecast/pipeline.hpp"

namespace lanecast::bench {

struct Size {
  int width;
  int height;
};

// One prepared operation. run() processes one whole frame. When the case is
// sliceable, run(pool) splits destination rows across the pool; otherwise
// the pool is ignored and only the single-thread configuration is timed.
struct Instance {
  std::function<Status(ThreadPool*)> run;
  std::size_t bytes = 0;  // visible bytes read plus written per run
  bool sliceable = true;
};

struct Case {
  std::string name;
  std::function<Instance(Size)> make;
};

// Visible bytes of a frame (all planes, without row padding).
inline std::size_t frame_bytes(const Frame& f) {
  std::size_t total = 0;
  for (int p = 0; p < f.plane_count(); ++p) {
    total += static_cast<std::size_t>(f.plane(p).row_bytes()) *
             static_cast<std::size_t>(f.plane(p).height);
  }
  return total;
}

// Allocated frame with deterministic pseudo-random visible bytes.
inline std::shared_ptr<Frame> make_frame(PixelFormat fmt, int width, int height,
                                         std::uint64_t seed) {
  auto f = std::make_shared<Frame>();
  if (Frame::allocate(fmt, width, height, *f) != Status::ok) return f;
  std::uint64_t x = seed * 0x9E3779B97F4A7C15ULL + 1;
  for (int p = 0; p < f->plane_count(); ++p) {
    const Plane pl = f->plane(p);
    for (int y = 0; y < pl.height; ++y) {
      std::uint8_t* row = pl.row(y);
      for (int i = 0; i < pl.row_bytes(); ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        row[i] = static_cast<std::uint8_t>(x >> 24);
      }
    }
  }
  return f;
}

// Wraps a row-range op over `height` destination rows as a pool-aware run.
inline std::function<Status(ThreadPool*)> sliced(int height, int alignment,
                                                  std::function<Status(RowRange)> fn) {
  return [height, alignment, fn = std::move(fn)](ThreadPool* pool) {
    return run_sliced(pool, height, alignment, fn);
  };
}

inline int row_alignment(PixelFormat a, PixelFormat b) {
  return std::max(pixfmt_row_alignment(a), pixfmt_row_alignment(b));
}

void register_convert(std::vector<Case>& out);
void register_scale(std::vector<Case>& out);
void register_filter(std::vector<Case>& out);
void register_metrics(std::vector<Case>& out);

}  // namespace lanecast::bench
