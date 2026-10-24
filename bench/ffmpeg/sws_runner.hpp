// libswscale driven directly through sws_scale_frame (no CLI, no IO). The
// source and destination are lanecast Frames, so both libraries read and
// write the same buffers and outputs compare byte for byte.
#pragma once

#include <memory>
#include <string>

#include "lanecast/convert.hpp"
#include "lanecast/frame.hpp"

namespace lanecast::bench::ff {

struct SwsSpec {
  unsigned flags = 0;  // SWS_* mask
  ChromaLocation loc = ChromaLocation::left;
  ColorMatrix matrix = ColorMatrix::bt601;
  ColorRange yuv_range = ColorRange::limited;  // RGB side is always full
};

class SwsRunner {
 public:
  // Binds src and dst (which must outlive the runner) and sets the graph up
  // outside the timed region. Returns null and fills `error` on failure.
  static std::unique_ptr<SwsRunner> create(const SwsSpec& spec, const Frame& src, Frame& dst,
                                           int threads, std::string& error);
  ~SwsRunner();
  SwsRunner(const SwsRunner&) = delete;
  SwsRunner& operator=(const SwsRunner&) = delete;

  bool run();
  int threads() const;  // thread count swscale resolved for this context

  struct Impl;

 private:
  explicit SwsRunner(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace lanecast::bench::ff
