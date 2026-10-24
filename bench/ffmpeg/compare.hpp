// One head-to-head comparison and the engine that times and scores it.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lanecast/pipeline.hpp"
#include "patterns.hpp"
#include "sws_runner.hpp"

namespace lanecast::bench::ff {

// Binds lanecast to a source and destination frame and returns the row-range
// op, or an empty function when this build cannot run it (skipped, noted).
using LanecastBinder = std::function<std::function<Status(RowRange)>(const Frame&, Frame&)>;

struct Comparison {
  std::string op;
  PixelFormat src_format = PixelFormat::yuv420p;
  PixelFormat dst_format = PixelFormat::rgb24;
  int src_width = 0, src_height = 0;
  int dst_width = 0, dst_height = 0;
  SwsSpec sws;  // flags and siting under test; the reference reuses siting
  LanecastBinder lanecast;
};

// Flags of the neutral reference: the highest quality swscale configuration.
unsigned reference_flags();

std::string flag_names(unsigned flags);

// Times lanecast and swscale on `pattern` with 1 and `threads` threads and
// scores both against the reference. `pool` has `threads` workers.
void run_comparison(const Comparison& c, Pattern pattern, const Options& opts, int threads,
                    ThreadPool& pool, std::vector<Row>& rows, std::vector<std::string>& notes);

void add_convert_comparisons(std::vector<Comparison>& out);

}  // namespace lanecast::bench::ff
