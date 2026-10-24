// Shared types for bench_vs_ffmpeg: one Row per (comparison, pattern, thread
// count), timing helpers and the table and JSON writers.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace lanecast::bench::ff {

struct Row {
  std::string op;       // e.g. "yuv420p>rgb24 bilinear left"
  std::string pattern;  // "testsrc" or "noise"
  std::string sws;      // swscale flags used
  int width = 0;        // source size
  int height = 0;
  int lc_threads = 1;
  int sws_threads = 1;
  double lc_ns = 0.0;  // median ns per frame
  double sws_ns = 0.0;
  double psnr_lc_ref = 0.0;  // dB against the neutral reference
  double psnr_sws_ref = 0.0;
  double psnr_lc_sws = 0.0;  // lanecast output against swscale output
};

struct Options {
  bool json = false;
  bool quick = false;
  std::string filter;  // substring of the op name
  int iters = 31;
  int warmup = 3;
  int threads = 0;  // 0: hardware threads
};

// Median wall time of fn over `iters` runs after `warmup` untimed runs.
inline double median_ns(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> t;
  t.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto a = std::chrono::steady_clock::now();
    fn();
    const auto b = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::nano>(b - a).count());
  }
  std::nth_element(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(t.size() / 2), t.end());
  return t[t.size() / 2];
}

void write_table(const std::vector<Row>& rows, const std::vector<std::string>& notes);
void write_json(const std::vector<Row>& rows, const std::vector<std::string>& notes);

}  // namespace lanecast::bench::ff
