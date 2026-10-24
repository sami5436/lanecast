// Slice-threaded processing: thread pool, filter chains, fused passes.
//
// Performance rules every type here follows:
//   - Slice threading is on by default (threads = 0 means hardware threads).
//   - No per-frame heap allocation in steady state: tables and scratch are
//     built in create(), output frames are reused via Frame::ensure.
//   - Fused passes keep intermediates in per-thread row strips sized for L1
//     and L2 instead of materializing full-frame temporaries.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lanecast/convert.hpp"
#include "lanecast/frame.hpp"
#include "lanecast/scale.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

class ThreadPool {
 public:
  explicit ThreadPool(int threads = 0);  // 0: std::thread::hardware_concurrency
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const noexcept;  // worker count including the calling thread
  // Calls fn(i) for every i in [0, count) on the pool and the calling
  // thread, and returns when all calls are done. Not reentrant.
  void parallel_for(int count, const std::function<void(int)>& fn);

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Splits rows [0, height) into about `slices` ranges whose starts are
// multiples of `alignment` (use pixfmt_row_alignment) and runs fn on the
// pool. Returns the first non-ok status. pool == nullptr runs inline.
Status run_sliced(ThreadPool* pool, int height, int alignment,
                  const std::function<Status(RowRange)>& fn);

// Fused yuv420p or nv12 -> scale -> rgb24, rgba or bgra in one pass per
// row strip: each strip scales the Y and chroma rows it needs into
// thread-local strip buffers and converts them straight into dst.
// Bit-exact contract: the output equals
//   scale(src, tmp_yuv at dst size, filter) followed by convert(tmp_yuv, dst, opts)
// so it can be tested against the unfused ops.
class ScaleConvert {
 public:
  static Status create(PixelFormat src_format, int src_width, int src_height,
                       PixelFormat dst_format, int dst_width, int dst_height,
                       ScaleFilter filter, const ConvertOptions& opts,
                       std::unique_ptr<ScaleConvert>& out);
  ~ScaleConvert();
  ScaleConvert(const ScaleConvert&) = delete;
  ScaleConvert& operator=(const ScaleConvert&) = delete;

  // Safe to call concurrently on disjoint destination row ranges.
  Status run(const Frame& src, Frame& dst, RowRange rows = {}) const;
  // Whole frame, sliced over pool (nullptr: single thread).
  Status run(const Frame& src, Frame& dst, ThreadPool* pool) const;

  struct Impl;

 private:
  explicit ScaleConvert(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Filter chain grammar (ffmpeg -vf subset):
//   chain  := filter ("," filter)*
//   filter := name ["=" arg (":" arg)*]
//   arg    := value | key "=" value      (positional args have an empty key)
// Filters:
//   scale=W:H[:flags=bilinear|area]   W or H = -1 keeps aspect ratio, rounded
//                                     to the format's subsampling multiple
//   format=PIXFMT[:matrix=bt601|bt709][:range=limited|full]
//   blur=3|5                          binomial 3x3 or 5x5
//   eq=brightness=B:contrast=C        B int in [-255, 255], C real in [0, 4]
// Adjacent scale + format stages from yuv to rgb run as one ScaleConvert.
struct FilterSpec {
  std::string name;
  std::vector<std::pair<std::string, std::string>> args;
};

Status parse_filter_chain(std::string_view text, std::vector<FilterSpec>& out,
                          std::string* error = nullptr);

struct PipelineOptions {
  int threads = 0;       // 0: hardware threads; 1: no threading
  int slice_rows = 32;   // target rows per slice (rounded to alignment)
};

class Pipeline {
 public:
  static Status create(std::string_view chain, const VideoInfo& input,
                       const PipelineOptions& opts, std::unique_ptr<Pipeline>& out,
                       std::string* error = nullptr);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  const VideoInfo& output_info() const noexcept;
  // Runs every stage, slice-threaded. out is reused across calls.
  Status process(const Frame& in, Frame& out);

  struct Impl;

 private:
  explicit Pipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace lanecast
