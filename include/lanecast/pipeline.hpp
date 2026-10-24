// Slice-threaded processing: thread pool, filter chains, fused passes.
//
// Performance rules every type here follows:
//   - Slice threading is on by default (threads = 0 means hardware threads).
//   - No per-frame heap allocation in steady state: tables and scratch are
//     built in create(), output frames are reused via Frame::ensure.
//   - Fused passes keep intermediates in per-thread row strips sized for L1
//     and L2 instead of materializing full-frame temporaries.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "lanecast/convert.hpp"
#include "lanecast/frame.hpp"
#include "lanecast/scale.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

// Non-owning reference to a callable, valid for the duration of the call it
// is passed to. Unlike std::function it never allocates.
template <class Sig>
class FunctionRef;
template <class R, class... Args>
class FunctionRef<R(Args...)> {
 public:
  template <class F, class = std::enable_if_t<!std::is_same_v<std::decay_t<F>, FunctionRef>>>
  FunctionRef(F&& f) noexcept  // NOLINT: implicit by design
      : obj_(const_cast<void*>(static_cast<const void*>(std::addressof(f)))),
        call_([](void* o, Args... args) -> R {
          return (*static_cast<std::add_pointer_t<F>>(o))(std::forward<Args>(args)...);
        }) {}
  R operator()(Args... args) const { return call_(obj_, std::forward<Args>(args)...); }

 private:
  void* obj_;
  R (*call_)(void*, Args...);
};

// Persistent worker threads. Workers spin briefly after a job and then park
// on a condition variable, so back-to-back stages see no wake-up latency and
// idle pools cost nothing.
class ThreadPool {
 public:
  explicit ThreadPool(int threads = 0);  // 0: std::thread::hardware_concurrency
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const noexcept;  // worker count including the calling thread
  // Calls fn(i) for every i in [0, count) on the pool and the calling
  // thread, and returns when all calls are done. No allocation. Not
  // reentrant and not safe to call from two threads at once.
  void parallel_for(int count, FunctionRef<void(int)> fn);

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Splits rows [0, height) into disjoint ranges whose starts are multiples of
// `alignment` (use pixfmt_row_alignment), about `slice_rows` rows each but
// never fewer slices than pool threads when the height allows, and runs fn
// on the pool. Returns the first non-ok status. pool == nullptr runs inline.
Status run_sliced(ThreadPool* pool, int height, int alignment,
                  FunctionRef<Status(RowRange)> fn, int slice_rows = 32);

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
//   arg    := value | key "=" value
// Whitespace around names, keys and values is ignored. Positional values
// are stored under their option name in FilterSpec::args, so
// "scale=640:360" and "scale=w=640:h=360" parse identically. Positional
// values must come before keyed ones. Unknown filters, unknown or repeated
// options, empty values and missing required options are parse errors.
// Filters (option names in order of position):
//   scale=W:H[:flags=bilinear|area]   W or H: a positive size, 0 for the input
//                                     size, or -N to keep the aspect ratio with
//                                     the result rounded to a multiple of N (-1
//                                     exact, -2 even). Both -N: input size.
//                                     Default flags: bilinear.
//   format=PIXFMT[:matrix=bt601|bt709][:range=limited|full][:chroma=nearest|bilinear]
//                                     matrix and range default to the input's
//   blur=3|5                          binomial 3x3 or 5x5 (option "size")
//   eq=brightness=B:contrast=C        B int in [-255, 255], C real in [0, 4]
// Adjacent scale + format stages from yuv to rgb may run as one ScaleConvert.
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
