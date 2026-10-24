// pipeline suite: thread pool, row slicing, filter chain parsing, Pipeline.
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lanecast/pipeline.hpp"
#include "test_support.hpp"

namespace lanecast {
namespace {

TEST(ThreadPoolRunsEveryIndexOnce) {
  for (const int threads : {1, 2, 3, 8}) {
    ThreadPool pool(threads);
    CHECK_EQ(pool.size(), threads);
    for (const int count : {0, 1, 2, 7, 64, 1000}) {
      std::vector<std::atomic<int>> hits(static_cast<std::size_t>(count));
      pool.parallel_for(count, [&](int i) { hits[static_cast<std::size_t>(i)]++; });
      for (int i = 0; i < count; ++i) {
        CHECK_EQ(hits[static_cast<std::size_t>(i)].load(), 1);
      }
    }
  }
}

TEST(ThreadPoolManyBackToBackJobs) {
  ThreadPool pool(4);
  std::atomic<long> sum{0};
  long expected = 0;
  for (int job = 0; job < 5000; ++job) {
    const int count = 1 + job % 9;
    pool.parallel_for(count, [&](int i) { sum += i + job; });
    for (int i = 0; i < count; ++i) expected += i + job;
  }
  CHECK_EQ(sum.load(), expected);
}

TEST(ThreadPoolZeroThreadsMeansHardware) {
  ThreadPool pool(0);
  CHECK(pool.size() >= 1);
}

TEST(RunSlicedCoversRowsWithAlignedDisjointSlices) {
  for (const int threads : {1, 3, 8}) {
    ThreadPool pool(threads);
    for (const int alignment : {1, 2}) {
      for (const int height : {1, 2, 3, 17, 64, 1080}) {
        for (const int slice_rows : {1, 8, 32, 5000}) {
          std::mutex mu;
          std::vector<RowRange> ranges;
          const Status s = run_sliced(
              &pool, height, alignment,
              [&](RowRange r) {
                std::lock_guard<std::mutex> lock(mu);
                ranges.push_back(r);
                return Status::ok;
              },
              slice_rows);
          CHECK_EQ(s, Status::ok);
          std::vector<int> covered(static_cast<std::size_t>(height), 0);
          for (const RowRange r : ranges) {
            CHECK_EQ(r.begin % alignment, 0);
            CHECK(r.end > r.begin);
            CHECK(r.end == height || r.end % alignment == 0);
            for (int y = r.begin; y < r.end; ++y) covered[static_cast<std::size_t>(y)]++;
          }
          for (int y = 0; y < height; ++y) CHECK_EQ(covered[static_cast<std::size_t>(y)], 1);
        }
      }
    }
  }
}

TEST(RunSlicedUsesAllThreadsOnShortFrames) {
  ThreadPool pool(4);
  std::atomic<int> slices{0};
  CHECK_EQ(run_sliced(&pool, 64, 2,
                      [&](RowRange) {
                        slices++;
                        return Status::ok;
                      },
                      1000),
           Status::ok);
  CHECK(slices.load() >= 4);
}

TEST(RunSlicedWithoutPoolRunsInline) {
  int calls = 0;
  RowRange seen;
  CHECK_EQ(run_sliced(nullptr, 9, 2,
                      [&](RowRange r) {
                        ++calls;
                        seen = r;
                        return Status::ok;
                      }),
           Status::ok);
  CHECK_EQ(calls, 1);
  CHECK_EQ(seen.begin, 0);
  CHECK_EQ(seen.end, 9);
}

TEST(RunSlicedReturnsFirstFailure) {
  ThreadPool pool(4);
  const Status s = run_sliced(&pool, 256, 2, [](RowRange r) {
    return r.begin >= 128 ? Status::io_error : Status::ok;
  });
  CHECK_EQ(s, Status::io_error);
  CHECK_EQ(run_sliced(&pool, 0, 2, [](RowRange) { return Status::ok; }),
           Status::invalid_argument);
}

using Args = std::vector<std::pair<std::string, std::string>>;

Status parse(const std::string& text, std::vector<FilterSpec>& out, std::string* error = nullptr) {
  return parse_filter_chain(text, out, error);
}

TEST(ParseAcceptsDocumentedChains) {
  std::vector<FilterSpec> f;
  REQUIRE_OK(parse("scale=1280:720,format=rgb24", f));
  REQUIRE(f.size() == 2);
  CHECK_EQ(f[0].name, std::string("scale"));
  CHECK(f[0].args == (Args{{"w", "1280"}, {"h", "720"}}));
  CHECK_EQ(f[1].name, std::string("format"));
  CHECK(f[1].args == (Args{{"pix_fmt", "rgb24"}}));

  REQUIRE_OK(parse("scale=640:-2:flags=area", f));
  CHECK(f[0].args == (Args{{"w", "640"}, {"h", "-2"}, {"flags", "area"}}));

  REQUIRE_OK(parse("scale=w=-1:h=360", f));
  CHECK(f[0].args == (Args{{"w", "-1"}, {"h", "360"}}));

  REQUIRE_OK(parse("format=yuv420p:matrix=bt709:range=full:chroma=nearest", f));
  CHECK(f[0].args == (Args{{"pix_fmt", "yuv420p"},
                           {"matrix", "bt709"},
                           {"range", "full"},
                           {"chroma", "nearest"}}));

  REQUIRE_OK(parse(" blur=3 , eq=brightness=10:contrast=1.5 ", f));
  REQUIRE(f.size() == 2);
  CHECK(f[0].args == (Args{{"size", "3"}}));
  CHECK(f[1].args == (Args{{"brightness", "10"}, {"contrast", "1.5"}}));

  REQUIRE_OK(parse("eq", f));
  REQUIRE(f.size() == 1);
  CHECK(f[0].args.empty());
}

TEST(ParseEmptyChainIsPassthrough) {
  std::vector<FilterSpec> f(3);
  REQUIRE_OK(parse("", f));
  CHECK(f.empty());
  REQUIRE_OK(parse("   ", f));
  CHECK(f.empty());
}

TEST(ParseRejectsBadChains) {
  const char* const bad[] = {
      "scale",                    // missing w and h
      "scale=640",                // missing h
      "scale=640:360:100",        // too many positionals
      "scale=640:360:flags=",     // empty value
      "scale=640:360:foo=1",      // unknown option
      "scale=w=640:360",          // positional after keyed
      "scale=640:360:w=1",        // repeated option
      "scale=640::flags=area",    // empty argument
      "format",                   // missing pix_fmt
      "format=rgb24:rgb24",       // too many positionals
      "blur",                     // missing size
      "eq=gamma=2",               // unknown option
      "nosuchfilter=1",           // unknown filter
      "=1",                       // no name
      "scale=1:1,,format=rgb24",  // empty filter
      "scale=1:1,",               // trailing comma
      ",scale=1:1",               // leading comma
  };
  for (const char* text : bad) {
    std::vector<FilterSpec> f;
    std::string error;
    CHECK_MSG(parse(text, f, &error) == Status::parse_error, text);
    CHECK_MSG(!error.empty(), text);
  }
}

VideoInfo info_of(PixelFormat fmt, int w, int h) {
  VideoInfo v;
  v.format = fmt;
  v.width = w;
  v.height = h;
  return v;
}

std::unique_ptr<Pipeline> make_pipeline(const std::string& chain, const VideoInfo& in,
                                        int threads, int slice_rows = 32,
                                        Status expect = Status::ok) {
  PipelineOptions opts;
  opts.threads = threads;
  opts.slice_rows = slice_rows;
  std::unique_ptr<Pipeline> p;
  std::string error;
  const Status s = Pipeline::create(chain, in, opts, p, &error);
  CHECK_MSG(s == expect, chain + ": " + error);
  return p;
}

TEST(PipelineOutputInfo) {
  auto p = make_pipeline("scale=-2:360,format=rgb24", info_of(PixelFormat::yuv420p, 1920, 1080), 1);
  REQUIRE(p != nullptr);
  CHECK_EQ(p->output_info().width, 640);
  CHECK_EQ(p->output_info().height, 360);
  CHECK(p->output_info().format == PixelFormat::rgb24);

  p = make_pipeline("scale=-1:100", info_of(PixelFormat::yuv420p, 150, 60), 1);
  REQUIRE(p != nullptr);
  CHECK_EQ(p->output_info().width, 250);

  p = make_pipeline("scale=w=0:h=-2", info_of(PixelFormat::rgb24, 33, 21), 1);
  REQUIRE(p != nullptr);
  CHECK_EQ(p->output_info().width, 33);
  CHECK_EQ(p->output_info().height, 22);

  p = make_pipeline("scale=-1:-1", info_of(PixelFormat::rgb24, 33, 21), 1);
  REQUIRE(p != nullptr);
  CHECK_EQ(p->output_info().width, 33);
  CHECK_EQ(p->output_info().height, 21);

  p = make_pipeline("format=yuv420p:matrix=bt709:range=full", info_of(PixelFormat::rgb24, 8, 8), 1);
  REQUIRE(p != nullptr);
  CHECK(p->output_info().matrix == ColorMatrix::bt709);
  CHECK(p->output_info().range == ColorRange::full);
}

TEST(PipelineCreateRejectsBadValues) {
  const VideoInfo in = info_of(PixelFormat::yuv420p, 64, 48);
  const char* const bad[] = {
      "scale=abc:10",         "scale=10:20:flags=bicubic", "scale=99999:10",
      "format=nosuchfmt",     "format=rgb24:matrix=bt2020", "format=rgb24:range=wide",
      "format=rgb24:chroma=cubic", "blur=4",               "eq=brightness=300",
      "eq=contrast=5",        "eq=contrast=x",             "nosuchfilter",
      "scale=10",
  };
  for (const char* chain : bad) {
    std::unique_ptr<Pipeline> p;
    std::string error;
    const Status s = Pipeline::create(chain, in, {}, p, &error);
    CHECK_MSG(s != Status::ok && p == nullptr && !error.empty(), chain);
  }
  std::unique_ptr<Pipeline> p;
  CHECK(Pipeline::create("", info_of(PixelFormat::yuv420p, 0, 4), {}, p) != Status::ok);
}

TEST(PipelineEmptyChainCopies) {
  Frame in = test::random_frame(PixelFormat::yuv420p, 13, 7, 5);
  auto p = make_pipeline("", info_of(PixelFormat::yuv420p, 13, 7), 2);
  REQUIRE(p != nullptr);
  Frame out;
  REQUIRE_OK(p->process(in, out));
  CHECK_MSG(test::first_difference(in, out).empty(), test::first_difference(in, out));
}

TEST(PipelineMatchesDirectOps) {
  Frame in = test::random_frame(PixelFormat::yuv420p, 67, 45, 11);
  Frame scaled;
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 33, 20, scaled));
  REQUIRE_OK(scale(in, scaled, ScaleFilter::area, {}, ChromaLocation::left));
  Frame expected;
  REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 33, 20, expected));
  ConvertOptions co;
  co.matrix = ColorMatrix::bt709;
  co.chroma = ChromaMode::nearest;
  REQUIRE_OK(convert(scaled, expected, co));

  for (const int threads : {1, 4}) {
    auto p = make_pipeline("scale=33:20:flags=area,format=rgb24:matrix=bt709:chroma=nearest",
                           info_of(PixelFormat::yuv420p, 67, 45), threads, 2);
    REQUIRE(p != nullptr);
    Frame out;
    REQUIRE_OK(p->process(in, out));
    CHECK_MSG(test::first_difference(expected, out).empty(), test::first_difference(expected, out));
  }
}

TEST(PipelineScaleThenFormatMatchesDirectOps) {
  struct Case {
    PixelFormat fmt;
    const char* chain;
    ScaleFilter filter;
    ChromaLocation convert_loc;  // the format stage's siting (the input's is left)
    ColorMatrix matrix;
    PixelFormat out;
  };
  const Case cases[] = {
      {PixelFormat::yuv420p, "scale=71:50,format=rgb24", ScaleFilter::bilinear,
       ChromaLocation::left, ColorMatrix::bt601, PixelFormat::rgb24},
      {PixelFormat::nv12, "scale=33:21:flags=area,format=bgra:matrix=bt709", ScaleFilter::area,
       ChromaLocation::left, ColorMatrix::bt709, PixelFormat::bgra},
      // Different sitings between the stages: runs unfused, same result.
      {PixelFormat::yuv420p, "scale=45:60,format=rgba:chroma_loc=center", ScaleFilter::bilinear,
       ChromaLocation::center, ColorMatrix::bt601, PixelFormat::rgba},
  };
  for (const Case& c : cases) {
    const VideoInfo vi = info_of(c.fmt, 67, 45);
    Frame in = test::random_frame(c.fmt, 67, 45, 21);
    auto p = make_pipeline(c.chain, vi, 1);
    REQUIRE(p != nullptr);
    const int w = p->output_info().width;
    const int h = p->output_info().height;
    Frame mid;
    REQUIRE_OK(Frame::allocate(c.fmt, w, h, mid));
    REQUIRE_OK(scale(in, mid, c.filter, {}, vi.chroma_loc));
    Frame expected;
    REQUIRE_OK(Frame::allocate(c.out, w, h, expected));
    ConvertOptions co;
    co.matrix = c.matrix;
    co.chroma_loc = c.convert_loc;
    REQUIRE_OK(convert(mid, expected, co));
    for (const int threads : {1, 3}) {
      auto q = make_pipeline(c.chain, vi, threads, 2);
      REQUIRE(q != nullptr);
      Frame out;
      REQUIRE_OK(q->process(in, out));
      CHECK_MSG(test::first_difference(expected, out).empty(),
                std::string(c.chain) + ": " + test::first_difference(expected, out));
    }
  }
}

TEST(PipelineHonorsInputChromaSiting) {
  // Center-sited input must reach both the scaler and the converter; with the
  // left default anywhere in the chain the bytes differ.
  Frame in = test::random_frame(PixelFormat::yuv420p, 67, 45, 12);
  VideoInfo info = info_of(PixelFormat::yuv420p, 67, 45);
  info.chroma_loc = ChromaLocation::center;

  const auto direct = [&](ChromaLocation loc) {
    Frame scaled;
    Frame rgb;
    REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 34, 20, scaled));
    REQUIRE_OK(scale(in, scaled, ScaleFilter::bilinear, {}, loc));
    REQUIRE_OK(Frame::allocate(PixelFormat::rgb24, 34, 20, rgb));
    ConvertOptions co;
    co.chroma_loc = loc;
    REQUIRE_OK(convert(scaled, rgb, co));
    return rgb;
  };
  const Frame expected = direct(ChromaLocation::center);
  CHECK(!test::first_difference(expected, direct(ChromaLocation::left)).empty());

  for (const int threads : {1, 4}) {
    auto p = make_pipeline("scale=34:20,format=rgb24", info, threads, 2);
    REQUIRE(p != nullptr);
    Frame out;
    REQUIRE_OK(p->process(in, out));
    CHECK_MSG(test::first_difference(expected, out).empty(), test::first_difference(expected, out));
  }
}

TEST(PipelineFormatChromaLocOption) {
  auto p = make_pipeline("format=yuv420p:chroma_loc=center", info_of(PixelFormat::rgb24, 8, 8), 1);
  REQUIRE(p != nullptr);
  CHECK(p->output_info().chroma_loc == ChromaLocation::center);
  p = make_pipeline("format=yuv420p", info_of(PixelFormat::rgb24, 8, 8), 1);
  REQUIRE(p != nullptr);
  CHECK(p->output_info().chroma_loc == ChromaLocation::left);
  make_pipeline("format=rgb24:chroma_loc=top", info_of(PixelFormat::yuv420p, 8, 8), 1,
                32, Status::invalid_argument);
}

TEST(PipelineThreadedEqualsSingleThread) {
  struct Case {
    PixelFormat fmt;
    const char* chain;
  };
  const Case cases[] = {
      {PixelFormat::yuv420p, "scale=61:37"},
      {PixelFormat::yuv420p, "scale=29:17:flags=area,format=rgb24"},
      {PixelFormat::yuv420p, "format=rgba:matrix=bt709:range=full,scale=-2:25"},
      {PixelFormat::yuv420p, "format=nv12,scale=40:31,format=bgra:chroma=nearest"},
      {PixelFormat::nv12, "scale=45:51,format=rgb24"},
      {PixelFormat::rgb24, "scale=23:44:flags=area,format=yuv420p"},
      {PixelFormat::rgba, "format=nv12:matrix=bt709,scale=17:9"},
      {PixelFormat::gray8, "scale=50:41,format=rgb24"},
      {PixelFormat::yuv420p, "scale=100:30,format=rgb24,scale=51:15,format=yuv420p"},
      {PixelFormat::yuv420p, ""},
      {PixelFormat::rgb24, "blur=3,eq=brightness=20:contrast=1.25,blur=5"},
  };
  const std::pair<int, int> sizes[] = {{1, 1}, {2, 3}, {7, 5}, {64, 33}, {97, 71}, {130, 50}};
  for (const Case& c : cases) {
    for (const auto& [w, h] : sizes) {
      const VideoInfo vi = info_of(c.fmt, w, h);
      Frame in = test::random_frame(c.fmt, w, h, static_cast<std::uint64_t>(w * 131 + h));
      std::unique_ptr<Pipeline> ref = make_pipeline(c.chain, vi, 1);
      if (ref == nullptr) continue;
      Frame expected;
      const Status rs = ref->process(in, expected);
      if (rs == Status::not_implemented) continue;  // filter ops not built yet
      REQUIRE_OK(rs);
      for (const auto& [threads, slice_rows] : {std::pair{2, 32}, {3, 2}, {8, 1}, {0, 4}}) {
        auto p = make_pipeline(c.chain, vi, threads, slice_rows);
        REQUIRE(p != nullptr);
        Frame out = test::random_frame(p->output_info().format, p->output_info().width,
                                       p->output_info().height, 99);  // stale contents
        for (int run = 0; run < 2; ++run) {
          REQUIRE_OK(p->process(in, out));
          const std::string d = test::first_difference(expected, out);
          CHECK_MSG(d.empty(), std::string(c.chain) + " " + std::to_string(w) + "x" +
                                   std::to_string(h) + " threads " + std::to_string(threads) +
                                   ": " + d);
        }
      }
    }
  }
}

TEST(PipelineProcessRejectsMismatchedInput) {
  auto p = make_pipeline("format=rgb24", info_of(PixelFormat::yuv420p, 16, 16), 1);
  REQUIRE(p != nullptr);
  Frame wrong_size = test::random_frame(PixelFormat::yuv420p, 18, 16, 1);
  Frame wrong_format = test::random_frame(PixelFormat::nv12, 16, 16, 1);
  Frame out;
  CHECK_EQ(p->process(wrong_size, out), Status::invalid_argument);
  CHECK_EQ(p->process(wrong_format, out), Status::invalid_argument);
  Frame empty;
  CHECK_EQ(p->process(empty, out), Status::invalid_argument);
}

// Deterministic frames, with optional failures at a given frame index.
class FakeReader final : public FrameReader {
 public:
  FakeReader(const VideoInfo& info, int frames) : info_(info), frames_(frames) {}
  int fail_at = -1;      // read returns parse_error for this frame
  int wrong_size_at = -1;  // returns a frame of another size (process rejects it)
  const VideoInfo& info() const override { return info_; }
  Status read(Frame& frame) override {
    if (next_ >= frames_) return Status::end_of_stream;
    const int i = next_++;
    if (i == fail_at) return Status::parse_error;
    const int w = i == wrong_size_at ? info_.width + 2 : info_.width;
    Frame f = test::random_frame(info_.format, w, info_.height, 1000 + static_cast<std::uint64_t>(i));
    frame = std::move(f);  // a real reader reuses the buffer; content is what matters here
    return Status::ok;
  }

 private:
  VideoInfo info_;
  int frames_;
  int next_ = 0;
};

class RecordingWriter final : public FrameWriter {
 public:
  int fail_at = -1;  // write returns io_error for this frame
  bool slow = false;
  std::string bytes;
  Status write(const Frame& frame) override {
    if (slow) std::this_thread::sleep_for(std::chrono::microseconds(300));
    if (count_++ == fail_at) return Status::io_error;
    for (int p = 0; p < frame.plane_count(); ++p) {
      const Plane pl = frame.plane(p);
      for (int y = 0; y < pl.height; ++y) {
        bytes.append(reinterpret_cast<const char*>(pl.row(y)), static_cast<std::size_t>(pl.row_bytes()));
      }
    }
    return Status::ok;
  }
  Status finish() override { return Status::ok; }

 private:
  int count_ = 0;
};

struct StreamRun {
  StreamResult result;
  std::string bytes;
};

StreamRun run_fake(const char* chain, int threads, int frames, bool overlap, int depth,
                   int read_fail, int write_fail, int wrong_size = -1, long long max_frames = -1,
                   bool slow = false) {
  const VideoInfo vi = info_of(PixelFormat::yuv420p, 38, 22);
  FakeReader reader(vi, frames);
  reader.fail_at = read_fail;
  reader.wrong_size_at = wrong_size;
  auto p = make_pipeline(chain, vi, threads);
  RecordingWriter writer;
  writer.fail_at = write_fail;
  writer.slow = slow;
  StreamOptions so;
  so.overlap = overlap;
  so.depth = depth;
  so.max_frames = max_frames;
  StreamRun r;
  r.result = run_stream(reader, *p, writer, so);
  r.bytes = std::move(writer.bytes);
  return r;
}

void check_same_as_serial(const char* chain, int threads, int frames, int depth, int read_fail,
                          int write_fail, int wrong_size = -1, long long max_frames = -1,
                          bool slow = false) {
  const StreamRun a = run_fake(chain, threads, frames, false, 1, read_fail, write_fail, wrong_size, max_frames);
  const StreamRun b = run_fake(chain, threads, frames, true, depth, read_fail, write_fail, wrong_size, max_frames, slow);
  const std::string tag = std::string(chain) + " frames " + std::to_string(frames) + " depth " +
                          std::to_string(depth);
  CHECK_MSG(a.bytes == b.bytes, tag + ": output bytes differ");
  CHECK_MSG(a.result.status == b.result.status, tag + ": status differs");
  CHECK_MSG(a.result.stage == b.result.stage, tag + ": stage differs");
  CHECK_MSG(a.result.frames == b.result.frames, tag + ": frame count differs");
  CHECK_MSG(a.result.failed_frame == b.result.failed_frame, tag + ": failed frame differs");
}

TEST(StreamOverlapMatchesSerial) {
  for (const char* chain : {"", "scale=21:13,format=rgb24"}) {
    for (const int threads : {1, 0}) {
      for (const int frames : {0, 1, 2, 3, 7, 25}) {
        for (const int depth : {2, 3, 6}) {
          check_same_as_serial(chain, threads, frames, depth, -1, -1);
        }
      }
    }
  }
  const StreamRun full = run_fake("format=rgb24", 0, 9, true, 3, -1, -1);
  CHECK_EQ(full.result.status, Status::ok);
  CHECK_EQ(full.result.frames, 9);
  CHECK_EQ(full.bytes.size(), static_cast<std::size_t>(9 * 38 * 22 * 3));
}

TEST(StreamStopsAtMaxFrames) {
  for (const long long cap : {1LL, 4LL, 40LL}) {
    check_same_as_serial("format=rgb24", 0, 11, 3, -1, -1, -1, cap);
  }
  CHECK_EQ(run_fake("", 1, 11, true, 3, -1, -1, -1, 4).result.frames, 4);
}

TEST(StreamEarlyWriteErrorMatchesSerial) {
  for (const int fail : {0, 1, 5, 19}) {
    for (const int depth : {2, 3, 8}) {
      check_same_as_serial("scale=21:13,format=rgb24", 0, 20, depth, -1, fail);
      check_same_as_serial("scale=21:13,format=rgb24", 1, 20, depth, -1, fail, -1, -1, true);
    }
  }
  const StreamRun r = run_fake("format=rgb24", 0, 200, true, 3, -1, 3);
  CHECK_EQ(r.result.status, Status::io_error);
  CHECK(r.result.stage == StreamResult::Stage::write);
  CHECK_EQ(r.result.failed_frame, 3);
  CHECK_EQ(r.result.frames, 3);
}

TEST(StreamReadAndProcessErrorsMatchSerial) {
  for (const int depth : {2, 3, 8}) {
    check_same_as_serial("format=rgb24", 0, 20, depth, 6, -1);
    check_same_as_serial("format=rgb24", 0, 20, depth, 0, -1);
    check_same_as_serial("format=rgb24", 0, 20, depth, -1, -1, 4);
    // The write error is the earlier frame, so it wins over a later read error.
    check_same_as_serial("format=rgb24", 0, 20, depth, 12, 2);
    // A read error at an earlier frame than a write error wins.
    check_same_as_serial("format=rgb24", 0, 20, depth, 2, 9);
  }
  const StreamRun r = run_fake("format=rgb24", 0, 20, true, 3, 6, -1);
  CHECK_EQ(r.result.status, Status::parse_error);
  CHECK(r.result.stage == StreamResult::Stage::read);
  CHECK_EQ(r.result.frames, 6);
  const StreamRun q = run_fake("format=rgb24", 0, 20, true, 3, -1, -1, 4);
  CHECK_EQ(q.result.status, Status::invalid_argument);
  CHECK(q.result.stage == StreamResult::Stage::process);
  CHECK_EQ(q.result.frames, 4);
}

}  // namespace
}  // namespace lanecast
