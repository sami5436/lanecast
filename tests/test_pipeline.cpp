// pipeline suite: thread pool, row slicing, filter chain parsing, Pipeline.
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
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

}  // namespace
}  // namespace lanecast
