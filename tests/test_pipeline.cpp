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

}  // namespace
}  // namespace lanecast
