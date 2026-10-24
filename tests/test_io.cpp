// io suite. Owned by the io unit.
// Round trips go through real files in a scratch directory that is removed
// afterwards. Header and truncation cases write their bytes in-test, so no
// fixture files and no external tools are needed.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "lanecast/io.hpp"
#include "test_support.hpp"

namespace lanecast::test {
namespace {

namespace fs = std::filesystem;

// Unique scratch directory, removed with everything in it on destruction.
class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = fs::temp_directory_path() /
            ("lanecast_io_" + std::to_string(stamp) + "_" + std::to_string(counter++));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string file(const std::string& name) const { return (path_ / name).string(); }

 private:
  fs::path path_;
};

void write_bytes(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

VideoInfo make_info(PixelFormat fmt, int w, int h) {
  VideoInfo info;
  info.width = w;
  info.height = h;
  info.format = fmt;
  return info;
}

// Writes `frames` through open_writer, reads them back through open_reader,
// and checks every frame and the end of stream.
void round_trip(const std::string& path, const VideoInfo& info, const ReaderOptions& opts,
                int count, std::uint64_t seed) {
  std::vector<Frame> frames;
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(path, info, std::nullopt, writer));
  for (int i = 0; i < count; ++i) {
    frames.push_back(random_frame(info.format, info.width, info.height, seed + i));
    REQUIRE_OK(writer->write(frames.back()));
  }
  REQUIRE_OK(writer->finish());

  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(path, opts, reader));
  CHECK_EQ(reader->info().width, info.width);
  CHECK_EQ(reader->info().height, info.height);
  CHECK(reader->info().format == info.format);
  Frame got;
  for (int i = 0; i < count; ++i) {
    REQUIRE_OK(reader->read(got));
    const std::string diff = first_difference(frames[static_cast<std::size_t>(i)], got);
    CHECK_MSG(diff.empty(), "frame " + std::to_string(i) + ": " + diff);
  }
  CHECK_EQ(reader->read(got), Status::end_of_stream);
  CHECK_EQ(reader->read(got), Status::end_of_stream);
}

}  // namespace

TEST(container_from_path) {
  CHECK(container_from_path("a.y4m") == Container::y4m);
  CHECK(container_from_path("dir.v1/a.Y4M") == Container::y4m);
  CHECK(container_from_path("a.yuv") == Container::raw);
  CHECK(container_from_path("a.raw") == Container::raw);
  CHECK(container_from_path("a.rgb") == Container::raw);
  CHECK(container_from_path("a.nv12") == Container::raw);
  CHECK(container_from_path("a.ppm") == Container::pnm);
  CHECK(container_from_path("a.pgm") == Container::pnm);
  CHECK(container_from_path("a.pnm") == Container::pnm);
  CHECK(!container_from_path("a.mp4").has_value());
  CHECK(!container_from_path("noext").has_value());
  CHECK(!container_from_path("dir.y4m/noext").has_value());
  CHECK(!container_from_path("-").has_value());
}

TEST(raw_round_trip_all_formats) {
  TempDir dir;
  const PixelFormat formats[] = {PixelFormat::gray8, PixelFormat::yuv420p, PixelFormat::nv12,
                                 PixelFormat::rgb24, PixelFormat::rgba,    PixelFormat::bgra};
  const std::pair<int, int> sizes[] = {{1, 1}, {7, 5}, {64, 3}, {33, 17}};
  for (const PixelFormat fmt : formats) {
    for (const auto& [w, h] : sizes) {
      const VideoInfo info = make_info(fmt, w, h);
      ReaderOptions opts;
      opts.raw = info;
      round_trip(dir.file("a.yuv"), info, opts, 3, 100);
    }
  }
}

TEST(raw_file_is_tightly_packed) {
  TempDir dir;
  const VideoInfo info = make_info(PixelFormat::yuv420p, 5, 3);
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.yuv"), info, std::nullopt, writer));
  REQUIRE_OK(writer->write(random_frame(info.format, 5, 3, 1)));
  REQUIRE_OK(writer->finish());
  // Y 5x3, U and V 3x2 each.
  CHECK_EQ(fs::file_size(dir.file("a.yuv")), static_cast<std::uintmax_t>(5 * 3 + 2 * 3 * 2));
}

TEST(raw_reader_reuses_the_frame_buffer) {
  TempDir dir;
  const VideoInfo info = make_info(PixelFormat::yuv420p, 40, 30);
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.yuv"), info, std::nullopt, writer));
  for (int i = 0; i < 3; ++i) REQUIRE_OK(writer->write(random_frame(info.format, 40, 30, i)));
  REQUIRE_OK(writer->finish());

  ReaderOptions opts;
  opts.raw = info;
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(dir.file("a.yuv"), opts, reader));
  Frame f;
  REQUIRE_OK(reader->read(f));
  const std::uint8_t* first = f.plane(0).data;
  REQUIRE_OK(reader->read(f));
  CHECK(f.plane(0).data == first);
  REQUIRE_OK(reader->read(f));
  CHECK(f.plane(0).data == first);
}

TEST(raw_truncated_frame) {
  TempDir dir;
  // yuv420p 4x4 is 24 bytes: one whole frame plus 10 stray bytes.
  write_bytes(dir.file("a.yuv"), std::string(24 + 10, 'x'));
  ReaderOptions opts;
  opts.raw = make_info(PixelFormat::yuv420p, 4, 4);
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(dir.file("a.yuv"), opts, reader));
  Frame f;
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(reader->read(f), Status::parse_error);
}

TEST(raw_empty_file_is_end_of_stream) {
  TempDir dir;
  write_bytes(dir.file("a.yuv"), "");
  ReaderOptions opts;
  opts.raw = make_info(PixelFormat::gray8, 4, 4);
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(dir.file("a.yuv"), opts, reader));
  Frame f;
  CHECK_EQ(reader->read(f), Status::end_of_stream);
}

TEST(raw_argument_errors) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  ReaderOptions opts;  // no size
  write_bytes(dir.file("a.yuv"), "abcd");
  CHECK_EQ(open_reader(dir.file("a.yuv"), opts, reader), Status::invalid_argument);
  opts.raw = make_info(PixelFormat::gray8, 2, 2);
  CHECK_EQ(open_reader(dir.file("missing.yuv"), opts, reader), Status::io_error);
  CHECK_EQ(open_reader(dir.file("a.bin"), opts, reader), Status::invalid_argument);

  std::unique_ptr<FrameWriter> writer;
  CHECK_EQ(open_writer(dir.file("b.yuv"), make_info(PixelFormat::gray8, 0, 4), std::nullopt,
                       writer),
           Status::invalid_argument);
  REQUIRE_OK(open_writer(dir.file("b.yuv"), make_info(PixelFormat::gray8, 4, 4), std::nullopt,
                         writer));
  const Frame wrong_size = random_frame(PixelFormat::gray8, 5, 4, 1);
  const Frame wrong_format = random_frame(PixelFormat::rgb24, 4, 4, 1);
  CHECK_EQ(writer->write(wrong_size), Status::invalid_argument);
  CHECK_EQ(writer->write(wrong_format), Status::invalid_argument);
  REQUIRE_OK(writer->finish());
  REQUIRE_OK(writer->finish());
  CHECK_EQ(writer->write(random_frame(PixelFormat::gray8, 4, 4, 1)), Status::invalid_argument);
}

}  // namespace lanecast::test
