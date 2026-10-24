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


// ---- y4m ----

TEST(y4m_round_trip) {
  TempDir dir;
  const std::pair<int, int> sizes[] = {{1, 1}, {2, 2}, {7, 5}, {64, 48}, {33, 17}};
  for (const auto& [w, h] : sizes) {
    for (const PixelFormat fmt : {PixelFormat::yuv420p, PixelFormat::gray8}) {
      VideoInfo info = make_info(fmt, w, h);
      round_trip(dir.file("a.y4m"), info, ReaderOptions{}, 3, 7);
    }
  }
}

TEST(y4m_round_trip_keeps_stream_info) {
  TempDir dir;
  VideoInfo info = make_info(PixelFormat::yuv420p, 16, 8);
  info.fps_num = 30000;
  info.fps_den = 1001;
  info.sar_num = 4;
  info.sar_den = 3;
  info.range = ColorRange::full;
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.y4m"), info, std::nullopt, writer));
  REQUIRE_OK(writer->write(random_frame(info.format, 16, 8, 1)));
  REQUIRE_OK(writer->finish());

  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(dir.file("a.y4m"), ReaderOptions{}, reader));
  const VideoInfo& got = reader->info();
  CHECK_EQ(got.fps_num, 30000);
  CHECK_EQ(got.fps_den, 1001);
  CHECK_EQ(got.sar_num, 4);
  CHECK_EQ(got.sar_den, 3);
  CHECK(got.range == ColorRange::full);
}

TEST(y4m_writer_header_text) {
  TempDir dir;
  VideoInfo info = make_info(PixelFormat::yuv420p, 4, 2);
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.y4m"), info, std::nullopt, writer));
  REQUIRE_OK(writer->finish());
  std::ifstream in(dir.file("a.y4m"), std::ios::binary);
  std::string line;
  std::getline(in, line);
  CHECK_EQ(line, std::string("YUV4MPEG2 W4 H2 F25:1 Ip A1:1 C420mpeg2 XCOLORRANGE=LIMITED"));
}

namespace {

// A minimal valid y4m with the given header tail, one 2x2 frame of `payload`.
std::string y4m_bytes(const std::string& header, const std::string& frames) {
  return "YUV4MPEG2 " + header + "\n" + frames;
}

const std::string kFrame2x2 = std::string("FRAME\n") + std::string(6, '\x55');

// Opens `bytes` as a y4m file and returns the open status.
Status open_y4m_bytes(const TempDir& dir, const std::string& bytes,
                      std::unique_ptr<FrameReader>& reader) {
  write_bytes(dir.file("t.y4m"), bytes);
  return open_reader(dir.file("t.y4m"), ReaderOptions{}, reader);
}

}  // namespace

TEST(y4m_chroma_tags) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  for (const char* tag : {"", " C420", " C420jpeg", " C420mpeg2", " C420paldv"}) {
    REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes(std::string("W2 H2") + tag, kFrame2x2), reader));
    CHECK(reader->info().format == PixelFormat::yuv420p);
  }
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2 Cmono", "FRAME\n\x01\x02\x03\x04"), reader));
  CHECK(reader->info().format == PixelFormat::gray8);
  Frame f;
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(f.plane(0).row(1)[1], 4);
}

TEST(y4m_unsupported_layouts) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  for (const char* tag : {" C444", " C422", " C411", " C420p10", " C444alpha", " Cmono12"}) {
    CHECK_EQ(open_y4m_bytes(dir, y4m_bytes(std::string("W2 H2") + tag, kFrame2x2), reader),
             Status::unsupported_format);
  }
  for (const char* tag : {" It", " Ib", " Im"}) {
    CHECK_EQ(open_y4m_bytes(dir, y4m_bytes(std::string("W2 H2") + tag, kFrame2x2), reader),
             Status::unsupported_format);
  }
  CHECK_EQ(open_y4m_bytes(dir, y4m_bytes("W99999 H2", ""), reader), Status::unsupported_format);
}

TEST(y4m_header_fields) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_y4m_bytes(
      dir, y4m_bytes("W2 H2 F24000:1001 Ip A32:27 C420jpeg XYSCSS=420JPEG XFOO=bar", kFrame2x2),
      reader));
  CHECK_EQ(reader->info().width, 2);
  CHECK_EQ(reader->info().fps_num, 24000);
  CHECK_EQ(reader->info().fps_den, 1001);
  CHECK_EQ(reader->info().sar_num, 32);
  CHECK_EQ(reader->info().sar_den, 27);
  CHECK(reader->info().range == ColorRange::full);  // C420jpeg implies full

  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2 C420jpeg XCOLORRANGE=LIMITED", kFrame2x2),
                            reader));
  CHECK(reader->info().range == ColorRange::limited);
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2 XCOLORRANGE=FULL", kFrame2x2), reader));
  CHECK(reader->info().range == ColorRange::full);

  // Missing F and A fall back to 25:1 and 1:1; unknown ratios 0:0 do too.
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", kFrame2x2), reader));
  CHECK_EQ(reader->info().fps_num, 25);
  CHECK_EQ(reader->info().sar_den, 1);
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2 F0:0 A0:0 I?", kFrame2x2), reader));
  CHECK_EQ(reader->info().fps_num, 25);
  CHECK_EQ(reader->info().sar_num, 1);
}

TEST(y4m_frame_parameters_are_ignored) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_y4m_bytes(
      dir, y4m_bytes("W2 H2", "FRAME Ixyz\n" + std::string(6, '\x07') + "FRAME\n" +
                                  std::string(6, '\x09')),
      reader));
  Frame f;
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(f.plane(0).row(0)[0], 7);
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(f.plane(2).row(0)[0], 9);
  CHECK_EQ(reader->read(f), Status::end_of_stream);
}

TEST(y4m_malformed_headers) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  CHECK_EQ(open_y4m_bytes(dir, "", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG1 W2 H2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2W2 H2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 H2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W0 H2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 Wabc H2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W-2 H2\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W2 H2 F30\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W2 H2 F30:0\n", reader), Status::parse_error);
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W2 H2", reader), Status::parse_error);  // no newline
  CHECK_EQ(open_y4m_bytes(dir, "YUV4MPEG2 W2 H2 X" + std::string(5000, 'a') + "\n", reader),
           Status::parse_error);
}

TEST(y4m_truncated_and_bad_frames) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  Frame f;
  // Header only: clean end of stream.
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", ""), reader));
  CHECK_EQ(reader->read(f), Status::end_of_stream);

  // Pixel data cut short, at several points.
  for (const std::size_t keep : {0u, 3u, 5u}) {
    REQUIRE_OK(open_y4m_bytes(
        dir, y4m_bytes("W2 H2", kFrame2x2.substr(0, 6 + keep)), reader));
    CHECK_EQ(reader->read(f), Status::parse_error);
  }
  // FRAME line cut short or garbled.
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", "FRAM"), reader));
  CHECK_EQ(reader->read(f), Status::parse_error);
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", "FRAME"), reader));
  CHECK_EQ(reader->read(f), Status::parse_error);
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", "FRAMEX\n" + std::string(6, 'a')), reader));
  CHECK_EQ(reader->read(f), Status::parse_error);
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", "\n" + std::string(6, 'a')), reader));
  CHECK_EQ(reader->read(f), Status::parse_error);
  // Second frame cut short after a good first one.
  REQUIRE_OK(open_y4m_bytes(dir, y4m_bytes("W2 H2", kFrame2x2 + kFrame2x2.substr(0, 9)), reader));
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(reader->read(f), Status::parse_error);
}

TEST(y4m_writer_rejects_unsupported) {
  TempDir dir;
  std::unique_ptr<FrameWriter> writer;
  for (const PixelFormat fmt : {PixelFormat::nv12, PixelFormat::rgb24, PixelFormat::rgba,
                                PixelFormat::bgra}) {
    CHECK_EQ(open_writer(dir.file("a.y4m"), make_info(fmt, 4, 4), std::nullopt, writer),
             Status::unsupported_format);
  }
  VideoInfo bad = make_info(PixelFormat::yuv420p, 4, 4);
  bad.fps_den = 0;
  CHECK_EQ(open_writer(dir.file("a.y4m"), bad, std::nullopt, writer), Status::invalid_argument);
}

TEST(y4m_reader_reuses_the_frame_buffer) {
  TempDir dir;
  const VideoInfo info = make_info(PixelFormat::yuv420p, 40, 30);
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.y4m"), info, std::nullopt, writer));
  for (int i = 0; i < 3; ++i) REQUIRE_OK(writer->write(random_frame(info.format, 40, 30, i)));
  REQUIRE_OK(writer->finish());
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(dir.file("a.y4m"), ReaderOptions{}, reader));
  Frame f;
  REQUIRE_OK(reader->read(f));
  const std::uint8_t* first = f.plane(1).data;
  REQUIRE_OK(reader->read(f));
  REQUIRE_OK(reader->read(f));
  CHECK(f.plane(1).data == first);
}


// ---- pnm ----

TEST(pnm_round_trip) {
  TempDir dir;
  const std::pair<int, int> sizes[] = {{1, 1}, {3, 2}, {17, 9}, {64, 5}};
  for (const auto& [w, h] : sizes) {
    round_trip(dir.file("a.ppm"), make_info(PixelFormat::rgb24, w, h), ReaderOptions{}, 1, 3);
    round_trip(dir.file("a.pgm"), make_info(PixelFormat::gray8, w, h), ReaderOptions{}, 1, 4);
  }
}

TEST(pnm_concatenated_images) {
  TempDir dir;
  round_trip(dir.file("a.ppm"), make_info(PixelFormat::rgb24, 9, 4), ReaderOptions{}, 4, 11);
}

TEST(pnm_file_layout) {
  TempDir dir;
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.ppm"), make_info(PixelFormat::rgb24, 2, 1), std::nullopt,
                         writer));
  REQUIRE_OK(writer->write(random_frame(PixelFormat::rgb24, 2, 1, 1)));
  REQUIRE_OK(writer->finish());
  // "P6\n2 1\n255\n" is 11 bytes, then 2 RGB pixels.
  CHECK_EQ(fs::file_size(dir.file("a.ppm")), static_cast<std::uintmax_t>(11 + 6));
}

namespace {

Status open_pnm_bytes(const TempDir& dir, const std::string& bytes,
                      std::unique_ptr<FrameReader>& reader) {
  write_bytes(dir.file("t.pnm"), bytes);
  return open_reader(dir.file("t.pnm"), ReaderOptions{}, reader);
}

}  // namespace

TEST(pnm_header_forms) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  // Comments and mixed whitespace between fields, CRLF after the maxval.
  REQUIRE_OK(open_pnm_bytes(
      dir, "P5 # a comment\n# another\n2\t1\r\n255\r\n\x0a\x0b", reader));
  CHECK(reader->info().format == PixelFormat::gray8);
  CHECK_EQ(reader->info().width, 2);
  Frame f;
  // Exactly one byte ('\r') follows the maxval; the pixels are "\n\x0a" and the
  // trailing whitespace byte is not another image.
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(f.plane(0).row(0)[0], 0x0a);
  CHECK_EQ(f.plane(0).row(0)[1], 0x0a);
  CHECK_EQ(reader->read(f), Status::end_of_stream);

  REQUIRE_OK(open_pnm_bytes(dir, "P6\n1 1\n255\n\x01\x02\x03", reader));
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(f.plane(0).row(0)[2], 3);
  CHECK_EQ(reader->read(f), Status::end_of_stream);
}

TEST(pnm_unsupported) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  for (const char* magic : {"P1", "P2", "P3", "P4", "P7"}) {
    CHECK_EQ(open_pnm_bytes(dir, std::string(magic) + "\n1 1\n255\n\x01\x02\x03", reader),
             Status::unsupported_format);
  }
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 1\n65535\n123456", reader), Status::unsupported_format);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 1\n15\n123", reader), Status::unsupported_format);
  CHECK_EQ(open_pnm_bytes(dir, "P5\n99999 1\n255\n", reader), Status::unsupported_format);
}

TEST(pnm_malformed) {
  TempDir dir;
  std::unique_ptr<FrameReader> reader;
  Frame f;
  CHECK_EQ(open_pnm_bytes(dir, "", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "   \n", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "Q6\n1 1\n255\n123", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 1\n", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 x\n255\n123", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n0 1\n255\n", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n-1 1\n255\n", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 1\n0\n123", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 1\n999999\n123", reader), Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6\n1 11111111111111111111111\n255\n", reader),
           Status::parse_error);
  CHECK_EQ(open_pnm_bytes(dir, "P6 # never ends", reader), Status::parse_error);

  // Truncated pixels, in the first image and in a later one.
  REQUIRE_OK(open_pnm_bytes(dir, "P6\n2 1\n255\n12345", reader));
  CHECK_EQ(reader->read(f), Status::parse_error);
  REQUIRE_OK(open_pnm_bytes(dir, "P5\n2 1\n255\n12P5\n2 1\n255\n1", reader));
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(reader->read(f), Status::parse_error);
  // A later image with a different shape.
  REQUIRE_OK(open_pnm_bytes(dir, "P5\n2 1\n255\n12P5\n3 1\n255\n123", reader));
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(reader->read(f), Status::parse_error);
  REQUIRE_OK(open_pnm_bytes(dir, "P5\n2 1\n255\n12P6\n2 1\n255\n123456", reader));
  REQUIRE_OK(reader->read(f));
  CHECK_EQ(reader->read(f), Status::parse_error);
}

TEST(pnm_writer_rejects_unsupported) {
  TempDir dir;
  std::unique_ptr<FrameWriter> writer;
  for (const PixelFormat fmt : {PixelFormat::yuv420p, PixelFormat::nv12, PixelFormat::rgba,
                                PixelFormat::bgra}) {
    CHECK_EQ(open_writer(dir.file("a.ppm"), make_info(fmt, 4, 4), std::nullopt, writer),
             Status::unsupported_format);
  }
}

TEST(explicit_container_overrides_extension) {
  TempDir dir;
  const VideoInfo info = make_info(PixelFormat::rgb24, 4, 3);
  std::unique_ptr<FrameWriter> writer;
  REQUIRE_OK(open_writer(dir.file("a.dat"), info, Container::pnm, writer));
  REQUIRE_OK(writer->write(random_frame(info.format, 4, 3, 1)));
  REQUIRE_OK(writer->finish());
  ReaderOptions opts;
  opts.container = Container::pnm;
  std::unique_ptr<FrameReader> reader;
  REQUIRE_OK(open_reader(dir.file("a.dat"), opts, reader));
  Frame f;
  REQUIRE_OK(reader->read(f));
  CHECK(first_difference(random_frame(info.format, 4, 3, 1), f).empty());
}

}  // namespace lanecast::test
