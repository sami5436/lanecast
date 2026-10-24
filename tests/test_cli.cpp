// cli suite: runs the lanecast binary end to end on generated inputs.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include "test_support.hpp"

#ifdef LANECAST_CLI_PATH

namespace {

namespace fs = std::filesystem;
using lanecast::test::Rng;

constexpr int kW = 64;
constexpr int kH = 48;
constexpr int kFrames = 3;

class TempDir {
 public:
  TempDir() {
    static int counter = 0;
    path_ = fs::temp_directory_path() /
            ("lanecast_cli_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  std::string file(const std::string& name) const { return (path_ / name).string(); }

 private:
  fs::path path_;
};

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void spit(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Random yuv420p frames, tightly packed.
std::string raw_frames(int w, int h, int n, std::uint64_t seed) {
  Rng rng(seed);
  std::string s;
  const std::size_t frame = static_cast<std::size_t>(w) * h + 2 * static_cast<std::size_t>((w + 1) / 2) * ((h + 1) / 2);
  for (std::size_t i = 0; i < frame * static_cast<std::size_t>(n); ++i) {
    s.push_back(static_cast<char>(rng.byte()));
  }
  return s;
}

std::string y4m_of(const std::string& raw, int w, int h, int n) {
  const std::size_t frame = raw.size() / static_cast<std::size_t>(n);
  std::string s = "YUV4MPEG2 W" + std::to_string(w) + " H" + std::to_string(h) +
                  " F30:1 Ip A1:1 C420mpeg2 XCOLORRANGE=LIMITED\n";
  for (int i = 0; i < n; ++i) {
    s += "FRAME\n";
    s += raw.substr(frame * static_cast<std::size_t>(i), frame);
  }
  return s;
}

// Runs `lanecast <args>` through the shell; returns the exit code. stderr
// goes to `err_file` when given.
int run_cli(const std::string& args, const std::string& err_file = "") {
  std::string cmd = std::string("'") + LANECAST_CLI_PATH + "' " + args;
  cmd += err_file.empty() ? " 2>/dev/null" : " 2>'" + err_file + "'";
  const int status = std::system(cmd.c_str());
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string q(const std::string& path) { return "'" + path + "'"; }

struct Fixture {
  TempDir dir;
  std::string raw = raw_frames(kW, kH, kFrames, 42);
  std::string y4m_path = dir.file("in.y4m");
  Fixture() { spit(y4m_path, y4m_of(raw, kW, kH, kFrames)); }
};

TEST(cli_y4m_to_ppm_with_scale_and_format) {
  Fixture f;
  const std::string out = f.dir.file("out.ppm");
  REQUIRE(run_cli("-i " + q(f.y4m_path) + " -vf scale=32:-2,format=rgb24 -o " + q(out)) == 0);
  const std::string ppm = slurp(out);
  const std::string header = "P6\n32 24\n255\n";
  const std::size_t image = header.size() + 32 * 24 * 3;
  REQUIRE(ppm.size() == image * kFrames);
  for (int i = 0; i < kFrames; ++i) {
    CHECK(ppm.compare(image * static_cast<std::size_t>(i), header.size(), header) == 0);
  }
}

TEST(cli_identity_y4m_roundtrip) {
  Fixture f;
  const std::string out = f.dir.file("out.y4m");
  REQUIRE(run_cli("-i " + q(f.y4m_path) + " -o " + q(out)) == 0);
  CHECK(slurp(out) == slurp(f.y4m_path));
}

TEST(cli_threads_do_not_change_output) {
  Fixture f;
  const std::string a = f.dir.file("a.ppm");
  const std::string b = f.dir.file("b.ppm");
  const std::string vf = " -vf scale=45:33:flags=area,format=rgb24 ";
  REQUIRE(run_cli("-i " + q(f.y4m_path) + vf + "-threads 1 -o " + q(a)) == 0);
  REQUIRE(run_cli("-i " + q(f.y4m_path) + vf + "-threads 4 -o " + q(b)) == 0);
  CHECK(!slurp(a).empty());
  CHECK(slurp(a) == slurp(b));
}

TEST(cli_raw_input_and_frames_limit) {
  Fixture f;
  const std::string raw_path = f.dir.file("in.yuv");
  spit(raw_path, f.raw);
  const std::string out = f.dir.file("out.y4m");
  REQUIRE(run_cli("-s 64x48 -pix_fmt yuv420p -r 30 -i " + q(raw_path) + " -frames 2 -o " + q(out)) == 0);
  CHECK(slurp(out) == y4m_of(f.raw.substr(0, f.raw.size() / kFrames * 2), kW, kH, 2));
  CHECK(run_cli("-i " + q(raw_path) + " -o " + q(f.dir.file("x.y4m"))) != 0);  // raw needs -s
}

TEST(cli_pipes_match_files) {
  Fixture f;
  const std::string by_file = f.dir.file("file.y4m");
  const std::string by_pipe = f.dir.file("pipe.y4m");
  const std::string vf = " -vf scale=30:20 ";
  REQUIRE(run_cli("-i " + q(f.y4m_path) + vf + "-o " + q(by_file)) == 0);
  REQUIRE(run_cli("-i - " + vf + "-o - < " + q(f.y4m_path) + " > " + q(by_pipe)) == 0);
  CHECK(!slurp(by_file).empty());
  CHECK(slurp(by_file) == slurp(by_pipe));

  // Raw on stdin, rgb on stdout.
  const std::string raw_path = f.dir.file("in.yuv");
  const std::string rgb = f.dir.file("out.rgb");
  spit(raw_path, f.raw);
  REQUIRE(run_cli("-s 64x48 -pix_fmt yuv420p -i - -vf format=rgb24 -f rawvideo -o - < " +
                  q(raw_path) + " > " + q(rgb)) == 0);
  CHECK_EQ(slurp(rgb).size(), static_cast<std::size_t>(kW) * kH * 3 * kFrames);
}

TEST(cli_refuses_to_overwrite_without_y) {
  Fixture f;
  const std::string out = f.dir.file("out.y4m");
  spit(out, "keep");
  const std::string err = f.dir.file("err.txt");
  CHECK(run_cli("-i " + q(f.y4m_path) + " -o " + q(out), err) != 0);
  CHECK(slurp(out) == "keep");
  CHECK(slurp(err).find("-y") != std::string::npos);
  CHECK(run_cli("-i " + q(f.y4m_path) + " -y -o " + q(out)) == 0);
  CHECK(slurp(out) == slurp(f.y4m_path));
}

TEST(cli_benchmark_reports_rates) {
  Fixture f;
  const std::string err = f.dir.file("err.txt");
  REQUIRE(run_cli("-i " + q(f.y4m_path) + " -benchmark -o " + q(f.dir.file("o.y4m")), err) == 0);
  const std::string text = slurp(err);
  CHECK(text.find("fps") != std::string::npos);
  CHECK(text.find("ms/frame") != std::string::npos);
  CHECK(text.find("frames: 3") != std::string::npos);
}

TEST(cli_bad_arguments_fail_with_a_message) {
  Fixture f;
  const std::string err = f.dir.file("err.txt");
  const std::string out = q(f.dir.file("o.y4m"));
  const std::string in = "-i " + q(f.y4m_path) + " ";
  const std::string cases[] = {
      "",                                              // nothing
      "-o " + out,                                     // no input
      in,                                              // no output
      "-i " + q(f.dir.file("missing.y4m")) + " -o " + out,
      in + "-vf nosuchfilter -o " + out,
      in + "-vf scale=10 -o " + out,
      in + "-vf format=rgb24 -o " + out,               // y4m cannot hold rgb24
      in + "-threads x -o " + out,
      in + "-frobnicate 1 -o " + out,
      in + "-o",                                       // missing value
      in + "-o " + q(f.dir.file("o.unknown")),         // unknown container
  };
  for (const std::string& args : cases) {
    CHECK_MSG(run_cli(args, err) != 0, args);
    CHECK_MSG(slurp(err).find("lanecast: error:") != std::string::npos ||
                  slurp(err).find("usage:") != std::string::npos,
              args);
  }
}

}  // namespace

#endif  // LANECAST_CLI_PATH
