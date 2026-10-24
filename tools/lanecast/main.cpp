// lanecast CLI: ffmpeg-style raw video conversion.
//   lanecast -i IN -o OUT [-vf CHAIN] [-threads N] [-frames N] [-y] [-benchmark]
// plus the `info` and `--version` commands.
#include <charconv>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "lanecast/cpu.hpp"
#include "lanecast/io.hpp"
#include "lanecast/lanecast.hpp"
#include "lanecast/version.hpp"

namespace {

using namespace lanecast;

void print_usage(std::FILE* out) {
  std::fputs(
      "usage: lanecast -i IN -o OUT [options]\n"
      "       lanecast info | --version | --help\n"
      "\n"
      "  -i PATH         input .y4m, raw (.yuv .raw .rgb .nv12) or .ppm/.pgm; '-' is stdin\n"
      "  -o PATH         output .y4m, raw, .ppm or .pgm; '-' is stdout\n"
      "  -s WxH          size of raw input\n"
      "  -pix_fmt FMT    pixel format of raw input (gray8 yuv420p nv12 rgb24 rgba bgra)\n"
      "  -r FPS          frame rate of raw input, N or N/D (default 25)\n"
      "  -f FMT          container: y4m, rawvideo, ppm, pgm. Before -i it names the input,\n"
      "                  after -i the output. Needed only for '-' and unusual extensions\n"
      "                  (stdin defaults to raw when -s is given, else y4m; stdout to y4m\n"
      "                  for yuv420p and gray8, else raw)\n"
      "  -vf CHAIN       filter chain, e.g. scale=1280:720,format=rgb24\n"
      "                  scale=W:H[:flags=bilinear|area]   W or H of -1 / -2 keeps the aspect\n"
      "                  format=PIXFMT[:matrix=bt601|bt709][:range=limited|full]\n"
      "                         [:chroma=nearest|bilinear]\n"
      "                  blur=3|5   eq=brightness=B:contrast=C\n"
      "  -threads N      worker threads (default: all hardware threads, 1 disables)\n"
      "  -frames N       stop after N frames\n"
      "  -y              overwrite the output file\n"
      "  -benchmark      print frames/s and ms/frame to stderr\n",
      out);
}

struct Options {
  std::string input;
  std::string output;
  std::string chain;
  std::optional<Container> input_container;
  std::optional<Container> output_container;
  std::optional<int> width;
  std::optional<int> height;
  std::optional<PixelFormat> pix_fmt;
  int fps_num = 25;
  int fps_den = 1;
  int threads = 0;
  long long frames = -1;
  bool overwrite = false;
  bool benchmark = false;
};

int fail(const std::string& message) {
  std::fprintf(stderr, "lanecast: error: %s\n", message.c_str());
  return 1;
}

bool parse_int(std::string_view s, long long& out) {
  const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
  return !s.empty() && r.ec == std::errc() && r.ptr == s.data() + s.size();
}

std::optional<Container> parse_container(std::string_view name) {
  if (name == "y4m") return Container::y4m;
  if (name == "rawvideo" || name == "raw") return Container::raw;
  if (name == "ppm" || name == "pgm" || name == "pnm") return Container::pnm;
  return std::nullopt;
}

// Returns an error message, or empty on success. `help` is set for -h.
std::string parse_args(int argc, char** argv, Options& o, bool& help) {
  bool seen_input = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    const auto value = [&](std::string& out) {
      if (i + 1 >= argc) return false;
      out = argv[++i];
      return true;
    };
    std::string v;
    if (a == "-y") {
      o.overwrite = true;
    } else if (a == "-benchmark") {
      o.benchmark = true;
    } else if (a == "-h" || a == "-help" || a == "--help") {
      help = true;
    } else if (a == "convert" && i == 1) {
      // optional subcommand name, accepted for readability
    } else if (a.size() > 1 && a[0] == '-') {
      if (!value(v)) return "option " + std::string(a) + " needs a value";
      long long n = 0;
      if (a == "-i") {
        o.input = v;
        seen_input = true;
      } else if (a == "-o") {
        o.output = v;
      } else if (a == "-vf") {
        o.chain = v;
      } else if (a == "-s") {
        const auto x = v.find('x');
        long long w = 0;
        long long h = 0;
        if (x == std::string::npos || !parse_int(std::string_view(v).substr(0, x), w) ||
            !parse_int(std::string_view(v).substr(x + 1), h) || w < 1 || h < 1 ||
            w > kMaxDimension || h > kMaxDimension) {
          return "-s expects WxH within 1.." + std::to_string(kMaxDimension) + ", got '" + v + "'";
        }
        o.width = static_cast<int>(w);
        o.height = static_cast<int>(h);
      } else if (a == "-pix_fmt") {
        o.pix_fmt = parse_pixfmt(v);
        if (!o.pix_fmt) return "unknown pixel format '" + v + "'";
      } else if (a == "-r") {
        const auto slash = v.find('/');
        long long num = 0;
        long long den = 1;
        if (!parse_int(std::string_view(v).substr(0, slash), num) ||
            (slash != std::string::npos && !parse_int(std::string_view(v).substr(slash + 1), den)) ||
            num < 1 || den < 1 || num > 1000000 || den > 1000000) {
          return "-r expects N or N/D, got '" + v + "'";
        }
        o.fps_num = static_cast<int>(num);
        o.fps_den = static_cast<int>(den);
      } else if (a == "-f") {
        const auto c = parse_container(v);
        if (!c) return "unknown container '" + v + "' (use y4m, rawvideo, ppm or pgm)";
        (seen_input ? o.output_container : o.input_container) = c;
      } else if (a == "-threads") {
        if (!parse_int(v, n) || n < 0 || n > 1024) return "-threads expects 0..1024, got '" + v + "'";
        o.threads = static_cast<int>(n);
      } else if (a == "-frames") {
        if (!parse_int(v, n) || n < 1) return "-frames expects a positive number, got '" + v + "'";
        o.frames = n;
      } else {
        return "unknown option '" + std::string(a) + "'";
      }
    } else {
      return "unexpected argument '" + std::string(a) + "'";
    }
  }
  return {};
}

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int cmd_convert(Options& o) {
  if (o.input.empty()) return fail("no input given (-i)");
  if (o.output.empty()) return fail("no output given (-o)");

  ReaderOptions ro;
  ro.container = o.input_container;
  if (!ro.container && o.input == "-") {
    ro.container = (o.width || o.pix_fmt) ? Container::raw : Container::y4m;
  }
  const auto in_container = ro.container ? ro.container : container_from_path(o.input);
  if (!in_container) return fail("cannot tell the container of '" + o.input + "' (use -f)");
  if (*in_container == Container::raw) {
    if (!o.width || !o.pix_fmt) return fail("raw input needs -s WxH and -pix_fmt");
    ro.raw.width = *o.width;
    ro.raw.height = *o.height;
    ro.raw.format = *o.pix_fmt;
    ro.raw.fps_num = o.fps_num;
    ro.raw.fps_den = o.fps_den;
  }

  if (o.output != "-" && !o.overwrite && std::filesystem::exists(o.output)) {
    return fail("output '" + o.output + "' exists, use -y to overwrite");
  }

  std::unique_ptr<FrameReader> reader;
  if (const Status s = open_reader(o.input, ro, reader); s != Status::ok) {
    return fail("cannot read input '" + o.input + "': " + std::string(status_name(s)));
  }

  PipelineOptions po;
  po.threads = o.threads;
  std::unique_ptr<Pipeline> pipeline;
  std::string chain_error;
  if (const Status s = Pipeline::create(o.chain, reader->info(), po, pipeline, &chain_error);
      s != Status::ok) {
    return fail(chain_error.empty() ? "cannot build the filter chain: " + std::string(status_name(s))
                                    : chain_error);
  }

  const VideoInfo out_info = pipeline->output_info();
  auto out_container = o.output_container;
  if (!out_container && o.output == "-") {
    const bool y4m = out_info.format == PixelFormat::yuv420p || out_info.format == PixelFormat::gray8;
    out_container = y4m ? Container::y4m : Container::raw;
  }
  if (!out_container && !container_from_path(o.output)) {
    return fail("cannot tell the container of '" + o.output + "' (use -f)");
  }
  std::unique_ptr<FrameWriter> writer;
  if (const Status s = open_writer(o.output, out_info, out_container, writer); s != Status::ok) {
    if (s == Status::unsupported_format) {
      return fail("this container cannot hold " + std::string(pixfmt_name(out_info.format)) +
                  " frames; add -vf format=... (y4m: yuv420p, gray8; ppm: rgb24; pgm: gray8)");
    }
    return fail("cannot write output '" + o.output + "': " + std::string(status_name(s)));
  }

  StreamOptions so;
  so.max_frames = o.frames;
  const auto start = std::chrono::steady_clock::now();
  const StreamResult run = run_stream(*reader, *pipeline, *writer, so);
  if (run.status != Status::ok) {
    const char* what = run.stage == StreamResult::Stage::read      ? "reading"
                       : run.stage == StreamResult::Stage::process ? "processing"
                                                                   : "writing";
    return fail(std::string(what) + " frame " + std::to_string(run.failed_frame) + ": " +
                std::string(status_name(run.status)));
  }
  const long long count = run.frames;
  if (const Status s = writer->finish(); s != Status::ok) {
    return fail("finishing output: " + std::string(status_name(s)));
  }
  if (count == 0) return fail("input has no frames");

  if (o.benchmark) {
    const double total = seconds_since(start);
    const double process_seconds = run.process_seconds;
    const auto n = static_cast<double>(count);
    std::fprintf(stderr, "frames: %lld\n", count);
    std::fprintf(stderr, "total:   %8.2f fps  %8.3f ms/frame  (read, process, write)\n", n / total,
                 total * 1000 / n);
    std::fprintf(stderr, "process: %8.2f fps  %8.3f ms/frame\n", n / process_seconds,
                 process_seconds * 1000 / n);
  }
  return 0;
}

int cmd_info() {
  std::printf("lanecast %.*s\n", static_cast<int>(kVersionString.size()), kVersionString.data());
  for (int i = 0; i < kIsaCount; ++i) {
    const auto isa = static_cast<Isa>(i);
    const std::string_view name = isa_name(isa);
    std::printf("  %-13.*s compiled=%d supported=%d\n", static_cast<int>(name.size()),
                name.data(), isa_compiled(isa) ? 1 : 0, isa_supported(isa) ? 1 : 0);
  }
  const std::string_view active = isa_name(active_isa());
  std::printf("active: %.*s\n", static_cast<int>(active.size()), active.data());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(stderr);
    return 1;
  }
  const std::string_view first = argv[1];
  if (first == "info") return cmd_info();
  if (first == "--version") {
    std::printf("%.*s\n", static_cast<int>(kVersionString.size()), kVersionString.data());
    return 0;
  }
  if (first == "help") {
    print_usage(stdout);
    return 0;
  }
  Options options;
  bool help = false;
  if (const std::string error = parse_args(argc, argv, options, help); !error.empty()) {
    return fail(error);
  }
  if (help) {
    print_usage(stdout);
    return 0;
  }
  return cmd_convert(options);
}
