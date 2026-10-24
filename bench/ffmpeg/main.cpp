// bench_vs_ffmpeg: lanecast vs libswscale on identical input, output format
// and thread count. Reports median ns/frame, speedup and PSNR of each output
// against a neutral swscale reference (ARCHITECTURE.md "Beating ffmpeg").
//
//   bench_vs_ffmpeg [--quick] [--json] [--filter TEXT] [--iters N] [--warmup N]
//                   [--threads N]
//
// Output carries no CPU model, host name or paths.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libswscale/swscale.h>
}

#include "compare.hpp"

namespace {

using namespace lanecast;
using namespace lanecast::bench::ff;

void usage() {
  std::fputs(
      "usage: bench_vs_ffmpeg [--quick] [--json] [--filter TEXT] [--iters N]\n"
      "                       [--warmup N] [--threads N]\n",
      stderr);
}

bool parse(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--json") {
      o.json = true;
    } else if (a == "--quick") {
      o.quick = true;
    } else if (a == "--filter" && has_value) {
      o.filter = argv[++i];
    } else if (a == "--iters" && has_value) {
      o.iters = std::atoi(argv[++i]);
    } else if (a == "--warmup" && has_value) {
      o.warmup = std::atoi(argv[++i]);
    } else if (a == "--threads" && has_value) {
      o.threads = std::atoi(argv[++i]);
    } else {
      return false;
    }
  }
  if (o.quick) o.iters = 9;
  return o.iters >= 1 && o.warmup >= 0 && o.threads >= 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse(argc, argv, opts)) {
    usage();
    return 2;
  }
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  const int threads = opts.threads > 0 ? opts.threads : (hw > 0 ? hw : 1);

  std::vector<Comparison> all;
  add_convert_comparisons(all);
  add_scale_comparisons(all);
  add_fused_comparisons(all);

  ThreadPool pool(threads);
  std::vector<Row> rows;
  std::vector<std::string> notes;
  const unsigned v = swscale_version();
  notes.push_back("libswscale " + std::to_string(v >> 16) + "." + std::to_string((v >> 8) & 0xFF) +
                  "." + std::to_string(v & 0xFF) + ", " + std::to_string(threads) +
                  " threads for the multi-thread rows, median of " + std::to_string(opts.iters) +
                  " runs. PSNR is the average over all planes against a swscale reference "
                  "(LANCZOS|ACCURATE_RND|FULL_CHR_H_INT|FULL_CHR_H_INP|BITEXACT, same chroma "
                  "siting); inf means identical.");
  for (const Comparison& c : all) {
    if (!opts.filter.empty() && c.op.find(opts.filter) == std::string::npos) continue;
    for (const Pattern p : {Pattern::testsrc, Pattern::noise}) {
      run_comparison(c, p, opts, threads, pool, rows, notes);
    }
  }
  std::vector<std::string> unique_notes;  // skips repeat once per pattern
  for (const std::string& n : notes) {
    if (std::find(unique_notes.begin(), unique_notes.end(), n) == unique_notes.end()) {
      unique_notes.push_back(n);
    }
  }
  if (opts.json) {
    write_json(rows, unique_notes);
  } else {
    write_table(rows, unique_notes);
  }
  return 0;
}
