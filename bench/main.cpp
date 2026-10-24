// lanecast_bench: per-kernel, per-ISA microbenchmarks.
//
//   lanecast_bench [--quick] [--json] [--filter TEXT] [--size WxH] [--iters N]
//                  [--warmup N] [--threads N]
//
// Every case runs under each supported ISA tier (forced with
// set_active_isa), on each frame size, single-threaded and, when the pool has
// more than one worker, on all workers. Each configuration is timed with
// `warmup` untimed runs and then the median of `iters` runs. A case whose
// scalar run is not implemented yet is skipped and listed at the end.
// Output carries ISA tier names only: no CPU model, host name or paths.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "bench.hpp"
#include "lanecast/cpu.hpp"

namespace lanecast::bench {
namespace {

struct Options {
  bool json = false;
  bool quick = false;
  std::string filter;
  std::vector<Size> sizes;
  int iters = 21;
  int warmup = 3;
  int threads = 0;  // 0: hardware threads
};

struct Row {
  std::string name;
  Size size;
  int threads;
  std::string isa;
  double ns_per_run;
  double ns_per_pixel;
  double mpix_per_s;
  double gb_per_s;
  double speedup;  // vs scalar, same case, size and thread count
};

struct Skip {
  std::string name;
  std::string reason;
};

void usage() {
  std::fputs(
      "usage: lanecast_bench [--quick] [--json] [--filter TEXT] [--size WxH]\n"
      "                      [--iters N] [--warmup N] [--threads N]\n",
      stderr);
}

bool parse_int(const char* s, int& out) {
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < 1 || v > 100000) return false;
  out = static_cast<int>(v);
  return true;
}

bool parse_args(int argc, char** argv, Options& o) {
  bool iters_set = false;
  bool warmup_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--json") {
      o.json = true;
    } else if (a == "--quick") {
      o.quick = true;
    } else if (a == "--filter" && has_value) {
      o.filter = argv[++i];
    } else if (a == "--size" && has_value) {
      Size s{};
      if (std::sscanf(argv[++i], "%dx%d", &s.width, &s.height) != 2 || s.width < 2 ||
          s.height < 2 || s.width > kMaxDimension || s.height > kMaxDimension) {
        return false;
      }
      o.sizes.push_back(s);
    } else if (a == "--iters" && has_value) {
      if (!parse_int(argv[++i], o.iters)) return false;
      iters_set = true;
    } else if (a == "--warmup" && has_value) {
      if (!parse_int(argv[++i], o.warmup)) return false;
      warmup_set = true;
    } else if (a == "--threads" && has_value) {
      if (!parse_int(argv[++i], o.threads)) return false;
    } else {
      return false;
    }
  }
  if (o.quick) {
    if (!iters_set) o.iters = 5;
    if (!warmup_set) o.warmup = 1;
    if (o.sizes.empty()) o.sizes = {{640, 360}};
  }
  if (o.sizes.empty()) o.sizes = {{640, 360}, {1920, 1080}, {3840, 2160}};
  return true;
}

// Median wall time of one run, or a non-ok status from the first failure.
Status median_ns(const Instance& in, ThreadPool* pool, int warmup, int iters, double& out) {
  for (int i = 0; i < warmup; ++i) {
    if (const Status s = in.run(pool); s != Status::ok) return s;
  }
  std::vector<double> samples(static_cast<std::size_t>(iters));
  for (double& sample : samples) {
    const auto t0 = std::chrono::steady_clock::now();
    const Status s = in.run(pool);
    const auto t1 = std::chrono::steady_clock::now();
    if (s != Status::ok) return s;
    sample = std::chrono::duration<double, std::nano>(t1 - t0).count();
  }
  const auto mid = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
  std::nth_element(samples.begin(), mid, samples.end());
  out = *mid;
  return Status::ok;
}

void print_header() {
  std::printf("%-34s %-10s %3s  %-13s %9s %10s %8s %8s\n", "kernel", "size", "thr", "isa",
              "ns/px", "Mpix/s", "GB/s", "vs scalar");
}

void print_row(const Row& r) {
  const std::string size = std::to_string(r.size.width) + "x" + std::to_string(r.size.height);
  std::printf("%-34s %-10s %3d  %-13s %9.3f %10.1f %8.2f %8.2fx\n", r.name.c_str(), size.c_str(),
              r.threads, r.isa.c_str(), r.ns_per_pixel, r.mpix_per_s, r.gb_per_s, r.speedup);
}

void print_json(const std::vector<Row>& rows, const std::vector<Skip>& skips,
                const Options& o) {
  std::printf("{\n  \"iters\": %d,\n  \"warmup\": %d,\n  \"results\": [", o.iters, o.warmup);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    std::printf(
        "%s\n    {\"kernel\": \"%s\", \"width\": %d, \"height\": %d, \"threads\": %d, "
        "\"isa\": \"%s\", \"ns_per_run\": %.1f, \"ns_per_pixel\": %.4f, \"mpix_per_s\": %.2f, "
        "\"gb_per_s\": %.3f, \"speedup_vs_scalar\": %.3f}",
        i == 0 ? "" : ",", r.name.c_str(), r.size.width, r.size.height, r.threads,
        r.isa.c_str(), r.ns_per_run, r.ns_per_pixel, r.mpix_per_s, r.gb_per_s, r.speedup);
  }
  std::printf("\n  ],\n  \"skipped\": [");
  for (std::size_t i = 0; i < skips.size(); ++i) {
    std::printf("%s\n    {\"kernel\": \"%s\", \"reason\": \"%s\"}", i == 0 ? "" : ",",
                skips[i].name.c_str(), skips[i].reason.c_str());
  }
  std::printf("\n  ]\n}\n");
}

int run(const Options& o) {
  std::vector<Case> cases;
  register_convert(cases);
  register_scale(cases);
  register_filter(cases);
  register_metrics(cases);

  const int pool_threads =
      o.threads > 0 ? o.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  ThreadPool pool(pool_threads);
  const int workers = pool.size();

  const Isa original = active_isa();
  const std::vector<Isa> isas = supported_isas();
  std::vector<Row> rows;
  std::vector<Skip> skips;
  if (!o.json) print_header();

  for (const Case& c : cases) {
    if (!o.filter.empty() && c.name.find(o.filter) == std::string::npos) continue;
    bool skipped = false;
    for (const Size size : o.sizes) {
      if (skipped) break;
      const Instance in = c.make(size);
      // The scalar run is the reference; if it is not available, skip the case.
      set_active_isa(Isa::scalar);
      if (const Status s = in.run(nullptr); s != Status::ok) {
        skips.push_back({c.name, std::string(status_name(s))});
        skipped = true;
        break;
      }
      std::vector<int> thread_counts = {1};
      if (in.sliceable && workers > 1) thread_counts.push_back(workers);

      const double pixels = static_cast<double>(size.width) * size.height;
      for (const int threads : thread_counts) {
        ThreadPool* const p = threads > 1 ? &pool : nullptr;
        double scalar_ns = 0.0;
        for (const Isa isa : isas) {
          set_active_isa(isa);
          double ns = 0.0;
          if (const Status s = median_ns(in, p, o.warmup, o.iters, ns); s != Status::ok) {
            skips.push_back({c.name + " " + std::string(isa_name(isa)),
                             std::string(status_name(s))});
            continue;
          }
          if (isa == Isa::scalar) scalar_ns = ns;
          Row r;
          r.name = c.name;
          r.size = size;
          r.threads = threads;
          r.isa = std::string(isa_name(isa));
          r.ns_per_run = ns;
          r.ns_per_pixel = ns / pixels;
          r.mpix_per_s = pixels / ns * 1e3;
          r.gb_per_s = static_cast<double>(in.bytes) / ns;
          r.speedup = scalar_ns > 0.0 ? scalar_ns / ns : 0.0;
          if (!o.json) print_row(r);
          rows.push_back(std::move(r));
        }
      }
    }
  }
  set_active_isa(original);

  if (o.json) {
    print_json(rows, skips, o);
  } else if (!skips.empty()) {
    std::printf("\nskipped:\n");
    for (const Skip& s : skips) std::printf("  %s (%s)\n", s.name.c_str(), s.reason.c_str());
  }
  return 0;
}

}  // namespace
}  // namespace lanecast::bench

int main(int argc, char** argv) {
  lanecast::bench::Options options;
  if (!lanecast::bench::parse_args(argc, argv, options)) {
    lanecast::bench::usage();
    return 2;
  }
  return lanecast::bench::run(options);
}
