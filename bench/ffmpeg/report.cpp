// Markdown table and JSON output. Carries sizes, thread counts and ratios
// only: no CPU model, host name or paths.
#include <cmath>
#include <cstdio>

#include "harness.hpp"

namespace lanecast::bench::ff {
namespace {

std::string db(double v) {
  if (std::isinf(v)) return "inf";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f", v);
  return buf;
}

std::string json_db(double v) { return std::isinf(v) ? "null" : db(v); }

std::string escaped(const std::string& s) {
  std::string out;
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

}  // namespace

void write_table(const std::vector<Row>& rows, const std::vector<std::string>& notes) {
  std::puts("| op | pattern | src | thr lc/sws | lanecast ns | swscale ns | speedup |"
            " PSNR lc/ref | PSNR sws/ref | PSNR lc/sws | sws flags |");
  std::puts("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|");
  for (const Row& r : rows) {
    std::printf("| %s | %s | %dx%d | %d/%d | %.0f | %.0f | %.2fx | %s | %s | %s | %s |\n",
                r.op.c_str(), r.pattern.c_str(), r.width, r.height, r.lc_threads, r.sws_threads,
                r.lc_ns, r.sws_ns, r.sws_ns / r.lc_ns, db(r.psnr_lc_ref).c_str(),
                db(r.psnr_sws_ref).c_str(), db(r.psnr_lc_sws).c_str(), r.sws.c_str());
  }
  for (const std::string& n : notes) std::printf("\n%s\n", n.c_str());
}

void write_json(const std::vector<Row>& rows, const std::vector<std::string>& notes) {
  std::puts("{\"rows\":[");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    std::printf(
        "{\"op\":\"%s\",\"pattern\":\"%s\",\"sws_flags\":\"%s\",\"width\":%d,\"height\":%d,"
        "\"lanecast_threads\":%d,\"swscale_threads\":%d,\"lanecast_ns\":%.0f,"
        "\"swscale_ns\":%.0f,\"speedup\":%.3f,\"psnr_lanecast_ref\":%s,"
        "\"psnr_swscale_ref\":%s,\"psnr_lanecast_swscale\":%s}%s\n",
        escaped(r.op).c_str(), r.pattern.c_str(), escaped(r.sws).c_str(), r.width, r.height,
        r.lc_threads, r.sws_threads, r.lc_ns, r.sws_ns, r.sws_ns / r.lc_ns,
        json_db(r.psnr_lc_ref).c_str(), json_db(r.psnr_sws_ref).c_str(),
        json_db(r.psnr_lc_sws).c_str(), i + 1 < rows.size() ? "," : "");
  }
  std::puts("],\"notes\":[");
  for (std::size_t i = 0; i < notes.size(); ++i) {
    std::printf("\"%s\"%s\n", escaped(notes[i]).c_str(), i + 1 < notes.size() ? "," : "");
  }
  std::puts("]}");
}

}  // namespace lanecast::bench::ff
