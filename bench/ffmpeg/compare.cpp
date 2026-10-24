#include "compare.hpp"

#include <algorithm>
#include <memory>

extern "C" {
#include <libswscale/swscale.h>
}

#include "lanecast/metrics.hpp"

namespace lanecast::bench::ff {
namespace {

double psnr(const Frame& a, const Frame& b) {
  PsnrResult r;
  return compute_psnr(a, b, r) == Status::ok ? r.average : -1.0;
}

}  // namespace

LanecastOp sliced_op(const Frame& dst, int src_row_alignment, std::function<Status(RowRange)> fn) {
  const int height = dst.height();
  const int align = std::max(src_row_alignment, pixfmt_row_alignment(dst.format()));
  return [height, align, fn = std::move(fn)](ThreadPool* pool) {
    return run_sliced(pool, height, align, fn);
  };
}

unsigned reference_flags() {
  return SWS_LANCZOS | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT | SWS_FULL_CHR_H_INP | SWS_BITEXACT;
}

std::string flag_names(unsigned flags) {
  static const struct {
    unsigned bit;
    const char* name;
  } names[] = {{SWS_BILINEAR, "BILINEAR"},         {SWS_BICUBIC, "BICUBIC"},
               {SWS_POINT, "POINT"},               {SWS_AREA, "AREA"},
               {SWS_LANCZOS, "LANCZOS"},           {SWS_FULL_CHR_H_INT, "FULL_CHR_H_INT"},
               {SWS_FULL_CHR_H_INP, "FULL_CHR_H_INP"}, {SWS_ACCURATE_RND, "ACCURATE_RND"},
               {SWS_BITEXACT, "BITEXACT"}};
  std::string out;
  for (const auto& n : names) {
    if (flags & n.bit) out += (out.empty() ? "" : "|") + std::string(n.name);
  }
  return out;
}

void run_comparison(const Comparison& c, Pattern pattern, const Options& opts, int threads,
                    ThreadPool& pool, std::vector<Row>& rows, std::vector<std::string>& notes) {
  Frame src, out_lc, out_sws, out_ref;
  if (make_pattern_frame(pattern, c.src_format, c.src_width, c.src_height, src) != Status::ok ||
      Frame::allocate(c.dst_format, c.dst_width, c.dst_height, out_lc) != Status::ok ||
      Frame::allocate(c.dst_format, c.dst_width, c.dst_height, out_sws) != Status::ok ||
      Frame::allocate(c.dst_format, c.dst_width, c.dst_height, out_ref) != Status::ok) {
    notes.push_back("skipped " + c.op + ": frame allocation failed");
    return;
  }
  const auto lc_op = c.lanecast(src, out_lc);
  if (!lc_op || lc_op(nullptr) != Status::ok) {
    notes.push_back("skipped " + c.op + ": lanecast side not implemented in this build");
    return;
  }

  std::string error;
  SwsSpec ref_spec = c.sws;
  ref_spec.flags = reference_flags();
  const auto ref = SwsRunner::create(ref_spec, src, out_ref, 1, error);
  if (!ref || !ref->run()) {
    notes.push_back("skipped " + c.op + ": swscale reference failed (" + error + ")");
    return;
  }

  std::vector<int> counts = {1};
  if (threads > 1) counts.push_back(threads);
  for (const int t : counts) {
    const auto sws = SwsRunner::create(c.sws, src, out_sws, t, error);
    if (!sws) {
      notes.push_back("skipped " + c.op + ": " + error);
      return;
    }
    ThreadPool* p = t > 1 ? &pool : nullptr;
    const auto run_lc = [&] { (void)lc_op(p); };
    const auto run_sws = [&] { (void)sws->run(); };
    Row row;
    row.op = c.op;
    row.pattern = pattern_name(pattern);
    row.sws = flag_names(c.sws.flags);
    row.width = c.src_width;
    row.height = c.src_height;
    row.lc_threads = t;
    row.sws_threads = std::max(1, sws->threads());
    row.lc_ns = median_ns(run_lc, opts.warmup, opts.iters);
    row.sws_ns = median_ns(run_sws, opts.warmup, opts.iters);
    row.psnr_lc_ref = psnr(out_lc, out_ref);
    row.psnr_sws_ref = psnr(out_sws, out_ref);
    row.psnr_lc_sws = psnr(out_lc, out_sws);
    rows.push_back(row);
  }
}

}  // namespace lanecast::bench::ff
