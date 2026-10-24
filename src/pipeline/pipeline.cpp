// The staged, slice-threaded Pipeline: filter specs are resolved against the
// input shape once in create(), every intermediate frame is allocated there,
// and process() only runs row slices on the pool.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "lanecast/filter.hpp"
#include "lanecast/pipeline.hpp"

namespace lanecast {
namespace {

// One resolved filter. run() writes all of `out` from `in`, sliced on `pool`.
class Stage {
 public:
  virtual ~Stage() = default;
  virtual Status run(const Frame& in, Frame& out, ThreadPool* pool, int slice_rows) const = 0;
  VideoInfo output;  // stream shape after this stage
};

class ScaleStage final : public Stage {
 public:
  Status run(const Frame& in, Frame& out, ThreadPool* pool, int slice_rows) const override {
    return run_sliced(
        pool, out.height(), pixfmt_row_alignment(out.format()),
        [&](RowRange r) { return scaler->run(in, out, r); }, slice_rows);
  }
  std::unique_ptr<Scaler> scaler;
  ScaleFilter filter = ScaleFilter::bilinear;
  VideoInfo input;  // stream shape before this stage, kept for fusion
};

class ScaleConvertStage final : public Stage {
 public:
  Status run(const Frame& in, Frame& out, ThreadPool* pool, int) const override {
    return fused->run(in, out, pool);
  }
  std::unique_ptr<ScaleConvert> fused;
};

class ConvertStage final : public Stage {
 public:
  Status run(const Frame& in, Frame& out, ThreadPool* pool, int slice_rows) const override {
    const int alignment =
        std::max(pixfmt_row_alignment(in.format()), pixfmt_row_alignment(out.format()));
    return run_sliced(
        pool, out.height(), alignment,
        [&](RowRange r) { return convert(in, out, opts, r); }, slice_rows);
  }
  ConvertOptions opts;
};

class BlurStage final : public Stage {
 public:
  Status run(const Frame& in, Frame& out, ThreadPool* pool, int slice_rows) const override {
    return run_sliced(
        pool, out.height(), pixfmt_row_alignment(out.format()),
        [&](RowRange r) { return blur(in, out, radius, r); }, slice_rows);
  }
  int radius = 1;
};

class EqStage final : public Stage {
 public:
  Status run(const Frame& in, Frame& out, ThreadPool* pool, int slice_rows) const override {
    return run_sliced(
        pool, out.height(), pixfmt_row_alignment(out.format()),
        [&](RowRange r) { return brightness_contrast(in, out, brightness, contrast_q8, r); },
        slice_rows);
  }
  int brightness = 0;
  int contrast_q8 = 256;
};

Status fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return Status::invalid_argument;
}

const std::string* find_arg(const FilterSpec& f, std::string_view key) {
  for (const auto& kv : f.args) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

bool parse_int(const std::string& s, int& out) {
  const char* end = s.data() + s.size();
  const auto r = std::from_chars(s.data(), end, out);
  return r.ec == std::errc() && r.ptr == end;
}

bool parse_real(const std::string& s, double& out) {
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return !s.empty() && end == s.c_str() + s.size() && std::isfinite(out);
}

// Rounds num / den to nearest, then to a multiple of `multiple`.
int scaled_extent(std::int64_t num, std::int64_t den, int multiple) {
  const std::int64_t v = (num * 2 + den) / (2 * den);
  const std::int64_t m = (v + multiple / 2) / multiple * multiple;
  return static_cast<int>(std::max<std::int64_t>(m, multiple));
}

// ffmpeg size rules: 0 keeps the input size, -N keeps the aspect ratio and
// rounds to a multiple of N, both negative keeps the input size.
Status resolve_size(const std::string& ws, const std::string& hs, int in_w, int in_h, int& out_w,
                    int& out_h, std::string* error) {
  int w = 0;
  int h = 0;
  if (!parse_int(ws, w) || !parse_int(hs, h) || std::abs(w) > kMaxDimension ||
      std::abs(h) > kMaxDimension) {
    return fail(error, "scale: size '" + ws + ":" + hs + "' is not a pair of integers");
  }
  if (w == 0) w = in_w;
  if (h == 0) h = in_h;
  if (w < 0 && h < 0) {
    w = in_w;
    h = in_h;
  } else if (w < 0) {
    w = scaled_extent(static_cast<std::int64_t>(h) * in_w, in_h, -w);
  } else if (h < 0) {
    h = scaled_extent(static_cast<std::int64_t>(w) * in_h, in_w, -h);
  }
  if (w < 1 || h < 1 || w > kMaxDimension || h > kMaxDimension) {
    return fail(error, "scale: resulting size " + std::to_string(w) + "x" + std::to_string(h) +
                           " is out of range");
  }
  out_w = w;
  out_h = h;
  return Status::ok;
}

Status build_scale(const FilterSpec& f, const VideoInfo& cur, std::unique_ptr<Stage>& stage,
                   std::string* error) {
  int w = 0;
  int h = 0;
  if (const Status s = resolve_size(*find_arg(f, "w"), *find_arg(f, "h"), cur.width, cur.height,
                                    w, h, error);
      s != Status::ok) {
    return s;
  }
  ScaleFilter filter = ScaleFilter::bilinear;
  if (const std::string* flags = find_arg(f, "flags")) {
    if (*flags == "area") {
      filter = ScaleFilter::area;
    } else if (*flags != "bilinear") {
      return fail(error, "scale: unknown flags '" + *flags + "' (use bilinear or area)");
    }
  }
  if (w == cur.width && h == cur.height) return Status::ok;  // no-op, no stage

  auto st = std::make_unique<ScaleStage>();
  if (const Status s = Scaler::create(cur.format, cur.width, cur.height, w, h, filter, st->scaler,
                                      cur.chroma_loc);
      s != Status::ok) {
    return fail(error, "scale: cannot scale " + std::string(pixfmt_name(cur.format)) + " " +
                           std::to_string(cur.width) + "x" + std::to_string(cur.height) +
                           " to " + std::to_string(w) + "x" + std::to_string(h) + ": " +
                           std::string(status_name(s)));
  }
  st->filter = filter;
  st->input = cur;
  st->output = cur;
  st->output.width = w;
  st->output.height = h;
  stage = std::move(st);
  return Status::ok;
}

Status build_format(const FilterSpec& f, const VideoInfo& cur, std::unique_ptr<Stage>& stage,
                    std::string* error) {
  const std::string& name = *find_arg(f, "pix_fmt");
  const auto target = parse_pixfmt(name);
  if (!target) return fail(error, "format: unknown pixel format '" + name + "'");

  auto st = std::make_unique<ConvertStage>();
  st->opts.matrix = cur.matrix;
  st->opts.range = cur.range;
  st->opts.chroma_loc = cur.chroma_loc;
  if (const std::string* v = find_arg(f, "matrix")) {
    const auto m = parse_color_matrix(*v);
    if (!m) return fail(error, "format: unknown matrix '" + *v + "' (use bt601 or bt709)");
    st->opts.matrix = *m;
  }
  if (const std::string* v = find_arg(f, "range")) {
    const auto r = parse_color_range(*v);
    if (!r) return fail(error, "format: unknown range '" + *v + "' (use limited or full)");
    st->opts.range = *r;
  }
  if (const std::string* v = find_arg(f, "chroma")) {
    if (*v == "nearest") {
      st->opts.chroma = ChromaMode::nearest;
    } else if (*v == "bilinear") {
      st->opts.chroma = ChromaMode::bilinear;
    } else {
      return fail(error, "format: unknown chroma '" + *v + "' (use nearest or bilinear)");
    }
  }
  if (const std::string* v = find_arg(f, "chroma_loc")) {
    if (*v == "left") {
      st->opts.chroma_loc = ChromaLocation::left;
    } else if (*v == "center") {
      st->opts.chroma_loc = ChromaLocation::center;
    } else {
      return fail(error, "format: unknown chroma_loc '" + *v + "' (use left or center)");
    }
  }
  if (!convert_supported(cur.format, *target)) {
    return fail(error, "format: cannot convert " + std::string(pixfmt_name(cur.format)) +
                           " to " + std::string(pixfmt_name(*target)));
  }
  st->output = cur;
  st->output.format = *target;
  if (!pixfmt_info(cur.format).is_yuv && pixfmt_info(*target).is_yuv) {
    st->output.matrix = st->opts.matrix;
    st->output.range = st->opts.range;
    st->output.chroma_loc = st->opts.chroma_loc;
  }
  stage = std::move(st);
  return Status::ok;
}

Status build_blur(const FilterSpec& f, const VideoInfo& cur, std::unique_ptr<Stage>& stage,
                  std::string* error) {
  int size = 0;
  if (!parse_int(*find_arg(f, "size"), size) || (size != 3 && size != 5)) {
    return fail(error, "blur: size must be 3 or 5, got '" + *find_arg(f, "size") + "'");
  }
  auto st = std::make_unique<BlurStage>();
  st->radius = (size - 1) / 2;
  st->output = cur;
  stage = std::move(st);
  return Status::ok;
}

Status build_eq(const FilterSpec& f, const VideoInfo& cur, std::unique_ptr<Stage>& stage,
                std::string* error) {
  auto st = std::make_unique<EqStage>();
  if (const std::string* v = find_arg(f, "brightness")) {
    if (!parse_int(*v, st->brightness) || st->brightness < -255 || st->brightness > 255) {
      return fail(error, "eq: brightness must be an integer in [-255, 255], got '" + *v + "'");
    }
  }
  if (const std::string* v = find_arg(f, "contrast")) {
    double c = 0;
    if (!parse_real(*v, c) || c < 0 || c > 4) {
      return fail(error, "eq: contrast must be a number in [0, 4], got '" + *v + "'");
    }
    st->contrast_q8 = static_cast<int>(std::lround(c * 256));
  }
  st->output = cur;
  stage = std::move(st);
  return Status::ok;
}

// Replaces the last stage (a scale) and `stage` (a yuv to rgb convert) with
// one ScaleConvert when they are adjacent. On success `stage` is the fused
// stage and the scale stage is gone from `stages`. A format stage that asks
// for nearest chroma keeps the two-pass path, which is where nearest applies.
void fuse_scale_convert(std::vector<std::unique_ptr<Stage>>& stages,
                        std::unique_ptr<Stage>& stage, FusedChroma chroma) {
  if (stages.empty()) return;
  const auto* sc = dynamic_cast<const ScaleStage*>(stages.back().get());
  const auto* cv = dynamic_cast<const ConvertStage*>(stage.get());
  if (sc == nullptr || cv == nullptr || cv->opts.chroma_loc != sc->input.chroma_loc) return;
  auto fused = std::make_unique<ScaleConvertStage>();
  if (ScaleConvert::create(sc->input.format, sc->input.width, sc->input.height,
                           cv->output.format, sc->output.width, sc->output.height, sc->filter,
                           cv->opts, fused->fused,
                           cv->opts.chroma == ChromaMode::nearest ? FusedChroma::two_pass
                                                                  : chroma) != Status::ok) {
    return;  // not a yuv to rgb pair: keep the two stages
  }
  fused->output = cv->output;
  stage = std::move(fused);
  stages.pop_back();
}

}  // namespace

struct Pipeline::Impl {
  VideoInfo input;
  VideoInfo output;
  int slice_rows = 32;
  std::unique_ptr<ThreadPool> pool;  // null: run inline
  std::vector<std::unique_ptr<Stage>> stages;
  std::vector<Frame> buffers;  // output of every stage but the last
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::~Pipeline() = default;

Status Pipeline::create(std::string_view chain, const VideoInfo& input,
                        const PipelineOptions& opts, std::unique_ptr<Pipeline>& out,
                        std::string* error) {
  if (input.width < 1 || input.height < 1 || input.width > kMaxDimension ||
      input.height > kMaxDimension) {
    return fail(error, "input size " + std::to_string(input.width) + "x" +
                           std::to_string(input.height) + " is out of range");
  }
  std::vector<FilterSpec> specs;
  if (const Status s = parse_filter_chain(chain, specs, error); s != Status::ok) return s;

  auto impl = std::make_unique<Impl>();
  impl->input = input;
  impl->slice_rows = std::max(opts.slice_rows, 1);
  VideoInfo cur = input;
  for (const FilterSpec& f : specs) {
    std::unique_ptr<Stage> stage;
    Status s = Status::ok;
    if (f.name == "scale") {
      s = build_scale(f, cur, stage, error);
    } else if (f.name == "format") {
      s = build_format(f, cur, stage, error);
      if (s == Status::ok && stage) fuse_scale_convert(impl->stages, stage, opts.fused_chroma);
    } else if (f.name == "blur") {
      s = build_blur(f, cur, stage, error);
    } else {
      s = build_eq(f, cur, stage, error);
    }
    if (s != Status::ok) return s;
    if (stage) {
      cur = stage->output;
      impl->stages.push_back(std::move(stage));
    }
  }
  if (impl->stages.empty()) {
    auto copy = std::make_unique<ConvertStage>();  // same-format convert is a copy
    copy->output = cur;
    impl->stages.push_back(std::move(copy));
  }
  impl->output = cur;

  impl->buffers.resize(impl->stages.size() - 1);
  for (std::size_t i = 0; i + 1 < impl->stages.size(); ++i) {
    const VideoInfo& o = impl->stages[i]->output;
    if (Frame::allocate(o.format, o.width, o.height, impl->buffers[i]) != Status::ok) {
      return Status::out_of_memory;
    }
  }
  if (opts.threads != 1) impl->pool = std::make_unique<ThreadPool>(opts.threads);

  out.reset(new Pipeline(std::move(impl)));
  return Status::ok;
}

const VideoInfo& Pipeline::output_info() const noexcept { return impl_->output; }

Status Pipeline::process(const Frame& in, Frame& out) {
  Impl& p = *impl_;
  if (in.empty() || in.format() != p.input.format || in.width() != p.input.width ||
      in.height() != p.input.height || &in == &out) {
    return Status::invalid_argument;
  }
  if (const Status s = out.ensure(p.output.format, p.output.width, p.output.height);
      s != Status::ok) {
    return s;
  }
  const std::size_t last = p.stages.size() - 1;
  for (std::size_t i = 0; i <= last; ++i) {
    const Frame& src = i == 0 ? in : p.buffers[i - 1];
    Frame& dst = i == last ? out : p.buffers[i];
    if (const Status s = p.stages[i]->run(src, dst, p.pool.get(), p.slice_rows); s != Status::ok) {
      return s;
    }
  }
  return Status::ok;
}

}  // namespace lanecast
