// Raw video IO: Y4M, raw planar/packed, PNM (PPM/PGM).
//
// Containers:
//   y4m  "YUV4MPEG2" header (W, H, F, A, I, C, XCOLORRANGE); C420jpeg,
//        C420mpeg2, C420paldv, C420 and no C tag read as yuv420p; Cmono as
//        gray8. Writer supports yuv420p and gray8; I must be p (progressive).
//   raw  planes back to back, rows tightly packed (no padding), plane order
//        of the format. Reading needs width, height and format in options.
//   pnm  P5 (gray8) and P6 (rgb24), maxval 255. Several images may be
//        concatenated in one file; the reader returns them in order and the
//        writer appends one image per write().
// Path "-" is stdin for readers and stdout for writers.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "lanecast/frame.hpp"
#include "lanecast/status.hpp"

namespace lanecast {

enum class Container : std::uint8_t { y4m, raw, pnm };

// .y4m -> y4m; .yuv .raw .rgb .nv12 -> raw; .ppm .pgm .pnm -> pnm.
std::optional<Container> container_from_path(std::string_view path) noexcept;

class FrameReader {
 public:
  virtual ~FrameReader() = default;
  virtual const VideoInfo& info() const = 0;
  // Fills `frame`, reusing its buffer when the shape matches (Frame::ensure).
  // Returns end_of_stream after the last frame.
  virtual Status read(Frame& frame) = 0;
};

class FrameWriter {
 public:
  virtual ~FrameWriter() = default;
  virtual Status write(const Frame& frame) = 0;  // frame must match info
  virtual Status finish() = 0;                   // flush and close
};

struct ReaderOptions {
  std::optional<Container> container;  // default: from the path extension
  VideoInfo raw;                       // required for raw input
};

Status open_reader(const std::string& path, const ReaderOptions& opts,
                   std::unique_ptr<FrameReader>& out);
Status open_writer(const std::string& path, const VideoInfo& info,
                   std::optional<Container> container, std::unique_ptr<FrameWriter>& out);

}  // namespace lanecast
