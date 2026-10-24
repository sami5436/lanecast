// Front end: validation, format routing, row loop. Calls kernels through
// active_table() only.
#include "lanecast/convert.hpp"

#include "kernels/convert/kernels.hpp"

namespace lanecast {

bool convert_supported(PixelFormat src, PixelFormat dst) noexcept {
  (void)src;
  (void)dst;
  return false;
}

Status convert(const Frame& src, Frame& dst, const ConvertOptions& opts, RowRange rows) {
  (void)src;
  (void)dst;
  (void)opts;
  (void)rows;
  return Status::not_implemented;
}

}  // namespace lanecast
