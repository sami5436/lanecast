#include "core/dispatch.hpp"
#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {
namespace {

const detail::Dispatcher<Table>& dispatcher() {
  static const detail::Dispatcher<Table> d({
      .scalar = &fill_scalar,
      .neon = LANECAST_IF_NEON(fill_neon),
      .sse41 = LANECAST_IF_SSE41(fill_sse41),
      .avx2 = LANECAST_IF_AVX2(fill_avx2),
  });
  return d;
}

}  // namespace

const Table& table(Isa isa) { return dispatcher().get(isa); }
const Table& active_table() { return dispatcher().active(); }

}  // namespace lanecast::kernels::scale
