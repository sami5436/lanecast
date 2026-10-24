// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
#include <smmintrin.h>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {

void fill_sse41(Table& t) { (void)t; }

}  // namespace lanecast::kernels::metrics
