// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
#include <smmintrin.h>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {

void fill_sse41(Table& t) { (void)t; }

}  // namespace lanecast::kernels::scale
