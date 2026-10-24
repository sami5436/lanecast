// SSE4.1, compiled with -msse4.1.
// Store only the kernels this tier specializes; others are inherited.
#include <smmintrin.h>

#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {

void fill_sse41(Table& t) { (void)t; }

}  // namespace lanecast::kernels::convert
