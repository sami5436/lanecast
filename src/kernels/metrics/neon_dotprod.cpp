// NEON with FEAT_DotProd, compiled with -march=armv8.2-a+dotprod. Runs on top of the neon table.
// Store only the kernels this tier specializes; others are inherited.
#include <arm_neon.h>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {

void fill_neon_dotprod(Table& t) { (void)t; }

}  // namespace lanecast::kernels::metrics
