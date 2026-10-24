// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
#include <arm_neon.h>

#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {

void fill_neon(Table& t) { (void)t; }

}  // namespace lanecast::kernels::metrics
