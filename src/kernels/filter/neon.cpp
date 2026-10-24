// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
#include <arm_neon.h>

#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {

void fill_neon(Table& t) { (void)t; }

}  // namespace lanecast::kernels::filter
