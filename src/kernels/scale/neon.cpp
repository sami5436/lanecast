// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
#include <arm_neon.h>

#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {

void fill_neon(Table& t) { (void)t; }

}  // namespace lanecast::kernels::scale
