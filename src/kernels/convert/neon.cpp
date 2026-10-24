// NEON (AArch64 baseline).
// Store only the kernels this tier specializes; others are inherited.
#include <arm_neon.h>

#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {

void fill_neon(Table& t) { (void)t; }

}  // namespace lanecast::kernels::convert
