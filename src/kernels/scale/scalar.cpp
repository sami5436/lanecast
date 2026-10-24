// Scalar reference kernels for scale. Defines the behavior every SIMD tier
// must match bit for bit. Pattern: see src/kernels/convert/scalar.cpp.
#include "kernels/scale/kernels.hpp"

namespace lanecast::kernels::scale {

void fill_scalar(Table& t) { (void)t; }

}  // namespace lanecast::kernels::scale
