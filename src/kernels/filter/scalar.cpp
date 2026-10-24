// Scalar reference kernels for filter. Defines the behavior every SIMD tier
// must match bit for bit. Pattern: see src/kernels/convert/scalar.cpp.
#include "kernels/filter/kernels.hpp"

namespace lanecast::kernels::filter {

void fill_scalar(Table& t) { (void)t; }

}  // namespace lanecast::kernels::filter
