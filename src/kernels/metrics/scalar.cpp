// Scalar reference kernels for metrics. Defines the behavior every SIMD tier
// must match bit for bit. Pattern: see src/kernels/convert/scalar.cpp.
#include "kernels/metrics/kernels.hpp"

namespace lanecast::kernels::metrics {

void fill_scalar(Table& t) { (void)t; }

}  // namespace lanecast::kernels::metrics
