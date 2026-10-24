// AVX2, compiled with -mavx2. Runs on top of the sse41 table.
// Store only the kernels this tier specializes; others are inherited.
#include <immintrin.h>

#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {

void fill_avx2(Table& t) { (void)t; }

}  // namespace lanecast::kernels::convert
