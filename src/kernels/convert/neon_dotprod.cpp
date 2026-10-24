// NEON with dotprod (armv8.2). Store only the kernels this tier specializes;
// others are inherited from neon.
#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {

void fill_neon_dotprod(Table& t) { fill_neon_dotprod_yuv2rgb(t); }

}  // namespace lanecast::kernels::convert
