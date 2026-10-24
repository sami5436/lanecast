// fused suite. Owned by the fused unit.
// Required: ScaleConvert output equals scale() then convert() bit for bit over sizes, filters, matrices, ranges and dst formats; sliced and threaded runs equal the single call; CHECK_ALL_ISAS_MATCH.
#include "test_support.hpp"
