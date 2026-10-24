// filter suite. Owned by the filter unit.
// Required: build_levels_lut known values; blend at alpha 0, 255 and mid; blur on constant planes is identity, edge replication, radius 1 and 2 known values; RowRange slices; CHECK_ALL_ISAS_MATCH for every kernel over test::test_sizes().
#include "test_support.hpp"
