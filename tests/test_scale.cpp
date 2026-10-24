// scale suite. Owned by the scale unit.
// Required: build_scale_filter invariants (weights sum to 16384, start + taps <= src) for bilinear and area over many size pairs; identity scale is exact; known 2x down / 2x up values; RowRange slices equal whole-frame output; CHECK_ALL_ISAS_MATCH for every format over test::test_sizes() x several target sizes.
#include "test_support.hpp"
