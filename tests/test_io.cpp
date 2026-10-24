// io suite. Owned by the io unit.
// Required: y4m, raw and pnm write-then-read round trips (yuv420p, gray8, rgb24, odd sizes), header parsing (C tags, XCOLORRANGE, missing tags), truncated files return parse_error or io_error, end_of_stream after the last frame, container_from_path.
#include "test_support.hpp"
