// pipeline suite. Owned by the pipeline unit.
// Required: ThreadPool runs every index exactly once for many counts and sizes; run_sliced covers [0, height) with aligned, disjoint slices; parse_filter_chain accepts the documented grammar and rejects bad input with a message; Pipeline output equals the same ops called directly, for threads = 1 and threads = 4.
#include "test_support.hpp"
