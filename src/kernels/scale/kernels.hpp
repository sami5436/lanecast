// Internal row-kernel contract for scale. Owned by the scale unit.
// SIMD TU rules: see src/kernels/convert/kernels.hpp (same rules apply).
// Formulas: include/lanecast/scale.hpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lanecast {
enum class Isa : std::uint8_t;
}

namespace lanecast::kernels::scale {

// POD view of a ScaleFilterTable (no std::vector in ISA files). Rows are
// padded for SIMD: output x has taps weights at weights[x * stride]; stride
// is taps when taps <= 2, else taps rounded up to a multiple of 4. Padding
// weights are zero and 16 more zeros follow the last row, so a kernel may
// load whole vectors from any row.
//
// Optional block tables for 1 channel rows with 2..6 taps (tbl_blocks == 0
// when absent): the first tbl_blocks groups of 8 outputs have all their
// source bytes inside the 32 bytes at src + start[8 * b], which lets a kernel
// replace gathers with table lookups. Per block, tbl holds ceil(taps / 2)
// vectors of 16 byte indices (bytes 0..7: tap 2m of outputs 0..7, bytes 8..15:
// tap 2m + 1; odd tap counts leave the last half zero), and wt holds taps
// vectors of 8 weights (tap-major, outputs in lanes).
struct FilterView {
  int taps;
  int stride;
  const std::int32_t* start;    // dst_size entries
  const std::int16_t* weights;  // Q14
  int tbl_blocks = 0;
  const std::uint8_t* tbl = nullptr;
  const std::int16_t* wt = nullptr;
};

// Horizontal pass of one row with C interleaved channels (C = table index):
//   mid[x*C + c] = (sum_k w[x*stride + k] * src[(start[x] + k)*C + c] + 64) >> 7
// for x in [0, dst_width). mid rows are scratch with the same 64-byte slack.
using HScaleRowFn = void (*)(const std::uint8_t* src, std::int16_t* mid, int dst_width,
                             const FilterView& f);
// Vertical pass: dst[i] = clamp8((sum_k w[k] * rows[k][i] + (1 << 20)) >> 21)
// for i in [0, count), count = dst_width * C.
using VScaleRowFn = void (*)(const std::int16_t* const* rows, const std::int16_t* w, int taps,
                             std::uint8_t* dst, int count);

struct Table {
  HScaleRowFn hscale[5] = {};  // index by channel count 1..4; [0] unused
  VScaleRowFn vscale = nullptr;
};

void fill_scalar(Table& t);
void fill_neon(Table& t);
void fill_sse41(Table& t);
void fill_avx2(Table& t);

const Table& table(Isa isa);
const Table& active_table();

}  // namespace lanecast::kernels::scale
