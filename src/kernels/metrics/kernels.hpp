// Internal row-kernel contract for metrics. Owned by the metrics unit.
// SIMD TU rules: see src/kernels/convert/kernels.hpp (same rules apply).
// Formulas: include/lanecast/metrics.hpp. Kernels produce exact integer
// sums; floating point math stays in the front end (metrics.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

namespace lanecast {
enum class Isa : std::uint8_t;
}

namespace lanecast::kernels::metrics {

// sum (a[i] - b[i])^2 for i in [0, count)
using SseRowFn = std::uint64_t (*)(const std::uint8_t* a, const std::uint8_t* b, int count);
// sum |a[i] - b[i]| for i in [0, count)
using SadRowFn = std::uint64_t (*)(const std::uint8_t* a, const std::uint8_t* b, int count);
// For "blocks" horizontally adjacent 4x4 blocks starting at a and b (4 rows
// each): sums[i] = {sum a, sum b, sum a^2 + sum b^2, sum a*b}.
using Ssim4x4RowFn = void (*)(const std::uint8_t* a, std::ptrdiff_t a_stride,
                              const std::uint8_t* b, std::ptrdiff_t b_stride, int blocks,
                              std::int32_t (*sums)[4]);

struct Table {
  SseRowFn sse = nullptr;
  SadRowFn sad = nullptr;
  Ssim4x4RowFn ssim_4x4 = nullptr;
};

void fill_scalar(Table& t);
void fill_neon(Table& t);
void fill_neon_dotprod(Table& t);  // UDOT for sse, sad, ssim sums
void fill_sse41(Table& t);
void fill_avx2(Table& t);

const Table& table(Isa isa);
const Table& active_table();

}  // namespace lanecast::kernels::metrics
