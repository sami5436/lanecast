// Internal row-kernel contract for filter. Owned by the filter unit.
// SIMD TU rules: see src/kernels/convert/kernels.hpp (same rules apply).
// Formulas: include/lanecast/filter.hpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lanecast {
enum class Isa : std::uint8_t;
}

namespace lanecast::kernels::filter {

// Straight-alpha "over" for 4-byte pixels, alpha at byte 3. In place on base.
using BlendRowFn = void (*)(const std::uint8_t* over, std::uint8_t* base, int pixels);
// Horizontal binomial pass. src points at pixel -radius of an edge-extended
// copy of the row (radius replicated pixels on each side, built by the front
// end), so the kernel has no edge branches. count = width * channels.
//   dst[i] = sum_k b_k * src[i + k * channels], k in [0, 2r]
using BlurHRowFn = void (*)(const std::uint8_t* src, std::uint16_t* dst, int count, int channels,
                            int radius);
// Vertical binomial pass over 2r + 1 rows of horizontal sums (already edge
// clamped by the front end). Sums fit uint16 (max 65280 + 128).
using BlurVRowFn = void (*)(const std::uint16_t* const* rows, std::uint8_t* dst, int count,
                            int radius);
// dst[i] = lut[src[i]] for i in [0, count). src may equal dst.
using LutRowFn = void (*)(const std::uint8_t* src, std::uint8_t* dst, int count,
                          const std::uint8_t* lut);
// Same for 4-byte pixels, byte 3 copied unchanged.
using LutSkipAlphaRowFn = void (*)(const std::uint8_t* src, std::uint8_t* dst, int pixels,
                                   const std::uint8_t* lut);

struct Table {
  BlendRowFn blend = nullptr;
  BlurHRowFn blur_h = nullptr;
  BlurVRowFn blur_v = nullptr;
  LutRowFn lut = nullptr;
  LutSkipAlphaRowFn lut_skip_alpha = nullptr;
};

void fill_scalar(Table& t);
void fill_neon(Table& t);
void fill_sse41(Table& t);
void fill_avx2(Table& t);

const Table& table(Isa isa);
const Table& active_table();

}  // namespace lanecast::kernels::filter
