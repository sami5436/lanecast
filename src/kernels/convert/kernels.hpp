// Internal row-kernel contract for convert. Owned by the convert unit.
//
// SIMD TU rules (all kernel directories):
//   - Include only <cstddef>, <cstdint>, intrinsic headers and this file.
//     Do not call inline functions or templates from other headers inside an
//     ISA file: the linker may keep the copy compiled with -mavx2 (or
//     +dotprod) for every caller and crash CPUs without that ISA.
//   - Put helpers in an anonymous namespace in the ISA file.
//   - Rows satisfy the frame padding guarantee (lanecast/frame.hpp): reads
//     and writes up to 64 bytes past the last visible byte of a row are
//     allowed; never touch other rows.
//   - Output must equal scalar.cpp bit for bit (formulas in convert.hpp).
#pragma once

#include <cstddef>
#include <cstdint>

#include "lanecast/convert.hpp"

namespace lanecast {
enum class Isa : std::uint8_t;
}

namespace lanecast::kernels::convert {

// Packed RGB layouts, used to index the tables below.
enum Packed : int { kRgb24 = 0, kRgba = 1, kBgra = 2, kPackedCount = 3 };

// One output row of yuv420p -> packed. u, v are the chroma row for this luma row.
using Yuv420ToPackedRowFn = void (*)(const std::uint8_t* y, const std::uint8_t* u,
                                     const std::uint8_t* v, std::uint8_t* dst, int width,
                                     const YuvToRgbCoeffs& c);
// One output row of nv12 -> packed. uv is the interleaved chroma row.
using Nv12ToPackedRowFn = void (*)(const std::uint8_t* y, const std::uint8_t* uv,
                                   std::uint8_t* dst, int width, const YuvToRgbCoeffs& c);
// Two source rows -> two luma rows and one chroma row. For the last row of
// an odd height, src1 == src0 and y1 == y0.
using PackedToYuv420RowFn = void (*)(const std::uint8_t* src0, const std::uint8_t* src1,
                                     std::uint8_t* y0, std::uint8_t* y1, std::uint8_t* u,
                                     std::uint8_t* v, int width, const RgbToYuvCoeffs& k);
using PackedToNv12RowFn = void (*)(const std::uint8_t* src0, const std::uint8_t* src1,
                                   std::uint8_t* y0, std::uint8_t* y1, std::uint8_t* uv,
                                   int width, const RgbToYuvCoeffs& k);
using SwizzleRowFn = void (*)(const std::uint8_t* src, std::uint8_t* dst, int width);
using InterleaveUvRowFn = void (*)(const std::uint8_t* u, const std::uint8_t* v,
                                   std::uint8_t* uv, int chroma_width);
using DeinterleaveUvRowFn = void (*)(const std::uint8_t* uv, std::uint8_t* u, std::uint8_t* v,
                                     int chroma_width);
using PackedToGrayRowFn = void (*)(const std::uint8_t* src, std::uint8_t* gray, int width,
                                   const RgbToYuvCoeffs& k);
using GrayToPackedRowFn = void (*)(const std::uint8_t* gray, std::uint8_t* dst, int width);

// A null entry means "not implemented"; the front end returns
// Status::not_implemented for it.
struct Table {
  Yuv420ToPackedRowFn yuv420_to_packed[kPackedCount] = {};
  Nv12ToPackedRowFn nv12_to_packed[kPackedCount] = {};
  PackedToYuv420RowFn packed_to_yuv420[kPackedCount] = {};
  PackedToNv12RowFn packed_to_nv12[kPackedCount] = {};
  SwizzleRowFn swizzle[kPackedCount][kPackedCount] = {};  // [src][dst], src != dst
  InterleaveUvRowFn interleave_uv = nullptr;
  DeinterleaveUvRowFn deinterleave_uv = nullptr;
  PackedToGrayRowFn packed_to_gray[kPackedCount] = {};
  GrayToPackedRowFn gray_to_packed[kPackedCount] = {};
};

void fill_scalar(Table& t);
void fill_neon(Table& t);
void fill_sse41(Table& t);
void fill_avx2(Table& t);

// Table for a tier (clamped to what is supported). Built once, thread-safe.
const Table& table(Isa isa);
const Table& active_table();

}  // namespace lanecast::kernels::convert
