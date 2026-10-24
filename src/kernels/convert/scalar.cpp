// Scalar reference kernels for convert. This file defines the behavior every
// SIMD tier must match bit for bit (formulas in include/lanecast/convert.hpp).
//
// Pattern for adding a kernel:
//   namespace {
//   void yuv420_to_rgb24_row(const std::uint8_t* y, const std::uint8_t* u,
//                            const std::uint8_t* v, std::uint8_t* dst, int width,
//                            const YuvToRgbCoeffs& c) { ... }
//   }  // namespace
//   void fill_scalar(Table& t) { t.yuv420_to_packed[kRgb24] = &yuv420_to_rgb24_row; }
#include "kernels/convert/kernels.hpp"

namespace lanecast::kernels::convert {

void fill_scalar(Table& t) { (void)t; }

}  // namespace lanecast::kernels::convert
