// convert suite. Owned by the convert unit.
// Required: known values (black, white, gray, primaries) per matrix and
// range, every supported pair, RowRange slices equal whole-frame output, and
// CHECK_ALL_ISAS_MATCH over test::test_sizes() for every kernel.
#include <cmath>
#include <cstdlib>

#include "lanecast/convert.hpp"
#include "test_support.hpp"

using namespace lanecast;

TEST(coefficients_match_spec) {
  struct M {
    ColorMatrix m;
    double kr, kb;
  };
  for (const M mat : {M{ColorMatrix::bt601, 0.299, 0.114}, M{ColorMatrix::bt709, 0.2126, 0.0722}}) {
    for (const ColorRange r : {ColorRange::limited, ColorRange::full}) {
      const bool lim = r == ColorRange::limited;
      const double kg = 1.0 - mat.kr - mat.kb;
      const double ys = lim ? 255.0 / 219.0 : 1.0, cs = lim ? 255.0 / 224.0 : 1.0;
      const YuvToRgbCoeffs c = yuv_to_rgb_coeffs(mat.m, r);
      const auto q13 = [](double v) { return static_cast<int>(std::lround(v * 8192.0)); };
      CHECK_EQ(c.y_offset, lim ? 16 : 0);
      CHECK_EQ(c.cy, q13(ys));
      CHECK_EQ(c.crv, q13(2 * (1 - mat.kr) * cs));
      CHECK_EQ(c.cbu, q13(2 * (1 - mat.kb) * cs));
      CHECK_EQ(c.cgu, q13(2 * (1 - mat.kb) * mat.kb / kg * cs));
      CHECK_EQ(c.cgv, q13(2 * (1 - mat.kr) * mat.kr / kg * cs));

      const RgbToYuvCoeffs k = rgb_to_yuv_coeffs(mat.m, r);
      const double ys2 = lim ? 219.0 / 255.0 : 1.0, cs2 = lim ? 224.0 / 255.0 : 1.0;
      const auto near_q15 = [](int got, double v) {
        return std::abs(got - static_cast<int>(std::lround(v * 32768.0))) <= 1;
      };
      CHECK_EQ(k.y_offset, lim ? 16 : 0);
      CHECK(near_q15(k.yr, mat.kr * ys2));
      CHECK(near_q15(k.yg, kg * ys2));
      CHECK(near_q15(k.yb, mat.kb * ys2));
      CHECK(near_q15(k.ur, -mat.kr / (2 * (1 - mat.kb)) * cs2));
      CHECK(near_q15(k.ub, 0.5 * cs2));
      CHECK(near_q15(k.vb, -mat.kb / (2 * (1 - mat.kr)) * cs2));
      CHECK_EQ(k.yr + k.yg + k.yb, static_cast<int>(std::lround(32768.0 * ys2)));
      CHECK_EQ(k.ur + k.ug + k.ub, 0);
      CHECK_EQ(k.vr + k.vg + k.vb, 0);
      CHECK_EQ(k.ub, k.vr);
    }
  }
}
