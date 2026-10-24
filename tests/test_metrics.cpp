// metrics suite. Owned by the metrics unit.
// Required: sse, sad and psnr known values; ssim of identical planes is 1;
// ssim matches a straightforward double reference of the spec; planes under
// 8x8 rejected; isa_value_mismatch over test::test_sizes() for every kernel.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "lanecast/metrics.hpp"
#include "test_support.hpp"

using namespace lanecast;

TEST(psnr_from_sse_formula) {
  CHECK(std::isinf(psnr_from_sse(0, 100)));
  CHECK(std::abs(psnr_from_sse(65025, 1) - 0.0) < 1e-12);
  CHECK(std::abs(psnr_from_sse(1, 1) - 20.0 * std::log10(255.0)) < 1e-9);
}

namespace {

Frame make_frame(PixelFormat fmt, int w, int h) {
  Frame f;
  REQUIRE_OK(Frame::allocate(fmt, w, h, f));
  return f;
}

void fill_plane(const Plane& p, int value) {
  for (int y = 0; y < p.height; ++y) {
    std::memset(p.row(y), value, static_cast<std::size_t>(p.row_bytes()));
  }
}

// Direct per-pixel double implementation of the SSIM spec for one plane.
double ssim_reference(const Plane& a, const Plane& b) {
  const int bw = a.width / 4;
  const int bh = a.height / 4;
  double total = 0.0;
  for (int wy = 0; wy + 1 < bh; ++wy) {
    for (int wx = 0; wx + 1 < bw; ++wx) {
      double s1 = 0, s2 = 0, ss = 0, s12 = 0;
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          const double pa = a.row(4 * wy + y)[4 * wx + x];
          const double pb = b.row(4 * wy + y)[4 * wx + x];
          s1 += pa;
          s2 += pb;
          ss += pa * pa + pb * pb;
          s12 += pa * pb;
        }
      }
      const double vars = ss * 64 - s1 * s1 - s2 * s2;
      const double covar = s12 * 64 - s1 * s2;
      total += (2 * s1 * s2 + 416) * (2 * covar + 235963) /
               ((s1 * s1 + s2 * s2 + 416) * (vars + 235963));
    }
  }
  return total / ((bw - 1) * (bh - 1));
}

}  // namespace

TEST(psnr_identical_is_infinite) {
  const Frame a = test::random_frame(PixelFormat::yuv420p, 17, 9, 1);
  PsnrResult r;
  REQUIRE_OK(compute_psnr(a, a, r));
  CHECK(r.planes == 3);
  for (int p = 0; p < 3; ++p) {
    CHECK(r.sse[p] == 0);
    CHECK(std::isinf(r.psnr[p]));
  }
  CHECK(std::isinf(r.average));
}

TEST(psnr_constant_offset_is_analytic) {
  // Every sample differs by 5: mse 25 in every plane, so every PSNR and the
  // average equal 10 log10(255^2 / 25). Odd size: chroma is 4x3.
  Frame a = make_frame(PixelFormat::yuv420p, 7, 5);
  Frame b = make_frame(PixelFormat::yuv420p, 7, 5);
  for (int p = 0; p < 3; ++p) {
    fill_plane(a.plane(p), 100);
    fill_plane(b.plane(p), 105);
  }
  PsnrResult r;
  REQUIRE_OK(compute_psnr(a, b, r));
  const double expect = 10.0 * std::log10(255.0 * 255.0 / 25.0);
  CHECK(r.samples[0] == 35 && r.samples[1] == 12 && r.samples[2] == 12);
  CHECK(r.sse[0] == 35 * 25 && r.sse[1] == 12 * 25);
  for (int p = 0; p < 3; ++p) CHECK(std::abs(r.psnr[p] - expect) < 1e-12);
  CHECK(std::abs(r.average - expect) < 1e-12);
}

TEST(psnr_packed_and_pooled_average) {
  Frame a = make_frame(PixelFormat::rgba, 3, 2);
  Frame b = make_frame(PixelFormat::rgba, 3, 2);
  b.plane(0).row(1)[5] = 3;  // one sample off by 3
  PsnrResult r;
  REQUIRE_OK(compute_psnr(a, b, r));
  CHECK(r.planes == 1 && r.samples[0] == 24 && r.sse[0] == 9);
  CHECK(std::abs(r.psnr[0] - 10.0 * std::log10(255.0 * 255.0 * 24 / 9)) < 1e-12);
  // yuv420p: average pools sse and samples over planes (16 + 4 + 4 samples).
  Frame ya = make_frame(PixelFormat::yuv420p, 4, 4);
  Frame yb = make_frame(PixelFormat::yuv420p, 4, 4);
  yb.plane(0).row(0)[0] = 4;  // sse 16
  REQUIRE_OK(compute_psnr(ya, yb, r));
  CHECK(std::isinf(r.psnr[1]));
  CHECK(std::abs(r.average - 10.0 * std::log10(255.0 * 255.0 * 24 / 16)) < 1e-12);
}

TEST(sad_known_values) {
  Frame a = make_frame(PixelFormat::yuv420p, 5, 3);
  Frame b = make_frame(PixelFormat::yuv420p, 5, 3);
  a.plane(0).row(0)[0] = 10;
  b.plane(0).row(0)[0] = 3;    // 7
  a.plane(0).row(2)[4] = 1;
  b.plane(0).row(2)[4] = 250;  // 249
  b.plane(2).row(1)[2] = 9;    // 9
  std::uint64_t sad = 0;
  REQUIRE_OK(compute_sad(a, b, sad));
  CHECK(sad == 7 + 249 + 9);
  REQUIRE_OK(compute_sad(a, a, sad));
  CHECK(sad == 0);
}

TEST(ssim_identical_is_one) {
  for (const PixelFormat fmt : {PixelFormat::gray8, PixelFormat::yuv420p}) {
    const Frame a = test::random_frame(fmt, 40, 33, 2);
    SsimResult r;
    REQUIRE_OK(compute_ssim(a, a, r));
    for (int p = 0; p < r.planes; ++p) CHECK(std::abs(r.ssim[p] - 1.0) < 1e-12);
    CHECK(std::abs(r.average - 1.0) < 1e-12);
  }
}

TEST(ssim_matches_double_reference) {
  for (const auto& [w, h] :
       {std::pair{16, 16}, std::pair{17, 19}, std::pair{33, 20}, std::pair{64, 64}}) {
    const Frame a = test::random_frame(PixelFormat::yuv420p, w, h, 3);
    Frame b = make_frame(PixelFormat::yuv420p, w, h);
    test::Rng rng(11);
    for (int p = 0; p < 3; ++p) {  // b = a plus noise
      const Plane pa = a.plane(p);
      for (int y = 0; y < pa.height; ++y) {
        for (int x = 0; x < pa.width; ++x) {
          b.plane(p).row(y)[x] =
              static_cast<std::uint8_t>(std::clamp(pa.row(y)[x] + rng.range(-20, 20), 0, 255));
        }
      }
    }
    SsimResult r;
    REQUIRE_OK(compute_ssim(a, b, r));
    double weighted = 0, samples = 0;
    for (int p = 0; p < 3; ++p) {
      const double ref = ssim_reference(a.plane(p), b.plane(p));
      CHECK(std::abs(r.ssim[p] - ref) < 1e-12);
      const double n = static_cast<double>(a.plane(p).width) * a.plane(p).height;
      weighted += ref * n;
      samples += n;
    }
    CHECK(std::abs(r.average - weighted / samples) < 1e-12);
  }
}

TEST(ssim_rejects_bad_arguments) {
  SsimResult r;
  const Frame small = make_frame(PixelFormat::gray8, 8, 7);
  CHECK(compute_ssim(small, small, r) == Status::invalid_argument);
  const Frame chroma_small = make_frame(PixelFormat::yuv420p, 15, 16);  // chroma 8x8 is ok
  CHECK(compute_ssim(chroma_small, chroma_small, r) == Status::ok);
  const Frame chroma_tiny = make_frame(PixelFormat::yuv420p, 14, 16);  // chroma 7x8
  CHECK(compute_ssim(chroma_tiny, chroma_tiny, r) == Status::invalid_argument);
  const Frame rgb = make_frame(PixelFormat::rgb24, 16, 16);
  CHECK(compute_ssim(rgb, rgb, r) == Status::unsupported_format);
  const Frame a = make_frame(PixelFormat::gray8, 16, 16);
  const Frame b = make_frame(PixelFormat::gray8, 16, 17);
  PsnrResult pr;
  std::uint64_t sad = 0;
  CHECK(compute_ssim(a, b, r) == Status::invalid_argument);
  CHECK(compute_psnr(a, b, pr) == Status::invalid_argument);
  CHECK(compute_sad(a, b, sad) == Status::invalid_argument);
  Frame empty;
  CHECK(compute_psnr(empty, empty, pr) == Status::invalid_argument);
}

TEST(all_isas_match) {
  for (const PixelFormat fmt : {PixelFormat::gray8, PixelFormat::yuv420p, PixelFormat::nv12,
                                PixelFormat::rgb24, PixelFormat::rgba}) {
    for (const auto& [w, h] : test::test_sizes()) {
      const Frame a = test::random_frame(fmt, w, h, 70);
      const Frame b = test::random_frame(fmt, w, h, 71);
      CHECK_MSG(test::isa_value_mismatch([&] {
                  PsnrResult r;
                  REQUIRE_OK(compute_psnr(a, b, r));
                  return std::vector<std::uint64_t>(r.sse, r.sse + kMaxPlanes);
                }).empty(),
                "psnr sse");
      CHECK_MSG(test::isa_value_mismatch([&] {
                  std::uint64_t s = 0;
                  REQUIRE_OK(compute_sad(a, b, s));
                  return s;
                }).empty(),
                "sad");
      if (fmt != PixelFormat::gray8 && fmt != PixelFormat::yuv420p) continue;
      if (plane_width(fmt, fmt == PixelFormat::gray8 ? 0 : 1, w) < 8 ||
          plane_height(fmt, fmt == PixelFormat::gray8 ? 0 : 1, h) < 8) {
        continue;
      }
      CHECK_MSG(test::isa_value_mismatch([&] {
                  SsimResult r;
                  REQUIRE_OK(compute_ssim(a, b, r));
                  return std::vector<double>(r.ssim, r.ssim + kMaxPlanes);
                }).empty(),
                "ssim");
    }
  }
}
