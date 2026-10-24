// Skeleton tests: cpu tiers, dispatch layering, frame layout guarantee,
// scratch, and the ISA comparison helper itself.
#include <cstdint>
#include <cstdlib>
#include <thread>

#include "core/dispatch.hpp"
#include "core/scratch.hpp"
#include "lanecast/cpu.hpp"
#include "lanecast/frame.hpp"
#include "lanecast/pixfmt.hpp"
#include "lanecast/status.hpp"
#include "test_support.hpp"

using namespace lanecast;

// Must stay the first test in this file: it observes the initial active ISA
// before any test changes it.
TEST(env_override_sets_initial_isa) {
  const char* env = std::getenv("LANECAST_ISA");
  const auto requested = env != nullptr ? parse_isa(env) : std::nullopt;
  const Isa expected = requested ? clamp_isa(*requested) : best_isa();
  CHECK_EQ(active_isa(), expected);
}

TEST(isa_names_round_trip) {
  for (int i = 0; i < kIsaCount; ++i) {
    const auto isa = static_cast<Isa>(i);
    CHECK(parse_isa(isa_name(isa)) == isa);
  }
  CHECK(!parse_isa("avx512").has_value());
}

TEST(isa_chains) {
  CHECK(!isa_parent(Isa::scalar).has_value());
  CHECK(isa_parent(Isa::neon) == Isa::scalar);
  CHECK(isa_parent(Isa::neon_dotprod) == Isa::neon);
  CHECK(isa_parent(Isa::neon_i8mm) == Isa::neon_dotprod);
  CHECK(isa_parent(Isa::sse41) == Isa::scalar);
  CHECK(isa_parent(Isa::avx2) == Isa::sse41);
}

TEST(supported_isas_are_consistent) {
  const auto isas = supported_isas();
  REQUIRE(!isas.empty());
  CHECK_EQ(isas.front(), Isa::scalar);
  for (const Isa isa : isas) {
    CHECK(isa_compiled(isa));
    CHECK_EQ(clamp_isa(isa), isa);
    if (auto parent = isa_parent(isa)) CHECK(isa_supported(*parent));
  }
  CHECK(isa_supported(best_isa()));
#if defined(__aarch64__) && LANECAST_HAVE_NEON
  CHECK(isa_supported(Isa::neon));
  CHECK_EQ(clamp_isa(Isa::avx2), Isa::scalar);
#endif
#if defined(__x86_64__)
  CHECK(!isa_supported(Isa::neon));
  CHECK_EQ(clamp_isa(Isa::neon_dotprod), Isa::scalar);
#endif
}

TEST(set_active_isa_clamps_and_scope_restores) {
  const Isa before = active_isa();
  {
    test::ScopedIsa scope(Isa::scalar);
    CHECK_EQ(active_isa(), Isa::scalar);
    for (int i = 0; i < kIsaCount; ++i) {
      const auto isa = static_cast<Isa>(i);
      CHECK_EQ(set_active_isa(isa), clamp_isa(isa));
      CHECK(isa_supported(active_isa()));
    }
  }
  CHECK_EQ(active_isa(), before);
}

namespace {

struct FakeTable {
  int a = 0, b = 0, c = 0;
};
void fake_scalar(FakeTable& t) { t = {1, 1, 1}; }
void fake_neon(FakeTable& t) { t.b = 2; }
void fake_dotprod(FakeTable& t) { t.c = 3; }
void fake_sse41(FakeTable& t) { t.b = 4; }
void fake_avx2(FakeTable& t) { t.c = 5; }

}  // namespace

TEST(dispatch_layers_tiers_and_falls_back) {
  const detail::Dispatcher<FakeTable> d({
      .scalar = &fake_scalar,
      .neon = &fake_neon,
      .neon_dotprod = &fake_dotprod,
      .sse41 = &fake_sse41,
      .avx2 = &fake_avx2,
  });
  const auto expect = [](Isa isa) -> FakeTable {
    switch (isa) {
      case Isa::scalar: return {1, 1, 1};
      case Isa::neon: return {1, 2, 1};
      case Isa::neon_dotprod:
      case Isa::neon_i8mm: return {1, 2, 3};  // i8mm has no fill: inherits dotprod
      case Isa::sse41: return {1, 4, 1};
      case Isa::avx2: return {1, 4, 5};
    }
    return {};
  };
  for (int i = 0; i < kIsaCount; ++i) {
    const auto isa = static_cast<Isa>(i);
    const FakeTable& t = d.get(isa);  // unsupported tiers clamp
    const FakeTable want = expect(clamp_isa(isa));
    CHECK_EQ(t.a, want.a);
    CHECK_EQ(t.b, want.b);
    CHECK_EQ(t.c, want.c);
  }
  CHECK_EQ(&d.get(Isa::scalar), &d.get(Isa::scalar));  // built once
}

TEST(dispatch_is_thread_safe_on_first_use) {
  const detail::Dispatcher<FakeTable> d({.scalar = &fake_scalar, .neon = &fake_neon});
  const FakeTable* seen[4] = {};
  std::thread threads[4];
  for (int i = 0; i < 4; ++i) threads[i] = std::thread([&, i] { seen[i] = &d.get(best_isa()); });
  for (auto& t : threads) t.join();
  for (const FakeTable* p : seen) CHECK_EQ(p, seen[0]);
}

TEST(frame_layout_guarantee) {
  for (int fi = 0; fi < kPixelFormatCount; ++fi) {
    const auto fmt = static_cast<PixelFormat>(fi);
    for (const auto& [w, h] : test::test_sizes()) {
      Frame f;
      REQUIRE_OK(Frame::allocate(fmt, w, h, f));
      CHECK_EQ(f.plane_count(), pixfmt_info(fmt).plane_count);
      const std::uint8_t* prev_end = nullptr;
      for (int p = 0; p < f.plane_count(); ++p) {
        const Plane pl = f.plane(p);
        CHECK_EQ(reinterpret_cast<std::uintptr_t>(pl.data) % kFrameAlignment, 0u);
        CHECK_EQ(static_cast<std::size_t>(pl.stride) % kFrameAlignment, 0u);
        CHECK(static_cast<std::size_t>(pl.stride) >=
              static_cast<std::size_t>(pl.row_bytes()) + kRowPadding);
        CHECK_EQ(pl.width, plane_width(fmt, p, w));
        CHECK_EQ(pl.height, plane_height(fmt, p, h));
        if (prev_end != nullptr) CHECK(pl.data >= prev_end);
        prev_end = pl.data + pl.stride * pl.height;
      }
    }
  }
}

TEST(plane_sizes_round_up) {
  CHECK_EQ(plane_width(PixelFormat::yuv420p, 1, 3), 2);
  CHECK_EQ(plane_height(PixelFormat::yuv420p, 2, 5), 3);
  CHECK_EQ(plane_row_bytes(PixelFormat::nv12, 1, 5), 6);
  CHECK_EQ(plane_row_bytes(PixelFormat::rgb24, 0, 7), 21);
  CHECK_EQ(pixfmt_row_alignment(PixelFormat::yuv420p), 2);
  CHECK_EQ(pixfmt_row_alignment(PixelFormat::nv12), 2);
  CHECK_EQ(pixfmt_row_alignment(PixelFormat::rgba), 1);
  CHECK_EQ(pixfmt_row_alignment(PixelFormat::gray8), 1);
}

TEST(pixfmt_names_round_trip) {
  for (int i = 0; i < kPixelFormatCount; ++i) {
    const auto fmt = static_cast<PixelFormat>(i);
    CHECK(parse_pixfmt(pixfmt_name(fmt)) == fmt);
  }
  CHECK(parse_pixfmt("gray") == PixelFormat::gray8);
  CHECK(!parse_pixfmt("yuv444p").has_value());
  CHECK(parse_color_range("tv") == ColorRange::limited);
  CHECK(parse_color_matrix("bt709") == ColorMatrix::bt709);
}

TEST(frame_allocate_rejects_bad_dims) {
  Frame f;
  CHECK_EQ(Frame::allocate(PixelFormat::rgb24, 0, 4, f), Status::invalid_argument);
  CHECK_EQ(Frame::allocate(PixelFormat::rgb24, 4, -1, f), Status::invalid_argument);
  CHECK_EQ(Frame::allocate(PixelFormat::rgb24, kMaxDimension + 1, 4, f), Status::invalid_argument);
  CHECK(f.empty());
}

TEST(frame_ensure_reuses_and_clone_copies) {
  Frame f = test::random_frame(PixelFormat::yuv420p, 33, 17, 7);
  const std::uint8_t* data = f.plane(0).data;
  REQUIRE_OK(f.ensure(PixelFormat::yuv420p, 33, 17));
  CHECK_EQ(f.plane(0).data, data);
  Frame c;
  REQUIRE_OK(f.clone(c));
  CHECK(test::first_difference(f, c).empty());
  REQUIRE_OK(f.ensure(PixelFormat::rgba, 8, 8));
  CHECK_EQ(f.format(), PixelFormat::rgba);
}

TEST(resolve_rows_validates_alignment) {
  Frame f;
  REQUIRE_OK(Frame::allocate(PixelFormat::yuv420p, 8, 7, f));
  RowRange r;
  REQUIRE_OK(resolve_rows(f, {}, r));
  CHECK_EQ(r.begin, 0);
  CHECK_EQ(r.end, 7);
  CHECK_EQ(resolve_rows(f, {1, 4}, r), Status::invalid_argument);
  CHECK_EQ(resolve_rows(f, {2, 5}, r), Status::invalid_argument);
  CHECK_EQ(resolve_rows(f, {4, 7}, r), Status::ok);
  CHECK_EQ(resolve_rows(f, {0, 8}, r), Status::invalid_argument);
}

TEST(scratch_is_aligned_and_grow_only) {
  std::uint8_t* a = detail::thread_scratch(0, 100);
  REQUIRE(a != nullptr);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(a) % kFrameAlignment, 0u);
  CHECK_EQ(detail::thread_scratch(0, 50), a);
  CHECK(detail::thread_scratch(1, 10) != a);
  CHECK(detail::thread_scratch(detail::kScratchSlots, 10) == nullptr);
}

TEST(isa_helper_detects_mismatch_and_ignores_padding) {
  const auto make_out = [] { return test::random_frame(PixelFormat::rgb24, 5, 3, 1); };
  const auto same = [](Frame& out) {
    for (int y = 0; y < out.height(); ++y) out.plane(0).row(y)[0] = 42;
    return Status::ok;
  };
  CHECK(test::isa_mismatch(make_out, same).empty());
  const auto differs = [](Frame& out) {
    out.plane(0).row(0)[0] = static_cast<std::uint8_t>(active_isa());
    return Status::ok;
  };
  const bool many = supported_isas().size() > 1;
  CHECK_EQ(test::isa_mismatch(make_out, differs).empty(), !many);
}

TEST(status_names) {
  CHECK_EQ(status_name(Status::ok), std::string_view("ok"));
  CHECK_EQ(status_name(Status::end_of_stream), std::string_view("end_of_stream"));
}
