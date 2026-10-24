// Helpers for kernel tests: deterministic random frames, padding poisoning,
// frame diffs, and running an op under every supported ISA tier.
//
// The core rule: every kernel test compares each tier to scalar with
// CHECK_ALL_ISAS_MATCH, over test_sizes() (odd widths, tails, tiny frames).
// SIMD units then need no new tests: adding a tier is covered automatically.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lanecast/cpu.hpp"
#include "lanecast/frame.hpp"
#include "lanecast_test.hpp"

namespace lanecast::test {

// splitmix64: tiny, deterministic, identical on every platform.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  std::uint8_t byte() { return static_cast<std::uint8_t>(next() >> 56); }
  int range(int lo, int hi) {  // inclusive
    return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1));
  }

 private:
  std::uint64_t state_;
};

// Width x height pairs covering 1-pixel frames, odd sizes, every vector
// tail length for 16, 32 and 64 byte vectors, and one wide row.
inline const std::vector<std::pair<int, int>>& test_sizes() {
  static const std::vector<std::pair<int, int>> sizes = {
      {1, 1},   {2, 2},  {3, 5},   {7, 3},   {15, 7},  {16, 16}, {17, 9},
      {31, 3},  {32, 2}, {33, 33}, {47, 4},  {63, 2},  {64, 64}, {65, 17},
      {127, 5}, {129, 3}, {255, 3}, {1920, 4},
  };
  return sizes;
}

inline void fill_random(Frame& f, std::uint64_t seed) {
  Rng rng(seed);
  for (int p = 0; p < f.plane_count(); ++p) {
    const Plane pl = f.plane(p);
    for (int y = 0; y < pl.height; ++y) {
      std::uint8_t* row = pl.row(y);
      for (int i = 0; i < pl.row_bytes(); ++i) row[i] = rng.byte();
    }
  }
}

// Overwrites every padding byte (row slack) with random data.
inline void poison_padding(Frame& f, std::uint64_t seed) {
  Rng rng(seed ^ 0xA5A5A5A5A5A5A5A5ULL);
  for (int p = 0; p < f.plane_count(); ++p) {
    const Plane pl = f.plane(p);
    for (int y = 0; y < pl.height; ++y) {
      std::uint8_t* row = pl.row(y);
      for (std::ptrdiff_t i = pl.row_bytes(); i < pl.stride; ++i) row[i] = rng.byte();
    }
  }
}

// Allocated frame with random visible bytes and poisoned padding.
inline Frame random_frame(PixelFormat fmt, int width, int height, std::uint64_t seed) {
  Frame f;
  REQUIRE_OK(Frame::allocate(fmt, width, height, f));
  fill_random(f, seed);
  poison_padding(f, seed + 1);
  return f;
}

// Returns "" when the visible bytes match, else where the first mismatch is.
inline std::string first_difference(const Frame& expected, const Frame& actual) {
  if (!expected.same_shape(actual)) return "frame shapes differ";
  for (int p = 0; p < expected.plane_count(); ++p) {
    const Plane e = expected.plane(p);
    const Plane a = actual.plane(p);
    for (int y = 0; y < e.height; ++y) {
      const std::uint8_t* er = e.row(y);
      const std::uint8_t* ar = a.row(y);
      if (std::memcmp(er, ar, static_cast<std::size_t>(e.row_bytes())) == 0) continue;
      for (int i = 0; i < e.row_bytes(); ++i) {
        if (er[i] != ar[i]) {
          return "plane " + std::to_string(p) + " row " + std::to_string(y) + " byte " +
                 std::to_string(i) + ": expected " + std::to_string(er[i]) + " got " +
                 std::to_string(ar[i]);
        }
      }
    }
  }
  return {};
}

// Sets the active tier for a scope, restoring the previous one after.
class ScopedIsa {
 public:
  explicit ScopedIsa(Isa isa) : previous_(active_isa()), applied_(set_active_isa(isa)) {}
  ~ScopedIsa() { set_active_isa(previous_); }
  ScopedIsa(const ScopedIsa&) = delete;
  ScopedIsa& operator=(const ScopedIsa&) = delete;
  Isa applied() const { return applied_; }

 private:
  Isa previous_;
  Isa applied_;
};

// Runs produce(out) with scalar active, then with each other supported tier
// active, on fresh outputs from make_out() whose padding is re-poisoned with
// a different seed per tier. Returns "" when every tier matches scalar.
//   make_out: () -> Frame        produce: (Frame&) -> Status
template <class MakeOut, class Produce>
std::string isa_mismatch(MakeOut&& make_out, Produce&& produce) {
  Frame reference = make_out();
  {
    ScopedIsa scope(Isa::scalar);
    if (const Status s = produce(reference); s != Status::ok) {
      return "scalar returned " + show(s);
    }
  }
  for (const Isa isa : supported_isas()) {
    if (isa == Isa::scalar) continue;
    Frame out = make_out();
    poison_padding(out, 0x5EED0000ULL + static_cast<std::uint64_t>(isa));
    ScopedIsa scope(isa);
    if (const Status s = produce(out); s != Status::ok) {
      return show(isa) + " returned " + show(s);
    }
    if (std::string d = first_difference(reference, out); !d.empty()) {
      return show(isa) + " differs from scalar: " + d;
    }
  }
  return {};
}

// Same for ops that return a value (metrics): compute() -> T, T has ==.
template <class Compute>
std::string isa_value_mismatch(Compute&& compute) {
  std::string out;
  const auto reference = [&] {
    ScopedIsa scope(Isa::scalar);
    return compute();
  }();
  for (const Isa isa : supported_isas()) {
    if (isa == Isa::scalar) continue;
    ScopedIsa scope(isa);
    const auto value = compute();
    if (!(value == reference)) return show(isa) + " differs from scalar";
  }
  return out;
}

}  // namespace lanecast::test

#define CHECK_ALL_ISAS_MATCH(make_out, produce)                                       \
  do {                                                                                \
    const std::string lc_m = ::lanecast::test::isa_mismatch((make_out), (produce));   \
    CHECK_MSG(lc_m.empty(), lc_m);                                                    \
  } while (0)
