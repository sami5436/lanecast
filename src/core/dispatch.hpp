// Per-kernel-family dispatch tables.
//
// Each kernel directory (src/kernels/<op>/) defines:
//   - kernels.hpp: a POD `Table` of row-kernel function pointers plus one
//     `void fill_<isa>(Table&)` declaration per ISA file in the directory.
//   - scalar.cpp, neon.cpp, sse41.cpp, avx2.cpp (and optionally
//     neon_dotprod.cpp, neon_i8mm.cpp): each fill function stores the
//     kernels that tier specializes and leaves the rest untouched.
//   - dispatch.cpp: one Dispatcher built from those fill functions.
//
// Building the table for a tier runs the fills from scalar up its chain
// (see cpu.hpp), so a tier inherits every kernel it does not override.
// Each tier's table is built once, on first use, under std::call_once.
// After that a lookup is an atomic load (active_isa) plus an array index.
#pragma once

#include <cstddef>
#include <mutex>
#include <optional>

#include "lanecast/cpu.hpp"

// Expands to the fill function when that tier is compiled in, else nullptr.
// The identifier is not referenced when the tier is off, so the symbol need
// not exist (its source file is not compiled on that architecture).
#if LANECAST_HAVE_NEON
#define LANECAST_IF_NEON(fn) (&fn)
#else
#define LANECAST_IF_NEON(fn) nullptr
#endif
#if LANECAST_HAVE_NEON_DOTPROD
#define LANECAST_IF_NEON_DOTPROD(fn) (&fn)
#else
#define LANECAST_IF_NEON_DOTPROD(fn) nullptr
#endif
#if LANECAST_HAVE_NEON_I8MM
#define LANECAST_IF_NEON_I8MM(fn) (&fn)
#else
#define LANECAST_IF_NEON_I8MM(fn) nullptr
#endif
#if LANECAST_HAVE_SSE41
#define LANECAST_IF_SSE41(fn) (&fn)
#else
#define LANECAST_IF_SSE41(fn) nullptr
#endif
#if LANECAST_HAVE_AVX2
#define LANECAST_IF_AVX2(fn) (&fn)
#else
#define LANECAST_IF_AVX2(fn) nullptr
#endif

namespace lanecast::detail {

template <class Table>
struct KernelFills {
  using Fill = void (*)(Table&);
  Fill scalar = nullptr;
  Fill neon = nullptr;
  Fill neon_dotprod = nullptr;
  Fill neon_i8mm = nullptr;
  Fill sse41 = nullptr;
  Fill avx2 = nullptr;

  Fill for_isa(Isa isa) const noexcept {
    switch (isa) {
      case Isa::scalar: return scalar;
      case Isa::neon: return neon;
      case Isa::neon_dotprod: return neon_dotprod;
      case Isa::neon_i8mm: return neon_i8mm;
      case Isa::sse41: return sse41;
      case Isa::avx2: return avx2;
    }
    return nullptr;
  }
};

template <class Table>
class Dispatcher {
 public:
  explicit Dispatcher(const KernelFills<Table>& fills) : fills_(fills) {}
  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  // Table for `isa`, clamped to what this CPU and build support.
  const Table& get(Isa isa) const {
    const Isa resolved = clamp_isa(isa);
    const auto i = static_cast<std::size_t>(resolved);
    std::call_once(once_[i], [&] { tables_[i] = build(resolved); });
    return tables_[i];
  }

  const Table& active() const { return get(active_isa()); }

 private:
  Table build(Isa isa) const {
    Isa chain[kIsaCount];
    int n = 0;
    for (std::optional<Isa> t = isa; t.has_value(); t = isa_parent(*t)) chain[n++] = *t;
    Table table{};
    for (int k = n - 1; k >= 0; --k) {
      if (auto fill = fills_.for_isa(chain[k])) fill(table);
    }
    return table;
  }

  KernelFills<Table> fills_;
  mutable std::once_flag once_[kIsaCount];
  mutable Table tables_[kIsaCount] = {};
};

}  // namespace lanecast::detail
