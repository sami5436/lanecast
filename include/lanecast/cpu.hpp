// ISA tiers, runtime detection, and the active-ISA override.
//
// Tiers form two chains. A tier includes everything below it, and a kernel
// table for a tier is built as scalar, then each lower tier, then the tier
// itself, so a tier with no specialization for a kernel falls back to the
// next lower tier's version:
//   AArch64: scalar < neon < neon_dotprod < neon_i8mm
//   x86-64:  scalar < sse41 < avx2
//
// Override: the environment variable LANECAST_ISA (scalar, neon,
// neon_dotprod, neon_i8mm, sse41, avx2) is read once, on the first call to
// active_isa(). The request is clamped: if it is not supported, the highest
// supported tier below it in the same chain is used (a tier from the other
// architecture's chain clamps to scalar). Unknown values are ignored.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace lanecast {

enum class Isa : std::uint8_t {
  scalar = 0,
  neon = 1,
  neon_dotprod = 2,
  neon_i8mm = 3,  // slot reserved; no v1 kernels required
  sse41 = 4,
  avx2 = 5,
};
inline constexpr int kIsaCount = 6;

struct CpuFeatures {
  bool neon = false;
  bool neon_dotprod = false;  // FEAT_DotProd (macOS sysctl, Linux HWCAP_ASIMDDP)
  bool neon_i8mm = false;     // FEAT_I8MM (macOS sysctl, Linux HWCAP2_I8MM)
  bool sse41 = false;         // __builtin_cpu_supports("sse4.1")
  bool avx2 = false;          // __builtin_cpu_supports("avx2"), includes OS YMM support
};

std::string_view isa_name(Isa isa) noexcept;
std::optional<Isa> parse_isa(std::string_view name) noexcept;

// The tier directly below `isa` in its chain (scalar has none).
std::optional<Isa> isa_parent(Isa isa) noexcept;

// What the running CPU reports, independent of what was compiled.
const CpuFeatures& cpu_features() noexcept;

bool isa_compiled(Isa isa) noexcept;   // kernels for this tier are in the binary
bool isa_supported(Isa isa) noexcept;  // compiled and the CPU can run it
Isa best_isa() noexcept;               // highest supported tier
Isa clamp_isa(Isa requested) noexcept;

// Supported tiers in ascending order, scalar first.
std::vector<Isa> supported_isas();

// The tier every op dispatches to. Resolved once (env override, clamped),
// then read with one relaxed atomic load per op call. Thread-safe.
Isa active_isa() noexcept;

// Forces the active tier (clamped); returns the tier actually applied.
// Meant for tests and benchmarks. Because every tier is bit-exact with
// scalar, changing it while other threads run ops cannot change results.
Isa set_active_isa(Isa requested) noexcept;

}  // namespace lanecast
