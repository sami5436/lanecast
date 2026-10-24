#include "lanecast/cpu.hpp"

#include <atomic>
#include <cstdlib>

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#elif defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

namespace lanecast {
namespace {

#if defined(__APPLE__) && defined(__aarch64__)
bool sysctl_flag(const char* name) {
  int value = 0;
  std::size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}
#endif

CpuFeatures detect() noexcept {
  CpuFeatures f;
#if defined(__aarch64__)
  f.neon = true;
#if defined(__APPLE__)
  f.neon_dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
  f.neon_i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__)
  constexpr unsigned long kHwcapAsimddp = 1UL << 20;  // HWCAP_ASIMDDP
  constexpr unsigned long kHwcap2I8mm = 1UL << 13;    // HWCAP2_I8MM
  f.neon_dotprod = (getauxval(AT_HWCAP) & kHwcapAsimddp) != 0;
  f.neon_i8mm = (getauxval(AT_HWCAP2) & kHwcap2I8mm) != 0;
#endif
#elif defined(__x86_64__)
  __builtin_cpu_init();
  f.sse41 = __builtin_cpu_supports("sse4.1");
  f.avx2 = __builtin_cpu_supports("avx2");
#endif
  return f;
}

constexpr int kUnresolved = -1;
std::atomic<int> g_active{kUnresolved};

Isa resolve_initial() noexcept {
  if (const char* env = std::getenv("LANECAST_ISA"); env != nullptr) {
    if (auto requested = parse_isa(env)) return clamp_isa(*requested);
  }
  return best_isa();
}

}  // namespace

std::string_view isa_name(Isa isa) noexcept {
  switch (isa) {
    case Isa::scalar: return "scalar";
    case Isa::neon: return "neon";
    case Isa::neon_dotprod: return "neon_dotprod";
    case Isa::neon_i8mm: return "neon_i8mm";
    case Isa::sse41: return "sse41";
    case Isa::avx2: return "avx2";
  }
  return "unknown";
}

std::optional<Isa> parse_isa(std::string_view name) noexcept {
  for (int i = 0; i < kIsaCount; ++i) {
    const auto isa = static_cast<Isa>(i);
    if (isa_name(isa) == name) return isa;
  }
  return std::nullopt;
}

std::optional<Isa> isa_parent(Isa isa) noexcept {
  switch (isa) {
    case Isa::scalar: return std::nullopt;
    case Isa::neon: return Isa::scalar;
    case Isa::neon_dotprod: return Isa::neon;
    case Isa::neon_i8mm: return Isa::neon_dotprod;
    case Isa::sse41: return Isa::scalar;
    case Isa::avx2: return Isa::sse41;
  }
  return std::nullopt;
}

const CpuFeatures& cpu_features() noexcept {
  static const CpuFeatures features = detect();
  return features;
}

bool isa_compiled(Isa isa) noexcept {
  switch (isa) {
    case Isa::scalar: return true;
    case Isa::neon: return LANECAST_HAVE_NEON != 0;
    case Isa::neon_dotprod: return LANECAST_HAVE_NEON_DOTPROD != 0;
    case Isa::neon_i8mm: return LANECAST_HAVE_NEON_I8MM != 0;
    case Isa::sse41: return LANECAST_HAVE_SSE41 != 0;
    case Isa::avx2: return LANECAST_HAVE_AVX2 != 0;
  }
  return false;
}

bool isa_supported(Isa isa) noexcept {
  if (!isa_compiled(isa)) return false;
  const CpuFeatures& f = cpu_features();
  switch (isa) {
    case Isa::scalar: return true;
    case Isa::neon: return f.neon;
    case Isa::neon_dotprod: return f.neon && f.neon_dotprod;
    case Isa::neon_i8mm: return f.neon && f.neon_dotprod && f.neon_i8mm;
    case Isa::sse41: return f.sse41;
    case Isa::avx2: return f.sse41 && f.avx2;
  }
  return false;
}

Isa clamp_isa(Isa requested) noexcept {
  for (std::optional<Isa> t = requested; t.has_value(); t = isa_parent(*t)) {
    if (isa_supported(*t)) return *t;
  }
  return Isa::scalar;
}

Isa best_isa() noexcept {
  Isa best = Isa::scalar;
  int best_depth = 0;
  for (int i = 0; i < kIsaCount; ++i) {
    const auto isa = static_cast<Isa>(i);
    if (!isa_supported(isa)) continue;
    int depth = 0;
    for (auto t = isa_parent(isa); t.has_value(); t = isa_parent(*t)) ++depth;
    if (depth > best_depth) {
      best = isa;
      best_depth = depth;
    }
  }
  return best;
}

std::vector<Isa> supported_isas() {
  std::vector<Isa> out;
  for (int i = 0; i < kIsaCount; ++i) {
    if (isa_supported(static_cast<Isa>(i))) out.push_back(static_cast<Isa>(i));
  }
  return out;
}

Isa active_isa() noexcept {
  int v = g_active.load(std::memory_order_relaxed);
  if (v == kUnresolved) {
    int expected = kUnresolved;
    g_active.compare_exchange_strong(expected, static_cast<int>(resolve_initial()),
                                     std::memory_order_relaxed);
    v = g_active.load(std::memory_order_relaxed);
  }
  return static_cast<Isa>(v);
}

Isa set_active_isa(Isa requested) noexcept {
  const Isa applied = clamp_isa(requested);
  g_active.store(static_cast<int>(applied), std::memory_order_relaxed);
  return applied;
}

}  // namespace lanecast
