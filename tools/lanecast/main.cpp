// lanecast CLI. Skeleton implements `info` and `--version`; the CLI unit
// adds convert, scale, psnr, ssim and the -vf pipeline.
#include <cstdio>
#include <string_view>

#include "lanecast/cpu.hpp"
#include "lanecast/version.hpp"

namespace {

void print_usage() {
  std::puts(
      "usage: lanecast <command> [options]\n"
      "  info        show compiled, supported and active ISA tiers\n"
      "  --version   print version\n"
      "\n"
      "planned: convert -i IN -o OUT [-vf CHAIN] [--threads N] [--isa ISA]\n"
      "         psnr A B, ssim A B");
}

int cmd_info() {
  using namespace lanecast;
  std::printf("lanecast %.*s\n", static_cast<int>(kVersionString.size()), kVersionString.data());
  for (int i = 0; i < kIsaCount; ++i) {
    const auto isa = static_cast<Isa>(i);
    const std::string_view name = isa_name(isa);
    std::printf("  %-13.*s compiled=%d supported=%d\n", static_cast<int>(name.size()),
                name.data(), isa_compiled(isa) ? 1 : 0, isa_supported(isa) ? 1 : 0);
  }
  const std::string_view active = isa_name(active_isa());
  std::printf("active: %.*s\n", static_cast<int>(active.size()), active.data());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }
  const std::string_view cmd = argv[1];
  if (cmd == "info") return cmd_info();
  if (cmd == "--version") {
    std::printf("%.*s\n", static_cast<int>(lanecast::kVersionString.size()),
                lanecast::kVersionString.data());
    return 0;
  }
  if (cmd == "help" || cmd == "--help" || cmd == "-h") {
    print_usage();
    return 0;
  }
  std::fprintf(stderr, "lanecast: '%s' is not implemented yet\n", argv[1]);
  return 2;
}
