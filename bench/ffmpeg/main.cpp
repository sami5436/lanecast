// bench_vs_ffmpeg: lanecast vs libswscale / libavfilter on identical input,
// output format and thread count. Reports ns/frame, speedup and PSNR of each
// output against the reference (see ARCHITECTURE.md "Beating ffmpeg").
#include <cstdio>

extern "C" {
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

int main() {
  const unsigned v = swscale_version();
  std::printf("bench_vs_ffmpeg: libswscale %u.%u.%u linked; no comparisons registered yet\n",
              v >> 16, (v >> 8) & 0xFF, v & 0xFF);
  return 0;
}
