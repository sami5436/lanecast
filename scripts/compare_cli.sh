#!/usr/bin/env bash
# End-to-end comparison: lanecast CLI vs ffmpeg CLI on a generated clip with
# the same filter chain. Prints median-of-5 frames per second for each and the
# ratio. Prints no machine information.
#
#   scripts/compare_cli.sh [-b LANECAST_BIN] [-c CHAIN] [-s WxH] [-n FRAMES] [-t THREADS]
#
# Defaults: build/tools/lanecast, "scale=1280:720,format=rgb24", 1920x1080,
# 120 frames, threads 0 (all). ffmpeg gets "-sws_flags bilinear" to match
# lanecast's default scale filter, and the same thread count.
set -euo pipefail

bin=build/tools/lanecast
chain="scale=1280:720,format=rgb24"
size=1920x1080
frames=120
threads=0
runs=5

while getopts "b:c:s:n:t:h" opt; do
  case "$opt" in
    b) bin=$OPTARG ;;
    c) chain=$OPTARG ;;
    s) size=$OPTARG ;;
    n) frames=$OPTARG ;;
    t) threads=$OPTARG ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
  esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
command -v perl >/dev/null || { echo "perl not found" >&2; exit 1; }
[ -x "$bin" ] || { echo "lanecast binary not found: $bin" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
clip="$work/clip.y4m"

ffmpeg -v error -f lavfi -i "testsrc2=size=$size:rate=30" -frames:v "$frames" \
  -pix_fmt yuv420p "$clip"

# Wall seconds of one command.
seconds() {
  perl -MTime::HiRes=time -e '
    my $t = time; system(@ARGV) == 0 or exit 1; printf "%.6f\n", time - $t' -- "$@"
}

median() { sort -n | awk '{ a[NR] = $1 } END { print a[int((NR + 1) / 2)] }'; }

# Output goes to /dev/null as raw frames, so neither side pays for a muxer.
lc_cmd=("$bin" -i "$clip" -vf "$chain" -f rawvideo -y -o /dev/null)
ff_cmd=(ffmpeg -v error -i "$clip" -sws_flags bilinear -vf "$chain" -f rawvideo -y /dev/null)
if [ "$threads" -gt 0 ]; then
  lc_cmd+=(-threads "$threads")
  ff_cmd=(ffmpeg -v error -threads "$threads" -i "$clip" -sws_flags bilinear -vf "$chain" \
    -filter_threads "$threads" -f rawvideo -y /dev/null)
fi

"${lc_cmd[@]}"  # warms the page cache and fails early on a bad chain
"${ff_cmd[@]}"

lc_t=$(for _ in $(seq "$runs"); do seconds "${lc_cmd[@]}"; done | median)
ff_t=$(for _ in $(seq "$runs"); do seconds "${ff_cmd[@]}"; done | median)

awk -v n="$frames" -v lc="$lc_t" -v ff="$ff_t" -v chain="$chain" -v size="$size" 'BEGIN {
  printf "chain: %s  input: %s yuv420p, %d frames, median of 5\n", chain, size, n
  printf "lanecast: %8.1f fps\nffmpeg:   %8.1f fps\nspeedup:  %8.2fx\n", n / lc, n / ff, ff / lc
}'
