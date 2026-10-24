# lanecast architecture

## Context

lanecast is a portable C++20 raw-video toolkit inspired by FFmpeg: pixel
format conversion, scaling, filters, quality metrics, Y4M/raw/PNM IO and a
slice-threaded CLI pipeline. No codecs. Every kernel has a scalar reference
and hand-written SIMD (NEON, NEON dotprod, SSE4.1, AVX2), selected at runtime.
The bar is to be faster than libswscale and libavfilter on every op lanecast
implements, at equal or better quality ("Beating ffmpeg" below).

Toolchain: CMake 3.20+, GCC or Clang, C++20, default generator (Unix
Makefiles works, Ninja optional). Targets: AArch64 (macOS, Linux) and x86-64.

## Layout

```
include/lanecast/     public API (one header per module, umbrella lanecast.hpp)
  status.hpp pixfmt.hpp frame.hpp cpu.hpp version.hpp
  convert.hpp scale.hpp filter.hpp metrics.hpp io.hpp pipeline.hpp
src/core/             frame alloc, cpu detection, dispatch.hpp, scratch arena
src/kernels/<op>/     convert, scale, filter, metrics
  <op>.cpp            front end: validation, routing, row loops
  kernels.hpp         POD Table of row-kernel function pointers (internal contract)
  dispatch.cpp        builds the per-ISA tables
  scalar.cpp          reference kernels (the spec in code)
  neon.cpp neon_dotprod.cpp sse41.cpp avx2.cpp   ISA kernels
src/io/               y4m, raw, pnm readers and writers
src/pipeline/         thread pool, filter chain, Pipeline, fused ScaleConvert
tools/lanecast/       CLI
bench/                lanecast_bench (self, per ISA); bench/ffmpeg/ bench_vs_ffmpeg
tests/                header-only runner, test_support.hpp, test_<suite>.cpp
cmake/                LanecastArch.cmake (arch, ISA flags), LanecastSources.cmake (globs)
```

All sources are globbed (`CONFIGURE_DEPENDS`), so units add files without
editing any CMakeLists.txt. File name suffix decides ISA handling:

| Stem suffix     | Compiled on | Per-file flags                     |
|-----------------|-------------|------------------------------------|
| `neon`          | AArch64     | none (baseline)                    |
| `neon_dotprod`  | AArch64     | `-march=armv8.2-a+dotprod`         |
| `neon_i8mm`     | AArch64     | `-march=armv8.2-a+dotprod+i8mm`    |
| `sse41`         | x86-64      | `-msse4.1`                         |
| `avx2`          | x86-64      | `-mavx2`                           |
| anything else   | everywhere  | none                               |

Flags never apply to a whole target, so one binary runs on any CPU of its
architecture. Architecture comes from `CMAKE_SYSTEM_PROCESSOR` (or a single
`CMAKE_OSX_ARCHITECTURES`). `-DLANECAST_ENABLE_SIMD=OFF` builds scalar only.

## Modules and dependency rules

```
tools, bench  ->  pipeline  ->  kernels/{convert,scale,filter,metrics}  ->  core
tools, bench  ->  io        ->  core
```

- `core` (frame, pixfmt, cpu, dispatch, scratch) depends on nothing else.
- Each kernel family depends only on core. Kernel families never include
  each other's headers.
- `io` depends only on core (frame). It never calls kernels.
- `pipeline` may use public ops and, for fused passes only, the internal
  `kernels/scale/kernels.hpp` and `kernels/convert/kernels.hpp` row kernels.
- `tools` and `bench` use public headers; `bench` may use internal kernel
  tables to time single kernels per ISA.
- Tests may include anything (`src/` is on their include path).

## Data flow

```
FrameReader (y4m/raw/pnm) -> Frame (64-byte aligned, padded)
  -> Pipeline::process: stages, each split into row slices on ThreadPool
       stage = convert | Scaler | blur | brightness_contrast | ScaleConvert (fused)
       each slice -> op(src, dst, RowRange) -> active_table() row kernels
  -> Frame -> FrameWriter
metrics: compute_psnr / compute_ssim(a, b) -> integer sums per row -> double math
```

## Core contracts

### Frame memory (include/lanecast/frame.hpp)
- Plane pointers are 64-byte aligned; strides are multiples of 64 and at
  least `row_bytes + 64`.
- A kernel may read and write up to 64 bytes past the last visible byte of
  any row it owns. Finish tails with one full vector, not a scalar loop.
- Padding is unspecified; results must not depend on it. Tests poison it.
- No vertical overread: kernels touch only the rows they are given, so
  slices on different threads never share bytes.

### ISA tiers and dispatch (include/lanecast/cpu.hpp, src/core/dispatch.hpp)
- Chains: AArch64 `scalar < neon < neon_dotprod < neon_i8mm`; x86-64
  `scalar < sse41 < avx2`. Detection: NEON always on AArch64; dotprod and
  i8mm via `sysctlbyname("hw.optional.arm.FEAT_DotProd" / "FEAT_I8MM")` on
  macOS and `getauxval` `HWCAP_ASIMDDP` / `HWCAP2_I8MM` on Linux; x86 via
  `__builtin_cpu_supports("sse4.1" / "avx2")`.
- `LANECAST_ISA` is read once and clamped down its chain to a supported
  tier (other-architecture tiers clamp to scalar, unknown values ignored).
- Each kernel family has a POD `Table`. A tier's table is built by running
  the fill functions from scalar up the chain, so a tier only stores what it
  specializes and inherits the rest (dotprod falls back to neon, avx2 to
  sse41). Built once per tier under `std::call_once`; lookups are an atomic
  load plus an index. A null entry means not implemented.

### Adding a kernel (the whole pattern)
1. Add the function pointer type and `Table` field in `src/kernels/<op>/kernels.hpp`.
2. Write the reference in `scalar.cpp`, store it in `fill_scalar`.
3. Call it from the front end through `active_table()` (null: `not_implemented`).
4. Add tests: known values, then `CHECK_ALL_ISAS_MATCH` over `test::test_sizes()`.
5. Add the SIMD version in `neon.cpp` / `sse41.cpp` / `avx2.cpp` and store it in
   that tier's fill. No new test needed: the ISA comparison covers it.
6. A new tier file in a directory (for example `convert/neon_dotprod.cpp`)
   also adds its fill to that directory's `dispatch.cpp`.

### SIMD translation unit rules
- Include only `<cstddef>`, `<cstdint>`, intrinsic headers and the
  directory's `kernels.hpp`. Kernel signatures take raw pointers and PODs.
- Do not call inline functions or templates from other headers inside ISA
  files: COMDAT folding can make every caller use the copy compiled with
  `-mavx2` (or `+dotprod`) and crash older CPUs. Helpers go in an anonymous
  namespace in the ISA file.
- No FMA flags, and `-ffp-contract=off` globally: integer kernels are exact
  and float math runs only in shared scalar code.

### Bit-exact contracts (full text in the public headers)
- **YUV to RGB** (convert.hpp): Q13 coefficients, int32,
  `R = clamp8((cy*(Y-yoff) + crv*(V-128) + 4096) >> 13)`, G and B alike,
  nearest chroma. Coefficient tables for BT.601/BT.709 x limited/full are
  constexpr in the header and fit int16.
- **RGB to YUV** (convert.hpp): Q15, `Y = clamp8((yr*R + yg*G + yb*B +
  (yoff << 15) + 16384) >> 15)`; chroma from 2x2 sums with edge replication,
  `>> 17` with `(128 << 17) + 65536`. Sums adjusted so gray maps to 128 exactly.
- **Scale** (scale.hpp): integer-built filter tables (bilinear center
  aligned, area box coverage), Q14 weights summing to 16384; horizontal
  first: `mid = (sum + 64) >> 7` into int16, then vertical
  `clamp8((sum + (1 << 20)) >> 21)`.
- **Filter** (filter.hpp): `div255` blend, binomial blur with exact integer
  sums and one final rounding, LUT for brightness and contrast.
- **Metrics** (metrics.hpp): kernels produce exact integer sums (SSE, SAD,
  SSIM 4x4 block sums); PSNR and SSIM final math in shared double code.
- **Fused** (pipeline.hpp): `ScaleConvert` equals `scale` then `convert`.

## Performance design

- **Fused passes.** `ScaleConvert` runs yuv420p/nv12 -> scale -> rgb per row
  strip. A strip scales only the Y and chroma rows it needs into
  thread-local strip buffers (L1/L2 sized), then converts straight into the
  destination. No full-frame intermediate. `Pipeline` fuses adjacent
  `scale` + `format` (yuv to rgb) stages automatically.
- **Slice threading by default.** `PipelineOptions::threads = 0` means all
  hardware threads. `ThreadPool` keeps persistent workers; `run_sliced` makes
  aligned, disjoint row ranges. Every op takes a `RowRange`.
- **No per-frame allocation.** `Scaler` and `ScaleConvert` build filter
  tables once in `create()`. Scratch comes from `core/scratch.hpp`
  (thread-local, grow-only). Readers and pipelines reuse frames via
  `Frame::ensure`.
- **Tight fixed point.** int16 lanes wherever the contract allows (madd,
  vmull, sdot); one rounding per pass; padding removes scalar tails;
  specialized kernels per channel count and tap count (2-tap bilinear,
  exact 2x area).
- **Extra tiers.** `neon_dotprod` (UDOT) for metric sums and other u8 dot
  products; `neon_i8mm` slot (USDOT) reserved for signed-coefficient
  convert paths.

## Beating ffmpeg

`bench_vs_ffmpeg` (optional, `-DLANECAST_BENCH_FFMPEG=AUTO|ON|OFF`, AUTO
when pkg-config finds libswscale and libavutil; libavfilter enables filter
and metric ops) links the libraries directly, so there is no process or IO
overhead. Never required for the core build or CI.

Method, for each op: identical input frames; both sides prebuild their
contexts and output buffers outside the timed region; 3 warmup runs, then
the median of at least 20 runs, reported as ns/frame and speedup. Sizes
1920x1080 and 3840x2160. Thread counts 1 and N (same N for both: lanecast
`threads = N`; swscale `threads` option via `sws_alloc_context` +
`sws_init_context` + `sws_scale_frame`; libavfilter graph `nb_threads = N`).
Each output is also compared with PSNR against a double-precision reference
of the op (no intermediate rounding) and against the other library's output.

| Op | lanecast | ffmpeg side and flags |
|----|----------|-----------------------|
| yuv420p, nv12 -> rgb24, rgba, bgra (BT.601, BT.709, limited) | `convert` | swscale, `SWS_BILINEAR` (default unscaled fast path) and an accurate run with `SWS_ACCURATE_RND \| SWS_FULL_CHR_H_INT \| SWS_FULL_CHR_H_INP`; `sws_setColorspaceDetails` with `sws_getCoefficients(SWS_CS_ITU601 / SWS_CS_ITU709)`, src range 0, dst range 1 |
| rgb24, rgba, bgra -> yuv420p, nv12 | `convert` | swscale `SWS_BILINEAR`, plus `SWS_ACCURATE_RND` run |
| yuv420p <-> nv12, rgba <-> bgra, rgb24 -> rgba | `convert` | swscale unscaled converters, `SWS_BILINEAR` |
| scale yuv420p, rgba: 1080p -> 720p, 4K -> 1080p, 720p -> 1080p | `Scaler` bilinear | `SWS_BILINEAR` |
| downscale 1080p -> 720p, 4K -> 1080p | `Scaler` area | `SWS_AREA` |
| fused yuv420p 1080p -> rgb24 720p (and 4K -> 1080p) | `ScaleConvert` | one swscale context doing both, `SWS_BILINEAR` |
| blur 3x3, 5x5 binomial | `blur` | libavfilter `convolution` with the same matrix and rdiv |
| brightness and contrast | `brightness_contrast` | libavfilter `lutyuv` / `lutrgb` with the same expression |
| alpha blend rgba | `alpha_blend` | libavfilter `overlay` (rgba) |
| PSNR, SSIM | `compute_psnr`, `compute_ssim` | libavfilter `psnr`, `ssim` (results must also match to 1e-6) |

Acceptance bar, for every row, both sizes, threads 1 and N: lanecast median
ns/frame is lower than ffmpeg's, and lanecast PSNR against the reference is
equal or better than ffmpeg's. Bench output never contains host names,
user names, paths or CPU model strings; results committed to the repo only
as ratios.

## Testing

- `tests/lanecast_test.hpp`: header-only runner, `TEST`, `CHECK`,
  `CHECK_EQ`, `CHECK_MSG`, `REQUIRE`, `REQUIRE_OK`. Suite = file name
  (`tests/test_scale.cpp` is suite `scale`). One ctest entry per file.
- `tests/test_support.hpp`: `Rng`, `random_frame` (random pixels, poisoned
  padding), `first_difference`, `ScopedIsa`, `isa_mismatch` /
  `CHECK_ALL_ISAS_MATCH` (every supported tier vs scalar, padding re-poisoned
  per tier), `isa_value_mismatch` for metrics, `test_sizes()` (1x1, odd
  widths, every vector tail, a 1920 wide row).
- Every kernel test: known values on scalar, then all-ISA equality. RowRange
  slices must equal the whole-frame result.
- x86 tiers cannot run on an arm64 Mac: they are compile-checked locally
  (below) and run in CI (ubuntu: best tier, `LANECAST_ISA=scalar`, `sse41`).

## Presubmit

```sh
cmake -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
LANECAST_ISA=scalar ctest --test-dir build --output-on-failure
# arm64 Mac only: compile-check x86 tiers
cmake -B build-x86 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DLANECAST_BENCH_FFMPEG=OFF && cmake --build build-x86 -j
```

CI (`.github/workflows/ci.yml`) adds `-DLANECAST_WERROR=ON`; build clean
with it locally before opening a PR.

## Work units

Ownership is exclusive across workstreams. Units inside one workstream run
strictly in sequence and may touch the same files. Everything not listed is
owned by the skeleton (lead approval to change): root and per-directory
CMake, `cmake/`, `include/lanecast/{status,pixfmt,frame,cpu,version,lanecast}.hpp`,
`src/core/*`, `tests/{lanecast_test.hpp,test_support.hpp,main.cpp,test_core.cpp}`,
CI, LICENSE. Every unit: one PR, about 300 changed lines (hard ceiling 500),
branch `feat/<unit>`, and must pass presubmit with `-DLANECAST_WERROR=ON`.

Common acceptance prefix `B` = `cmake -B build -DLANECAST_WERROR=ON && cmake --build build -j`.

### Workstream A: convert (`include/lanecast/convert.hpp`, `src/kernels/convert/*`, `tests/test_convert.cpp`)

| Unit | Owns | Depends | Acceptance |
|------|------|---------|------------|
| A1 `convert-yuv2rgb` | convert.hpp, convert.cpp, kernels.hpp, scalar.cpp, test_convert.cpp | skeleton | `B && ctest --test-dir build -R '^convert$' --output-on-failure` passes; `convert_supported` matches the header table; yuv420p and nv12 to rgb24/rgba/bgra scalar with known values and all-ISA checks; other pairs return not_implemented |
| A2 `convert-rgb2yuv` | same files | A1 | same command; rgb to yuv420p/nv12, swizzles, uv (de)interleave, gray paths, copy; no pair returns not_implemented |
| A3 `convert-neon` | convert/neon.cpp | A2 | same command passes with `LANECAST_ISA` unset and `=neon` |
| A4 `convert-x86` | convert/sse41.cpp, convert/avx2.cpp | A2 | x86 compile check builds; CI ubuntu jobs green |

### Workstream B: scale (`include/lanecast/scale.hpp`, `src/kernels/scale/*`, `tests/test_scale.cpp`)

| Unit | Owns | Depends | Acceptance |
|------|------|---------|------------|
| B1 `scale-scalar` | scale.hpp, scale.cpp, kernels.hpp, scalar.cpp, test_scale.cpp | skeleton | `B && ctest --test-dir build -R '^scale$' --output-on-failure` passes; filter table invariants; Scaler for all six formats, bilinear and area |
| B2 `scale-neon` | scale/neon.cpp | B1 | same command with `LANECAST_ISA` unset and `=neon` |
| B3 `scale-x86` | scale/sse41.cpp, scale/avx2.cpp | B1 | x86 compile check builds; CI green |

### Workstream C: io, pipeline, CLI, fused (`include/lanecast/{io,pipeline}.hpp`, `src/io/*`, `src/pipeline/*`, `tools/lanecast/*`, `tests/test_{io,pipeline,cli,fused}.cpp`)

| Unit | Owns | Depends | Acceptance |
|------|------|---------|------------|
| C1 `io` | io.hpp, src/io/{io,y4m,raw,pnm}.cpp, test_io.cpp | skeleton | `B && ctest --test-dir build -R '^io$' --output-on-failure` passes; round trips and malformed input cases |
| C2 `pipeline` | pipeline.hpp, src/pipeline/{thread_pool,pipeline}.cpp (+ chain_parse.cpp), test_pipeline.cpp | A1, B1 | `ctest -R '^pipeline$'` passes; persistent pool; scale and format stages; blur and eq stages wired to the filter API |
| C3 `cli` | tools/lanecast/*, test_cli.cpp | C1, C2 | `ctest -R '^cli$'` passes; `./build/tools/lanecast convert -i in.y4m -o out.ppm -vf scale=320:-1,format=rgb24` exits 0 |
| F1 `fused-scale-convert` | src/pipeline/fused.cpp, test_fused.cpp, auto-fusion in pipeline.cpp | A1, B1, C2 | `ctest -R '^fused$'` passes; ScaleConvert equals scale then convert for all filters, matrices, ranges, dst formats; threaded equals single |

### Workstream D: filter and metrics (`include/lanecast/{filter,metrics}.hpp`, `src/kernels/{filter,metrics}/*`, `tests/test_{filter,metrics}.cpp`)

| Unit | Owns | Depends | Acceptance |
|------|------|---------|------------|
| D1 `filter-scalar` | filter.hpp, filter/{filter,kernels,scalar}, test_filter.cpp | skeleton | `ctest -R '^filter$'` passes; blend, blur 3 and 5, LUT known values; Pipeline `blur=3` equals `blur()` |
| D2 `metrics-scalar` | metrics.hpp, metrics/{metrics,kernels,scalar}, test_metrics.cpp | skeleton | `ctest -R '^metrics$'` passes; PSNR, SSIM, SAD known values; SSIM equals a double reference of the spec |
| D3 `filter-simd` | filter/{neon,sse41,avx2}.cpp | D1 | `ctest -R '^filter$'` passes on arm64; x86 compile check; CI green. Split per ISA if over 400 lines |
| D4 `metrics-simd` | metrics/{neon,neon_dotprod,sse41,avx2}.cpp | D2 | `ctest -R '^metrics$'` passes with `LANECAST_ISA` unset, `=neon`, `=neon_dotprod`; CI green |

### Workstream E: benchmarks and docs (`bench/*.cpp`, `bench/*.hpp`, `bench/ffmpeg/*`, `README.md`, `scripts/*`)

| Unit | Owns | Depends | Acceptance |
|------|------|---------|------------|
| E1 `bench` | bench/main.cpp, bench/bench.hpp, bench/bench_{convert,scale,filter,metrics}.cpp | A1, B1 | `./build/bench/lanecast_bench --quick` exits 0 and prints one row per kernel and supported ISA with Mpix/s and speedup vs scalar |
| E2 `bench-vs-swscale` | bench/ffmpeg/{main.cpp,compare.hpp,swscale_ops.cpp} | E1, F1, A2 | `./build/bench/bench_vs_ffmpeg --quick` exits 0; convert, scale and fused rows with ns/frame, speedup, PSNR vs reference and vs ffmpeg, threads 1 and N |
| E3 `bench-vs-avfilter` | bench/ffmpeg/avfilter_ops.cpp | E2, D2 | same command adds blur, LUT, blend, PSNR, SSIM rows; skipped cleanly when built with `LANECAST_HAVE_AVFILTER=0` |
| E4 `docs-compare` | README.md, scripts/compare_ffmpeg.sh | C3, D2 | `scripts/compare_ffmpeg.sh` generates a testsrc Y4M with the ffmpeg CLI, converts it with both tools and prints PSNR; README documents build, CLI, ISA override and how to read bench output |

### Waves (at most 3 units in parallel; each wave branches from main after the previous wave merges)

| Wave | Dev 1 | Dev 2 | Dev 3 |
|------|-------|-------|-------|
| 1 | A1 | B1 | C1 |
| 2 | A2 | B2 | C2 |
| 3 | A3 | B3 | C3 |
| 4 | A4 | D1 | F1 |
| 5 | D2 | D3 | E1 |
| 6 | D4 | | E2 |
| 7 | | E3 | E4 |

## Public repo hygiene

No host names, user names, absolute local paths or machine specs in code,
docs, commit messages, test fixtures or committed bench output. Bench
binaries print ISA tier names only, never CPU model strings. No em dashes.
