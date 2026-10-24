# lanecast

A portable C++20, SIMD-first raw video toolkit inspired by FFmpeg: pixel
format conversion, scaling, filters, quality metrics, Y4M/PPM IO and a
slice-threaded CLI pipeline. Hand-written NEON, SSE4.1 and AVX2 kernels with
runtime dispatch, each bit-exact against a scalar reference. The goal is to
beat libswscale and libavfilter on every op lanecast implements.

Status: skeleton. See [ARCHITECTURE.md](ARCHITECTURE.md).

```sh
cmake -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
./build/tools/lanecast info
```

`LANECAST_ISA=scalar|neon|neon_dotprod|neon_i8mm|sse41|avx2` forces an ISA tier.

License: MIT.
