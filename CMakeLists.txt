cmake_minimum_required(VERSION 3.20)
project(lanecast VERSION 0.1.0 LANGUAGES CXX)

# Release by default for single-config generators (Unix Makefiles, Ninja).
get_property(_lc_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _lc_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LANECAST_BUILD_TESTS "Build the test runner" ON)
option(LANECAST_BUILD_BENCH "Build the microbenchmarks" ON)
option(LANECAST_BUILD_TOOLS "Build the lanecast CLI" ON)
option(LANECAST_ENABLE_SIMD "Compile SIMD kernels (OFF builds scalar only)" ON)
option(LANECAST_WERROR "Treat warnings as errors" OFF)
set(LANECAST_BENCH_FFMPEG "AUTO" CACHE STRING
  "Build bench_vs_ffmpeg against libswscale/libavfilter: AUTO, ON or OFF")
set_property(CACHE LANECAST_BENCH_FFMPEG PROPERTY STRINGS AUTO ON OFF)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "lanecast supports GCC and Clang only")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(LanecastArch)
include(LanecastSources)

add_compile_options(-Wall -Wextra -Wpedantic -ffp-contract=off)
if(LANECAST_WERROR)
  add_compile_options(-Werror)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
endif()

find_package(Threads REQUIRED)

# Every .cpp under these directories is picked up by glob. Units add files
# without editing this file. ISA files are recognized by their name suffix,
# see cmake/LanecastSources.cmake.
lanecast_collect_sources(_lc_sources src/core src/io src/pipeline)
file(GLOB _lc_kernel_dirs LIST_DIRECTORIES true CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/src/kernels/*")
foreach(_dir IN LISTS _lc_kernel_dirs)
  if(IS_DIRECTORY "${_dir}")
    file(RELATIVE_PATH _rel "${PROJECT_SOURCE_DIR}" "${_dir}")
    lanecast_collect_sources(_lc_kernel_sources "${_rel}")
    list(APPEND _lc_sources ${_lc_kernel_sources})
  endif()
endforeach()

add_library(lanecast STATIC ${_lc_sources})
target_include_directories(lanecast
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lanecast PUBLIC Threads::Threads)
target_compile_definitions(lanecast PUBLIC
  LANECAST_HAVE_NEON=$<BOOL:${LANECAST_HAVE_NEON}>
  LANECAST_HAVE_NEON_DOTPROD=$<BOOL:${LANECAST_HAVE_NEON_DOTPROD}>
  LANECAST_HAVE_NEON_I8MM=$<BOOL:${LANECAST_HAVE_NEON_I8MM}>
  LANECAST_HAVE_SSE41=$<BOOL:${LANECAST_HAVE_SSE41}>
  LANECAST_HAVE_AVX2=$<BOOL:${LANECAST_HAVE_AVX2}>)

if(LANECAST_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(LANECAST_BUILD_BENCH)
  add_subdirectory(bench)
endif()
if(LANECAST_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "lanecast: arch=${LANECAST_ARCH} neon=${LANECAST_HAVE_NEON} "
  "neon_dotprod=${LANECAST_HAVE_NEON_DOTPROD} neon_i8mm=${LANECAST_HAVE_NEON_I8MM} "
  "sse41=${LANECAST_HAVE_SSE41} avx2=${LANECAST_HAVE_AVX2}")
