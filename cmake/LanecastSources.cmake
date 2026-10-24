# lanecast_collect_sources(<out_var> <dir>...)
#
# Globs <dir>/*.cpp (relative to the project root) and classifies each file by
# the suffix of its stem:
#   neon.cpp,         *_neon.cpp          AArch64 only, baseline flags
#   neon_dotprod.cpp, *_neon_dotprod.cpp  AArch64 only, LANECAST_FLAGS_NEON_DOTPROD
#   neon_i8mm.cpp,    *_neon_i8mm.cpp     AArch64 only, LANECAST_FLAGS_NEON_I8MM
#   sse41.cpp,        *_sse41.cpp         x86-64 only, LANECAST_FLAGS_SSE41
#   avx2.cpp,         *_avx2.cpp          x86-64 only, LANECAST_FLAGS_AVX2
#   anything else                         every platform, baseline flags
# ISA flags are applied per source file, never to the whole target, so the
# library stays runnable on any CPU of the target architecture.
function(lanecast_collect_sources out_var)
  set(_result "")
  foreach(_dir IN LISTS ARGN)
    file(GLOB _files CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/${_dir}/*.cpp")
    list(SORT _files)
    foreach(_f IN LISTS _files)
      get_filename_component(_stem "${_f}" NAME_WE)
      set(_isa "")
      if(_stem MATCHES "(^|_)neon_dotprod$")
        set(_isa NEON_DOTPROD)
      elseif(_stem MATCHES "(^|_)neon_i8mm$")
        set(_isa NEON_I8MM)
      elseif(_stem MATCHES "(^|_)neon$")
        set(_isa NEON)
      elseif(_stem MATCHES "(^|_)sse41$")
        set(_isa SSE41)
      elseif(_stem MATCHES "(^|_)avx2$")
        set(_isa AVX2)
      endif()
      if(_isa STREQUAL "")
        list(APPEND _result "${_f}")
      elseif(LANECAST_HAVE_${_isa})
        list(APPEND _result "${_f}")
        if(NOT LANECAST_FLAGS_${_isa} STREQUAL "")
          separate_arguments(_flags UNIX_COMMAND "${LANECAST_FLAGS_${_isa}}")
          set_source_files_properties("${_f}" DIRECTORY "${PROJECT_SOURCE_DIR}"
            PROPERTIES COMPILE_OPTIONS "${_flags}")
        endif()
      endif()
    endforeach()
  endforeach()
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()
