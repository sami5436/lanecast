# One test executable; one ctest entry per tests/test_<suite>.cpp file,
# running only that file's tests (lanecast_tests --suite <suite>).
file(GLOB _lc_test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
add_executable(lanecast_tests "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" ${_lc_test_sources})
target_link_libraries(lanecast_tests PRIVATE lanecast)
target_include_directories(lanecast_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
if(TARGET lanecast_cli)
  add_dependencies(lanecast_tests lanecast_cli)
  target_compile_definitions(lanecast_tests PRIVATE
    LANECAST_CLI_PATH="$<TARGET_FILE:lanecast_cli>")
endif()

foreach(_src IN LISTS _lc_test_sources)
  get_filename_component(_stem "${_src}" NAME_WE)
  string(REGEX REPLACE "^test_" "" _suite "${_stem}")
  add_test(NAME ${_suite} COMMAND lanecast_tests --suite ${_suite}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
