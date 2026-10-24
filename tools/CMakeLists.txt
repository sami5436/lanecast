file(GLOB _lc_cli_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lanecast/*.cpp")
add_executable(lanecast_cli ${_lc_cli_sources})
set_target_properties(lanecast_cli PROPERTIES OUTPUT_NAME lanecast)
target_link_libraries(lanecast_cli PRIVATE lanecast)
