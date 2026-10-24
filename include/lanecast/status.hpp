// Error reporting shared by every lanecast API. No exceptions cross the API.
#pragma once

#include <cstdint>
#include <string_view>

namespace lanecast {

enum class Status : std::uint8_t {
  ok = 0,
  invalid_argument,    // bad dimensions, mismatched frames, bad row range
  unsupported_format,  // the op does not support this pixel format pair
  not_implemented,     // declared in the v1 API but not built yet
  out_of_memory,
  io_error,            // open, read or write failed
  parse_error,         // malformed file header or filter chain
  end_of_stream,       // reader has no more frames (not a failure)
};

std::string_view status_name(Status s) noexcept;

}  // namespace lanecast
