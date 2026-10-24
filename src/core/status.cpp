#include "lanecast/status.hpp"

namespace lanecast {

std::string_view status_name(Status s) noexcept {
  switch (s) {
    case Status::ok: return "ok";
    case Status::invalid_argument: return "invalid_argument";
    case Status::unsupported_format: return "unsupported_format";
    case Status::not_implemented: return "not_implemented";
    case Status::out_of_memory: return "out_of_memory";
    case Status::io_error: return "io_error";
    case Status::parse_error: return "parse_error";
    case Status::end_of_stream: return "end_of_stream";
  }
  return "unknown";
}

}  // namespace lanecast
