// Filter chain text -> FilterSpec list. Checks grammar, filter names, option
// names and positional counts; option values are checked when the Pipeline
// resolves them against the input (pipeline.cpp).
#include <algorithm>
#include <cstddef>
#include <string_view>

#include "lanecast/pipeline.hpp"

namespace lanecast {
namespace {

struct FilterSchema {
  std::string_view name;
  // Positional argument i is stored under positional[i]; keyed arguments
  // may use any name in `keys`.
  std::vector<std::string_view> positional;
  std::vector<std::string_view> keys;
  std::vector<std::string_view> required;
};

const std::vector<FilterSchema>& schemas() {
  static const std::vector<FilterSchema> s = {
      {"scale", {"w", "h"}, {"w", "h", "flags"}, {"w", "h"}},
      {"format", {"pix_fmt"}, {"pix_fmt", "matrix", "range", "chroma", "chroma_loc"}, {"pix_fmt"}},
      {"blur", {"size"}, {"size"}, {"size"}},
      {"eq", {}, {"brightness", "contrast"}, {}},
  };
  return s;
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

std::vector<std::string_view> split(std::string_view s, char sep) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  for (;;) {
    const std::size_t pos = s.find(sep, start);
    if (pos == std::string_view::npos) {
      parts.push_back(s.substr(start));
      return parts;
    }
    parts.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
}

Status fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return Status::parse_error;
}

bool contains(const std::vector<std::string_view>& v, std::string_view s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

}  // namespace

Status parse_filter_chain(std::string_view text, std::vector<FilterSpec>& out,
                          std::string* error) {
  out.clear();
  if (trim(text).empty()) return Status::ok;

  for (const std::string_view raw : split(text, ',')) {
    const std::string_view filter = trim(raw);
    if (filter.empty()) return fail(error, "empty filter in chain");
    const std::size_t eq = filter.find('=');
    const std::string_view name = trim(filter.substr(0, eq));
    if (name.empty()) return fail(error, "filter without a name");

    const auto schema = std::find_if(schemas().begin(), schemas().end(),
                                     [&](const FilterSchema& f) { return f.name == name; });
    if (schema == schemas().end()) {
      return fail(error, "unknown filter '" + std::string(name) + "'");
    }
    const std::string prefix = "filter '" + std::string(name) + "': ";

    FilterSpec spec;
    spec.name = std::string(name);
    if (eq != std::string_view::npos) {
      std::size_t positional = 0;
      bool keyed = false;
      for (const std::string_view raw_arg : split(filter.substr(eq + 1), ':')) {
        const std::string_view arg = trim(raw_arg);
        if (arg.empty()) return fail(error, prefix + "empty argument");
        std::string key;
        std::string value;
        const std::size_t kv = arg.find('=');
        if (kv == std::string_view::npos) {
          if (keyed) return fail(error, prefix + "positional argument '" + std::string(arg) +
                                            "' after a keyed one");
          if (positional >= schema->positional.size()) {
            return fail(error, prefix + "too many arguments");
          }
          key = std::string(schema->positional[positional++]);
          value = std::string(arg);
        } else {
          keyed = true;
          key = std::string(trim(arg.substr(0, kv)));
          value = std::string(trim(arg.substr(kv + 1)));
          if (!contains(schema->keys, key)) {
            return fail(error, prefix + "unknown option '" + key + "'");
          }
          if (value.empty()) return fail(error, prefix + "option '" + key + "' has no value");
        }
        const bool duplicate =
            std::any_of(spec.args.begin(), spec.args.end(),
                        [&](const auto& kvp) { return kvp.first == key; });
        if (duplicate) return fail(error, prefix + "option '" + key + "' given twice");
        spec.args.emplace_back(std::move(key), std::move(value));
      }
    }
    for (const std::string_view key : schema->required) {
      const bool present = std::any_of(spec.args.begin(), spec.args.end(),
                                       [&](const auto& kvp) { return kvp.first == key; });
      if (!present) return fail(error, prefix + "missing '" + std::string(key) + "'");
    }
    out.push_back(std::move(spec));
  }
  return Status::ok;
}

}  // namespace lanecast
