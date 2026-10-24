// Minimal dependency-free test runner.
//
//   TEST(name) { CHECK(x); CHECK_EQ(a, b); REQUIRE_OK(status); }
//
// Tests register at static init. A test's suite is its file name without
// "test_" and ".cpp" (tests/test_convert.cpp -> "convert"); ctest runs one
// suite per entry: lanecast_tests --suite convert.
// CHECK* records a failure and continues; REQUIRE* aborts the current test.
#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "lanecast/cpu.hpp"
#include "lanecast/status.hpp"

namespace lanecast::test {

struct TestCase {
  const char* name;
  const char* file;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, const char* file, void (*fn)()) {
    registry().push_back({name, file, fn});
  }
};

struct AbortTest {};

inline int& failure_count() {
  static int n = 0;
  return n;
}

inline std::string_view base_name(std::string_view path) {
  const auto slash = path.find_last_of("/\\");
  return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

inline std::string suite_of(const char* file) {
  std::string_view s = base_name(file);
  if (const auto dot = s.find('.'); dot != std::string_view::npos) s = s.substr(0, dot);
  if (s.substr(0, 5) == "test_") s = s.substr(5);
  return std::string(s);
}

inline void report_failure(const char* file, int line, const std::string& what) {
  ++failure_count();
  const std::string_view f = base_name(file);
  std::fprintf(stderr, "    FAIL %.*s:%d: %s\n", static_cast<int>(f.size()), f.data(), line,
               what.c_str());
}

template <class T>
std::string show(const T& v) {
  if constexpr (std::is_same_v<T, Status>) {
    return std::string(status_name(v));
  } else if constexpr (std::is_same_v<T, Isa>) {
    return std::string(isa_name(v));
  } else if constexpr (std::is_enum_v<T>) {
    return std::to_string(static_cast<long long>(v));
  } else if constexpr (std::is_same_v<T, unsigned char> || std::is_same_v<T, signed char>) {
    return std::to_string(static_cast<int>(v));
  } else {
    std::ostringstream os;
    os << v;
    return os.str();
  }
}

inline int run(int argc, char** argv) {
  std::string suite;
  std::string filter;
  bool list = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--suite" && i + 1 < argc) {
      suite = argv[++i];
    } else if (a == "--filter" && i + 1 < argc) {
      filter = argv[++i];
    } else if (a == "--list") {
      list = true;
    } else {
      std::fprintf(stderr, "usage: lanecast_tests [--suite NAME] [--filter SUBSTR] [--list]\n");
      return 2;
    }
  }

  int ran = 0;
  int failed = 0;
  for (const TestCase& tc : registry()) {
    const std::string s = suite_of(tc.file);
    if (!suite.empty() && s != suite) continue;
    if (!filter.empty() && std::string_view(tc.name).find(filter) == std::string_view::npos) {
      continue;
    }
    if (list) {
      std::printf("%s.%s\n", s.c_str(), tc.name);
      continue;
    }
    ++ran;
    const int before = failure_count();
    std::printf("[ RUN  ] %s.%s\n", s.c_str(), tc.name);
    try {
      tc.fn();
    } catch (const AbortTest&) {
      // Failure already recorded by REQUIRE.
    } catch (const std::exception& e) {
      report_failure(tc.file, 0, std::string("uncaught exception: ") + e.what());
    } catch (...) {
      report_failure(tc.file, 0, "uncaught non-standard exception");
    }
    const bool ok = failure_count() == before;
    if (!ok) ++failed;
    std::printf("[ %s ] %s.%s\n", ok ? " OK " : "FAIL", s.c_str(), tc.name);
  }
  if (list) return 0;
  const std::string_view active = isa_name(active_isa());
  std::printf("%d test(s), %d failed, active isa %.*s\n", ran, failed,
              static_cast<int>(active.size()), active.data());
  if (ran == 0) std::printf("note: no tests registered for this selection yet\n");
  return failed == 0 ? 0 : 1;
}

}  // namespace lanecast::test

#define LC_TEST_CAT2(a, b) a##b
#define LC_TEST_CAT(a, b) LC_TEST_CAT2(a, b)

#define TEST(name)                                                                    \
  static void LC_TEST_CAT(lc_test_, name)();                                          \
  static const ::lanecast::test::Registrar LC_TEST_CAT(lc_test_reg_, name)(           \
      #name, __FILE__, &LC_TEST_CAT(lc_test_, name));                                 \
  static void LC_TEST_CAT(lc_test_, name)()

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) ::lanecast::test::report_failure(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_MSG(cond, msg)                                                          \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      ::lanecast::test::report_failure(__FILE__, __LINE__,                            \
                                       std::string("CHECK(" #cond "): ") + (msg));    \
    }                                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                                \
  do {                                                                                \
    const auto& lc_a = (a);                                                           \
    const auto& lc_b = (b);                                                           \
    if (!(lc_a == lc_b)) {                                                            \
      ::lanecast::test::report_failure(                                               \
          __FILE__, __LINE__,                                                         \
          "CHECK_EQ(" #a ", " #b "): " + ::lanecast::test::show(lc_a) + " vs " +      \
              ::lanecast::test::show(lc_b));                                          \
    }                                                                                 \
  } while (0)

#define REQUIRE(cond)                                                                 \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      ::lanecast::test::report_failure(__FILE__, __LINE__, "REQUIRE(" #cond ")");     \
      throw ::lanecast::test::AbortTest{};                                            \
    }                                                                                 \
  } while (0)

#define REQUIRE_OK(expr)                                                              \
  do {                                                                                \
    const ::lanecast::Status lc_s = (expr);                                           \
    if (lc_s != ::lanecast::Status::ok) {                                             \
      ::lanecast::test::report_failure(                                               \
          __FILE__, __LINE__, "REQUIRE_OK(" #expr "): " + ::lanecast::test::show(lc_s)); \
      throw ::lanecast::test::AbortTest{};                                            \
    }                                                                                 \
  } while (0)
