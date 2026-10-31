#pragma once
// include/core/notices.hpp
// Warnings about how a run was carried out (a precision or method fallback,
// a MAXRATIO breach, an ill-conditioned solve) that belong in the F06 as well
// as the log. Free of spdlog so CUDA units can include it.

#include "core/logger.hpp"
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace vibestran {

class Notices {
public:
  /// Logs a warning and keeps it for the F06.
  void add(std::string text) {
    log_warn(text);
    items_.push_back(std::move(text));
  }
  /// Keeps notices that were already logged, e.g. by an inner solver.
  void keep(std::vector<std::string> logged) {
    items_.insert(items_.end(), std::make_move_iterator(logged.begin()),
                  std::make_move_iterator(logged.end()));
  }
  /// The notices since the last call, oldest first.
  [[nodiscard]] std::vector<std::string> take() { return std::exchange(items_, {}); }

private:
  std::vector<std::string> items_;
};

} // namespace vibestran
