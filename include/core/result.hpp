#pragma once
// include/core/result.hpp
// Error values for recoverable failures (bad input, a matrix that cannot be
// factored, a solve that did not converge). Exceptions are reserved for
// programming errors; see CLAUDE.md.

#include <optional>
#include <string>
#include <variant>

namespace vibestran {

struct Error {
  std::string message;
};
/// A value or the reason there is none.
template <class T> using Result = std::variant<T, Error>;
/// Success (nullopt) or the reason for failure.
using Status = std::optional<Error>;

} // namespace vibestran
