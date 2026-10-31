#pragma once
// include/core/numbers.hpp
// Non-throwing parsing of Nastran numeric text. A value must be the whole
// field (surrounding blanks aside); anything else is nullopt, so callers
// report malformed input instead of silently using a prefix or a default.
//
// Kept free of <format> so CUDA compilation units (GCC 12) can include it.

#include <charconv>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>

namespace vibestran {

namespace detail {
inline std::string_view trim_blanks(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos)
    return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}
} // namespace detail

/// An integer field: optional sign and digits only, within int range.
/// "12.5", "12abc", "" and "+" are all nullopt.
[[nodiscard]] inline std::optional<int> parse_integer(std::string_view text) {
  text = detail::trim_blanks(text);
  if (!text.empty() && text.front() == '+')
    text.remove_prefix(1);
  if (text.empty() || text.front() == '+')
    return std::nullopt;
  int value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size())
    return std::nullopt;
  return value;
}

/// A finite real field in any Nastran form: 1.5E+3, 1.5D+3, 1.5+3 (implicit
/// exponent), .5, 5., or an integer such as 12. "1.5abc", "inf" and an
/// out-of-range exponent are nullopt.
[[nodiscard]] inline std::optional<double> parse_real(std::string_view text) {
  text = detail::trim_blanks(text);
  if (!text.empty() && text.front() == '+')
    text.remove_prefix(1);
  if (text.empty() || text.front() == '+')
    return std::nullopt;
  std::string clean(text);
  bool exponent = false;
  for (size_t i = 0; i < clean.size(); ++i) {
    char &c = clean[i];
    if (c == 'D' || c == 'd' || c == 'e')
      c = 'E';
    if (c == 'E') {
      exponent = true;
    } else if (!exponent && i > 0 && (c == '+' || c == '-')) {
      // Implicit exponent: the first sign after the mantissa starts it.
      clean.insert(i, 1, 'E');
      exponent = true;
      ++i;
    }
  }
  double value = 0;
  const auto [end, error] =
      std::from_chars(clean.data(), clean.data() + clean.size(), value);
  if (error != std::errc{} || end != clean.data() + clean.size() ||
      !std::isfinite(value))
    return std::nullopt;
  return value;
}

} // namespace vibestran
