#pragma once
// CHOLMOD records failures in cholmod_common::status, which each call resets.
// Eigen's CholmodBase does not check it: analyzePattern() reports success
// when cholmod_analyze() returned no factor (out of memory, or a factor too
// large for the int interface), and factorize() then dereferences that
// missing factor. Callers check the status after each step instead.
#if defined(EIGEN_CHOLMOD_SUPPORT)
#include "core/result.hpp"
#include <Eigen/CholmodSupport>
#include <iterator>
#include <numeric>
#include <format>
#include <string>
#include <string_view>

namespace vibestran::detail {

inline std::string_view cholmod_status_text(int status) {
  switch (status) {
  case CHOLMOD_NOT_INSTALLED:
    return "a required CHOLMOD method is not installed";
  case CHOLMOD_OUT_OF_MEMORY:
    return "out of memory";
  case CHOLMOD_TOO_LARGE:
    return "integer overflow: the problem exceeds the 32-bit index range of "
           "this CHOLMOD interface";
  case CHOLMOD_INVALID:
    return "invalid input";
  case CHOLMOD_GPU_PROBLEM:
    return "GPU failure";
  default:
    return "unknown error";
  }
}

/// An error when the last CHOLMOD call failed (negative status; positive
/// values are warnings such as CHOLMOD_NOT_POSDEF). step names the call.
[[nodiscard]] inline Status cholmod_failure(const cholmod_common &common,
                                            std::string_view label,
                                            std::string_view step, long n,
                                            long nnz) {
  if (common.status >= CHOLMOD_OK)
    return std::nullopt;
  // The fill of the best ordering CHOLMOD finished evaluating, when any.
  const double lnz = std::accumulate(
      std::begin(common.method), std::end(common.method), common.lnz > 0 ? common.lnz : 0.0,
      [](double best, const auto &method) {
        return method.lnz > 0 && (best <= 0 || method.lnz < best) ? method.lnz : best;
      });
  const std::string fill =
      lnz > 0 ? std::format(", estimated nnz(L)={:.4g} ({:.1f} GiB of factor values)",
                            lnz, lnz * 8.0 / (1024.0 * 1024.0 * 1024.0))
              : std::string{};
  return Error{std::format("{} {} failed: {} (CHOLMOD status {}; n={}, nnz(lower)={}{})",
                           label, step, cholmod_status_text(common.status),
                           common.status, n, nnz, fill)};
}

} // namespace vibestran::detail
#endif
