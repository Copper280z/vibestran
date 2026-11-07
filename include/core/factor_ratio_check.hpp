#pragma once

#include "core/model.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace vibestran {

/// A K_ii/D_i above this is taken as a singular K unless the deck sets PARAM,MAXRATIO.
/// Measured: at least 1.39e12 with a null mode (or an exact zero pivot), at most 6.6e9 without,
/// and the ratio moves by a factor of 40 with the ordering, so this sits midway between them.
/// Nastran's 1e7 would stop healthy models: a stiffened panel reaches 2.15e7.
inline constexpr double kDefaultFatalRatio = 1.0e11;

struct FactorRatioCheckPolicy {
  double maxratio{1.0e7};                 ///< A breach is a warning.
  double fatal_ratio{kDefaultFatalRatio}; ///< A breach is fatal, when fatal is set.
  bool fatal{true};                       ///< PARAM,BAILOUT,0 and not lenient.
  std::string context;
  std::string matrix_name;
};

enum class FactorRatioCheckStatus {
  Ok,
  RatioExceeded,
  NonPositiveFactorDiagonal,
};

struct FactorRatioCheckResult {
  FactorRatioCheckStatus status{FactorRatioCheckStatus::Ok};
  double max_ratio{0.0};
  int row_1based{0};
  double matrix_diag{0.0};
  double factor_diag{0.0};
};

inline FactorRatioCheckPolicy
build_factor_ratio_check_policy(const Model &model, std::string context,
                                std::string matrix_name) {
  FactorRatioCheckPolicy policy;
  policy.context = std::move(context);
  policy.matrix_name = std::move(matrix_name);

  // A limit in the deck is what the deck asks to stop at, as in Nastran.
  if (const auto maxratio = model.param_real("MAXRATIO")) {
    policy.maxratio = *maxratio;
    policy.fatal_ratio = *maxratio;
  }
  const auto bailout_param = model.param_integer("BAILOUT");
  int bailout = bailout_param.value_or(0);

  const auto cm_it = model.params.find("CHECKMODE");
  const bool lenient =
      (cm_it != model.params.end() && cm_it->second == "LENIENT");
  if (lenient && !bailout_param)
    bailout = -1;

  policy.fatal = (bailout == 0 && !lenient);
  return policy;
}

inline FactorRatioCheckResult
evaluate_factor_ratio_check(std::span<const double> matrix_diag,
                            std::span<const double> factor_diag,
                            double maxratio,
                            double factor_diag_epsilon = 1.0e-14) {
  if (matrix_diag.size() != factor_diag.size()) {
    throw std::invalid_argument(
        "Factor ratio check requires equal matrix and factor diagonal lengths");
  }

  FactorRatioCheckResult result;
  for (std::size_t i = 0; i < matrix_diag.size(); ++i) {
    const double mdiag = matrix_diag[i];
    const double fdiag = factor_diag[i];
    if (fdiag <= factor_diag_epsilon) {
      result.status = FactorRatioCheckStatus::NonPositiveFactorDiagonal;
      result.row_1based = static_cast<int>(i) + 1;
      result.matrix_diag = mdiag;
      result.factor_diag = fdiag;
      return result;
    }

    const double ratio = std::abs(mdiag / fdiag);
    if (ratio > result.max_ratio) {
      result.max_ratio = ratio;
      result.row_1based = static_cast<int>(i) + 1;
      result.matrix_diag = mdiag;
      result.factor_diag = fdiag;
    }
  }

  if (result.max_ratio > maxratio)
    result.status = FactorRatioCheckStatus::RatioExceeded;
  return result;
}

} // namespace vibestran
