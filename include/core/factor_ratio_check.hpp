#pragma once

#include "core/model.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace vibestran {

struct FactorRatioCheckPolicy {
  double maxratio{1.0e7};
  bool fatal{true};
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

  policy.maxratio = model.param_real("MAXRATIO").value_or(policy.maxratio);
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
