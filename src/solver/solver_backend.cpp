// src/solver/solver_backend.cpp
#include "solver/solver_backend.hpp"
#include "amg_cpu.hpp"
#include "core/factor_ratio_check.hpp"
#include "core/types.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <type_traits>
#ifdef HAVE_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif
#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#include "cholmod_status.hpp"
#endif
#include <format>
#include <spdlog/spdlog.h>

namespace vibestran {

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;
using ESM = Eigen::SparseMatrix<double>;
using ESMR = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using CsrData = SparseMatrixBuilder::CsrData;

/// Shape errors shared by every factor() implementation.
Status check_matrix(const CsrData &K) {
  if (K.n == 0)
    return Error{"Matrix is empty — no free DOFs"};
  if (K.row_ptr.size() != static_cast<std::size_t>(K.n) + 1 ||
      K.col_ind.size() != static_cast<std::size_t>(K.nnz) ||
      K.values.size() != static_cast<std::size_t>(K.nnz))
    return Error{std::format("Malformed CSR matrix: n={}, nnz={}, row_ptr {}, "
                             "col_ind {}, values {}",
                             K.n, K.nnz, K.row_ptr.size(), K.col_ind.size(),
                             K.values.size())};
  return std::nullopt;
}

Status check_rhs(int n, std::span<const double> B, int nrhs) {
  if (nrhs < 1 || B.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs))
    return Error{std::format("Right-hand side has {} values; expected {} x {}",
                             B.size(), n, nrhs)};
  return std::nullopt;
}

/// Lower triangle of K in compressed column-major storage, the input of every
/// direct factorization below. A full mirrored matrix is reduced to its lower
/// triangle so the factorization sees symmetric storage either way.
ESM lower_csc(const CsrData &K) {
  const Eigen::Map<const ESMR> K_row(K.n, K.n, K.nnz, K.row_ptr.data(),
                                     K.col_ind.data(), K.values.data());
  ESM lower;
  if (K.stores_lower_triangle_only())
    lower = K_row;
  else
    lower = K_row.triangularView<Eigen::Lower>();
  lower.makeCompressed();
  return lower;
}

#if !defined(HAVE_ACCELERATE)
std::vector<double> extract_csr_diagonal(const CsrData &csr) {
  std::vector<double> diag(static_cast<std::size_t>(csr.n), 0.0);
  for (int row = 0; row < csr.n; ++row) {
    for (int j = csr.row_ptr[row]; j < csr.row_ptr[row + 1]; ++j) {
      if (csr.col_ind[j] == row) {
        diag[static_cast<std::size_t>(row)] = csr.values[j];
        break;
      }
    }
  }
  return diag;
}

template <typename DiagonalExpr>
std::vector<double> copy_diagonal(const DiagonalExpr &diag_expr) {
  Eigen::VectorXd dense = diag_expr;
  return std::vector<double>(dense.data(), dense.data() + dense.size());
}

/// An error for a non-positive factor diagonal or a fatal MAXRATIO breach; a
/// non-fatal breach is a notice.
Status factor_ratio_status(const FactorRatioCheckPolicy &policy,
                           const FactorRatioCheckResult &result,
                           std::string_view backend_label, Notices &notices) {
  if (result.status == FactorRatioCheckStatus::Ok)
    return std::nullopt;

  const std::string matrix_name =
      policy.matrix_name.empty() ? "matrix" : policy.matrix_name;
  if (result.status == FactorRatioCheckStatus::NonPositiveFactorDiagonal) {
    return Error{std::format(
        "{} {} has non-positive or near-zero factor diagonal at row {} "
        "(matrix diag {:.3e}, factor diag {:.3e}, backend {})",
        policy.context, matrix_name, result.row_1based, result.matrix_diag,
        result.factor_diag, backend_label)};
  }

  std::string message = std::format(
      "{} {} diagonal / factor diagonal ratio {:.3e} exceeds "
      "MAXRATIO={:.3e} at row {} (matrix diag {:.3e}, factor diag {:.3e}, "
      "backend {})",
      policy.context, matrix_name, result.max_ratio, policy.maxratio,
      result.row_1based, result.matrix_diag, result.factor_diag, backend_label);
  if (policy.fatal)
    return Error{std::move(message)};
  notices.add(std::move(message));
  return std::nullopt;
}
#endif

#if defined(HAVE_ACCELERATE)
struct AccelerateOrderConfig {
  SparseOrder_t order;
  const char *label;
};

// VIBESTRAN_ACCELERATE_ORDER, when set, overrides the choice of minimum
// degree (FactorOptions::dense_block or minimum_degree).
AccelerateOrderConfig accelerate_order_config(bool minimum_degree = false) {
  const char *env = std::getenv("VIBESTRAN_ACCELERATE_ORDER");
  if (env == nullptr || env[0] == '\0')
    return minimum_degree ? AccelerateOrderConfig{SparseOrderAMD, "amd"}
                          : AccelerateOrderConfig{SparseOrderMetis, "metis"};
  if (std::strcmp(env, "default") == 0)
    return {SparseOrderDefault, "default"};
  if (std::strcmp(env, "amd") == 0)
    return {SparseOrderAMD, "amd"};
  if (std::strcmp(env, "metis") == 0)
    return {SparseOrderMetis, "metis"};

  spdlog::warn("Ignoring invalid VIBESTRAN_ACCELERATE_ORDER='{}' "
               "(valid: default, amd, metis)",
               env);
  return {SparseOrderMetis, "metis"};
}

using DirectLDLT = Eigen::AccelerateLDLT<ESM>;
constexpr const char *kLdltLabel = "accelerate-ldlt";
#else
using DirectLDLT = Eigen::SimplicialLDLT<ESM>;
#if defined(EIGEN_CHOLMOD_SUPPORT)
const char *cholmod_order_label(int ordering) {
  switch (ordering) {
  case CHOLMOD_NATURAL:
    return "natural";
  case CHOLMOD_GIVEN:
    return "given";
  case CHOLMOD_AMD:
    return "amd";
  case CHOLMOD_METIS:
    return "metis";
  case CHOLMOD_NESDIS:
    return "nesdis";
  case CHOLMOD_COLAMD:
    return "colamd";
  case CHOLMOD_POSTORDERED:
    return "postordered";
  default:
    return "unknown";
  }
}
#endif
constexpr const char *kLdltLabel = "simplicial-ldlt";
#endif

/// Eigen direct factorization with symbolic reuse. Primary is the Cholesky
/// tried first; without require_spd a failure falls back to LDLT, which
/// handles near-singular and mildly indefinite systems.
template <class Primary> class EigenDirectFactor final : public LinearFactor {
public:
  EigenDirectFactor(const FactorOptions &options, const char *primary_label,
                    Notices &notices)
      : require_spd_(options.require_spd), notices_(notices),
        primary_label_(primary_label), label_(primary_label) {
    if (options.factor_ratio_policy != nullptr)
      policy_ = *options.factor_ratio_policy;
#if defined(HAVE_ACCELERATE)
    const auto order =
        accelerate_order_config(options.dense_block || options.minimum_degree);
    order_ = order.order;
    order_label_ = order.label;
    llt_.setOrder(order_);
#elif defined(EIGEN_CHOLMOD_SUPPORT)
    if constexpr (kCholmod) {
      auto &common = llt_.cholmod();
      if (options.minimum_degree) {
        // Only AMD: by default CHOLMOD also tries METIS when AMD's fill
        // looks high.
        common.nmethods = 1;
        common.method[0].ordering = CHOLMOD_AMD;
      }
      if (options.dense_block)
        for (auto &method : common.method)
          method.prune_dense = -1;
    }
#endif
  }

  [[nodiscard]] int size() const noexcept override { return n_; }
  [[nodiscard]] std::string_view method() const noexcept override { return label_; }
  [[nodiscard]] std::string_view ordering() const noexcept override {
#if defined(HAVE_ACCELERATE)
    return order_label_;
#elif defined(EIGEN_CHOLMOD_SUPPORT)
    if constexpr (kCholmod)
      return order_label_;
    return "amd"; // Eigen's simplicial default, AMDOrdering
#else
    return "amd";
#endif
  }
  [[nodiscard]] FactorStats stats() const noexcept override { return stats_; }

  [[nodiscard]] Status refactor(const CsrData &K) override {
    ready_ = false;
    if (auto e = check_matrix(K))
      return e;
    if (n_ != 0 && K.n != n_)
      return Error{std::format("refactor() of a {}-equation factor with a "
                               "{}-equation matrix",
                               n_, K.n)};
    const bool analyze = !same_pattern(K);
    if (analyze) {
      n_ = K.n;
      lower_only_ = K.stores_lower_triangle_only();
      row_ptr_ = K.row_ptr;
      col_ind_ = K.col_ind;
      llt_analyzed_ = false;
      ldlt_analyzed_ = false;
    }

    const auto t_convert = Clock::now();
    const ESM A = lower_csc(K);
    const auto t_factor = Clock::now();
    if (!llt_analyzed_) {
      llt_.analyzePattern(A);
      ++stats_.symbolic_analyses;
      llt_analyzed_ = llt_.info() == Eigen::Success;
#if defined(EIGEN_CHOLMOD_SUPPORT) && !defined(HAVE_ACCELERATE)
      if constexpr (kCholmod) {
        // No factor to try LDL^T with either: the failure is the problem's
        // size or the memory, not its definiteness.
        if (auto e = cholmod_step_failure("symbolic analysis", A)) {
          llt_analyzed_ = false;
          return e;
        }
        const auto &common = llt_.cholmod();
        order_label_ = common.selected >= 0
                           ? cholmod_order_label(common.method[common.selected].ordering)
                           : "unknown";
      }
#endif
    }
    bool factored = false;
    if (llt_analyzed_) {
      llt_.factorize(A);
      ++stats_.numeric_factorizations;
      factored = llt_.info() == Eigen::Success;
#if defined(EIGEN_CHOLMOD_SUPPORT) && !defined(HAVE_ACCELERATE)
      if constexpr (kCholmod)
        if (auto e = cholmod_step_failure("numeric factorization", A))
          return e;
#endif
    }
    use_ldlt_ = false;
    label_ = primary_label_;
    if (!factored) {
      if (require_spd_)
        return Error{std::format("{} factorization failed: the matrix is not "
                                 "numerically positive definite",
                                 primary_label_)};
      if (auto e = factor_ldlt(A))
        return e;
      notices_.add(std::format("The matrix is not numerically positive definite: "
                               "{} failed and it was solved with {}",
                               primary_label_, kLdltLabel));
    }
    const auto t_done = Clock::now();
    if (auto e = check_factor_ratio(K))
      return e;
    spdlog::debug("[cpu-direct] backend={}, {}storage={}, n={}, nnz(lower)={}, "
                  "symbolic analysis {}, convert CSR->Eigen: {:.3f} ms, "
                  "factorization: {:.3f} ms",
                  label_, order_text(), lower_only_ ? "lower" : "full", n_,
                  A.nonZeros(), analyze ? "new" : "reused",
                  Ms(t_factor - t_convert).count(), Ms(t_done - t_factor).count());
    ready_ = true;
    return std::nullopt;
  }

  [[nodiscard]] Result<std::vector<double>> solve(std::span<const double> B,
                                                  int nrhs) override {
    if (!ready_)
      return Error{"solve() with a factor whose last factorization failed"};
    if (auto e = check_rhs(n_, B, nrhs))
      return *e;
    const auto t_solve = Clock::now();
    const Eigen::Map<const Eigen::MatrixXd> rhs(B.data(), n_, nrhs);
    Eigen::MatrixXd X;
    bool solved = false;
    if (use_ldlt_) {
      X = ldlt_->solve(rhs);
      solved = ldlt_->info() == Eigen::Success;
    } else {
      X = llt_.solve(rhs);
      solved = llt_.info() == Eigen::Success;
    }
    if (!solved)
      return Error{std::format("Back-substitution failed ({})", label_)};
    if (!X.allFinite())
      return Error{std::format("{} solve produced a nonfinite solution", label_)};
    stats_.solves += nrhs;
    spdlog::debug("[cpu-direct] backend={}, {} right-hand side(s), solve: {:.3f} ms",
                  label_, nrhs, Ms(Clock::now() - t_solve).count());
    return std::vector<double>(X.data(), X.data() + X.size());
  }

private:
#if defined(EIGEN_CHOLMOD_SUPPORT) && !defined(HAVE_ACCELERATE)
  [[nodiscard]] Status cholmod_step_failure(std::string_view step, const ESM &A) {
    return detail::cholmod_failure(llt_.cholmod(), primary_label_, step, A.rows(),
                                   A.nonZeros());
  }
#endif

  [[nodiscard]] bool same_pattern(const CsrData &K) const {
    return n_ == K.n && lower_only_ == K.stores_lower_triangle_only() &&
           row_ptr_ == K.row_ptr && col_ind_ == K.col_ind;
  }

  [[nodiscard]] Status factor_ldlt(const ESM &A) {
    if (!ldlt_) {
      ldlt_.emplace();
#if defined(HAVE_ACCELERATE)
      ldlt_->setOrder(order_);
#endif
    }
    if (!ldlt_analyzed_) {
      ldlt_->analyzePattern(A);
      ++stats_.symbolic_analyses;
      ldlt_analyzed_ = ldlt_->info() == Eigen::Success;
    }
    if (ldlt_analyzed_) {
      ldlt_->factorize(A);
      ++stats_.numeric_factorizations;
    }
    if (!ldlt_analyzed_ || ldlt_->info() != Eigen::Success)
      return Error{std::format("Matrix factorization failed ({} and {} fallback) "
                               "— check boundary conditions (SPCs)",
                               primary_label_, kLdltLabel)};
    use_ldlt_ = true;
    label_ = kLdltLabel;
    return std::nullopt;
  }

  /// MAXRATIO where the factor diagonal is available (Simplicial LLT/LDLT).
  [[nodiscard]] Status check_factor_ratio([[maybe_unused]] const CsrData &K) const {
    if (!policy_)
      return std::nullopt;
#if !defined(HAVE_ACCELERATE)
    if (use_ldlt_)
      return factor_ratio_status(
          *policy_,
          evaluate_factor_ratio_check(extract_csr_diagonal(K),
                                      copy_diagonal(ldlt_->vectorD()),
                                      policy_->maxratio),
          label_, notices_);
    if constexpr (std::is_same_v<Primary, Eigen::SimplicialLLT<ESM>>)
      return factor_ratio_status(
          *policy_,
          evaluate_factor_ratio_check(extract_csr_diagonal(K),
                                      copy_diagonal(llt_.matrixL().diagonal()),
                                      policy_->maxratio),
          label_, notices_);
#endif
    spdlog::debug("[cpu-direct] skipping MAXRATIO check for {} because factor "
                  "diagonal is unavailable",
                  label_);
    return std::nullopt;
  }

  [[nodiscard]] std::string order_text() const {
    return std::format("order={}, ", ordering());
  }

  bool require_spd_;
  Notices &notices_;
  std::optional<FactorRatioCheckPolicy> policy_;
  const char *primary_label_;
  const char *label_;
#if defined(HAVE_ACCELERATE)
  SparseOrder_t order_{SparseOrderMetis};
  const char *order_label_{"metis"};
#elif defined(EIGEN_CHOLMOD_SUPPORT)
  static constexpr bool kCholmod =
      std::is_base_of_v<Eigen::CholmodBase<ESM, Eigen::Lower, Primary>, Primary>;
  const char *order_label_{"unknown"}; // CHOLMOD's choice at the last analysis
#endif
  int n_{0};
  bool lower_only_{false};
  std::vector<int> row_ptr_, col_ind_;
  Primary llt_;
  std::optional<DirectLDLT> ldlt_;
  bool llt_analyzed_{false}, ldlt_analyzed_{false};
  bool use_ldlt_{false}, ready_{false};
  FactorStats stats_;
};

template <class Primary>
Result<std::unique_ptr<LinearFactor>> make_direct_factor(const CsrData &K,
                                                         const FactorOptions &options,
                                                         const char *label, Notices &notices) {
  auto factor = std::make_unique<EigenDirectFactor<Primary>>(options, label, notices);
  if (auto e = factor->refactor(K))
    return *e;
  return std::unique_ptr<LinearFactor>(std::move(factor));
}

/// SolverBackend::factor() for backends without a reusable factorization:
/// each right-hand side is a full solve(K, b).
class AdaptedFactor final : public LinearFactor {
public:
  AdaptedFactor(SolverBackend &backend, const CsrData &K, const FactorOptions &options)
      : backend_(backend), K_(K) {
    if (options.factor_ratio_policy != nullptr)
      policy_ = *options.factor_ratio_policy;
  }

  [[nodiscard]] int size() const noexcept override { return K_.n; }
  [[nodiscard]] std::string_view method() const noexcept override { return backend_.name(); }
  [[nodiscard]] FactorStats stats() const noexcept override { return stats_; }

  [[nodiscard]] Status refactor(const CsrData &K) override {
    if (auto e = check_matrix(K))
      return e;
    if (K.n != K_.n)
      return Error{std::format("refactor() of a {}-equation factor with a "
                               "{}-equation matrix",
                               K_.n, K.n)};
    K_ = K;
    return std::nullopt;
  }

  [[nodiscard]] Result<std::vector<double>> solve(std::span<const double> B,
                                                  int nrhs) override {
    if (auto e = check_rhs(K_.n, B, nrhs))
      return *e;
    const auto n = static_cast<std::size_t>(K_.n);
    std::vector<double> X;
    X.reserve(B.size());
    for (int k = 0; k < nrhs; ++k) {
      const auto column = B.subspan(static_cast<std::size_t>(k) * n, n);
      std::vector<double> x;
      // Transitional boundary: backend internals still throw, and this is
      // where those failures become error values (MAGNETICS_ROADMAP.md, S11).
      try {
        x = backend_.solve(K_, std::vector<double>(column.begin(), column.end()),
                           policy_ ? &*policy_ : nullptr);
      } catch (const std::exception &e) {
        return Error{std::format("{}: {}", backend_.name(), e.what())};
      }
      ++stats_.symbolic_analyses;
      ++stats_.numeric_factorizations;
      ++stats_.solves;
      if (!Eigen::Map<const Eigen::VectorXd>(x.data(), K_.n).allFinite())
        return Error{std::format("{} produced a nonfinite solution", backend_.name())};
      X.insert(X.end(), x.begin(), x.end());
    }
    return X;
  }

private:
  SolverBackend &backend_;
  CsrData K_;
  std::optional<FactorRatioCheckPolicy> policy_;
  FactorStats stats_;
};

} // namespace

Result<std::unique_ptr<LinearFactor>>
SolverBackend::factor(const CsrData &K, const FactorOptions &options) {
  if (options.require_spd && may_factor_indefinite())
    return Error{std::format("{} may fall back to an indefinite factorization, "
                             "so it cannot provide the positive definite "
                             "factorization required here",
                             name())};
  if (auto e = check_matrix(K))
    return *e;
  return std::unique_ptr<LinearFactor>(std::make_unique<AdaptedFactor>(*this, K, options));
}

Result<std::unique_ptr<LinearFactor>>
EigenSolverBackend::factor(const CsrData &K, const FactorOptions &options) {
  // Priority: Apple Accelerate (macOS) > SuiteSparse CHOLMOD (Linux) >
  // SimplicialLLT.
#if defined(HAVE_ACCELERATE)
  return make_direct_factor<Eigen::AccelerateLLT<ESM>>(K, options, "accelerate-llt",
                                                     notices());
#elif defined(EIGEN_CHOLMOD_SUPPORT)
  // CholmodDecomposition may choose a simplicial LDL^T, which accepts
  // indefinite matrices, so require_spd uses the supernodal LL^T.
  if (options.require_spd)
    return make_direct_factor<Eigen::CholmodSupernodalLLT<ESM>>(K, options,
                                                                "cholmod-llt", notices());
  return make_direct_factor<Eigen::CholmodDecomposition<ESM>>(K, options, "cholmod",
                                                              notices());
#else
  return make_direct_factor<Eigen::SimplicialLLT<ESM>>(K, options, "simplicial-llt",
                                                     notices());
#endif
}

std::vector<std::pair<std::string, std::string>> EigenSolverBackend::details() const {
#if defined(HAVE_ACCELERATE)
  return {{"Method", "sparse Cholesky (LL^T)"},
          {"Indefinite fallback", "LDL^T"},
          {"Ordering", accelerate_order_config().label},
          {"Precision", "fp64"},
          {"Iterative refinement", refine_ ? "on" : "off"}};
#elif defined(EIGEN_CHOLMOD_SUPPORT)
  return {{"Method", "CHOLMOD sparse Cholesky"},
          {"Indefinite fallback", "simplicial LDL^T"},
          {"Ordering", "CHOLMOD default (AMD or METIS)"},
          {"Precision", "fp64"},
          {"Iterative refinement", refine_ ? "on" : "off"}};
#else
  return {{"Method", "Eigen simplicial Cholesky (LL^T)"},
          {"Indefinite fallback", "LDL^T"},
          {"Ordering", "AMD"},
          {"Precision", "fp64"},
          {"Iterative refinement", refine_ ? "on" : "off"}};
#endif
}

std::vector<std::pair<std::string, std::string>> CpuPCGSolverBackend::details() const {
  return {{"Method", "conjugate gradient, smoothed-aggregation AMG preconditioner"},
          {"Tolerance", std::format("{:.1e} (estimated relative error)", tolerance_)},
          {"Iteration limit", std::to_string(max_iters_ > 0 ? max_iters_ : detail::kAmgCpuIterationLimit) + " per solve"},
          {"Precision", "fp64, with an fp32 preconditioner"},
          {"Iterative refinement", "always"}};
}

std::vector<double>
EigenSolverBackend::solve(const CsrData &K_csr, const std::vector<double> &F,
                          const FactorRatioCheckPolicy *factor_ratio_policy) {
  const int n = K_csr.n;
  if (n == 0)
    throw SolverError("Stiffness matrix is empty — no free DOFs");
  if (static_cast<int>(F.size()) != n)
    throw SolverError(
        std::format("Force vector size {} != matrix size {}", F.size(), n));

  FactorOptions options;
  options.factor_ratio_policy = factor_ratio_policy;
  auto factored = factor(K_csr, options);
  if (auto *e = std::get_if<Error>(&factored))
    throw SolverError(e->message);
  LinearFactor &factor = *std::get<std::unique_ptr<LinearFactor>>(factored);
  const auto solve_one = [&factor](std::span<const double> b) {
    auto x = factor.solve(b);
    if (auto *e = std::get_if<Error>(&x))
      throw SolverError(e->message);
    return std::get<std::vector<double>>(std::move(x));
  };

  std::vector<double> u = solve_one(F);
  last_refinement_.reset();
  if (!refine_)
    return u;
  // Refine with the same factorization (one triangular solve per correction)
  // and compensated residuals.
  last_refinement_ = refine_solution(K_csr, F, u, solve_one);
  spdlog::debug("[cpu-direct] iterative refinement: {} correction(s), estimated relative "
                "error {:.2e} -> {:.2e}, backward error {:.2e}{}",
                last_refinement_->corrections, last_refinement_->initial_error,
                last_refinement_->estimated_error, last_refinement_->backward_error,
                last_refinement_->converged ? "" : " (not converged)");
  return u;
}


Status CpuPCGSolverBackend::accept_amg(const detail::AmgPcgResult &result) {
  last_iters_ = result.iterations;
  last_error_ = result.refinement.estimated_error;
  last_refinement_ = result.refinement;
  if (result.refinement.converged)
    return std::nullopt;
  return Error{std::format(
      "PCG did not converge: after {} iterations and {} refinement correction(s) the "
      "estimated relative error of the solution is {:.3e} (tolerance {:.3e}). The "
      "stiffness matrix is too ill-conditioned for AMG PCG, or singular "
      "(check boundary conditions); use the direct solver (--backend=cpu).",
      result.iterations, result.refinement.corrections, result.refinement.estimated_error,
      tolerance_)};
}

Result<std::unique_ptr<LinearFactor>>
CpuPCGSolverBackend::factor(const SparseMatrixBuilder::CsrData &K, const FactorOptions &) {
  if (K.n == 0)
    return Error{"Stiffness matrix is empty — no free DOFs"};
  try {
    return std::unique_ptr<LinearFactor>(std::make_unique<detail::AmgFactor>(
        K,
        [this, modes = near_nullspace()](const SparseMatrixBuilder::CsrData &k) {
          return std::make_unique<detail::AmgSystem>(k, tolerance_, detail::amg_cpu_engine(max_iters_), modes);
        },
        [this](const detail::AmgPcgResult &result) { return accept_amg(result); }));
  } catch (const std::exception &e) {
    return Error{std::format("{}: {}", name(), e.what())};
  }
}

std::vector<double>
CpuPCGSolverBackend::solve(const SparseMatrixBuilder::CsrData &K_csr, const std::vector<double> &F,
                           const FactorRatioCheckPolicy *) {
  if (K_csr.n == 0)
    throw SolverError("Stiffness matrix is empty — no free DOFs");
  if (static_cast<int>(F.size()) != K_csr.n)
    throw SolverError(std::format("Force vector size {} != matrix size {}", F.size(), K_csr.n));
  detail::AmgSystem system(K_csr, tolerance_, detail::amg_cpu_engine(max_iters_), near_nullspace());
  auto result = system.solve(K_csr, F);
  if (auto e = accept_amg(result))
    throw SolverError(e->message);
  return std::move(result.solution);
}

} // namespace vibestran
