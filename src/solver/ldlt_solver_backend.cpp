// src/solver/ldlt_solver_backend.cpp
// The default CPU direct backend, on the vendored LDL^T library (src/solver/ldlt).
#include "core/types.hpp"
#include "ldlt/ldlt.h"
#include "linear_factor_checks.hpp"
#include "solver/solver_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <spdlog/spdlog.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibestran {

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;
using CsrData = SparseMatrixBuilder::CsrData;

struct FreeSymbolic {
  void operator()(ldlt_symbolic *s) const noexcept { ldlt_free_symbolic(s); }
};
struct FreeNumeric {
  void operator()(ldlt_numeric *n) const noexcept { ldlt_free_numeric(n); }
};

int solver_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

/// The lower triangle of K in the library's column storage. Row i's entries at or left of the
/// diagonal go to their columns in order of i, so rows ascend in every column whether K holds
/// the lower triangle or both.
void lower_csc(const CsrData &K, std::vector<int32_t> &Ap, std::vector<int32_t> &Ai,
               std::vector<double> &Ax) {
  const auto n = static_cast<std::size_t>(K.n);
  Ap.assign(n + 1, 0);
  for (int i = 0; i < K.n; ++i)
    for (int p = K.row_ptr[i]; p < K.row_ptr[i + 1]; ++p)
      if (K.col_ind[p] <= i)
        ++Ap[static_cast<std::size_t>(K.col_ind[p]) + 1];
  for (std::size_t j = 0; j < n; ++j)
    Ap[j + 1] += Ap[j];
  const auto nnz = static_cast<std::size_t>(Ap[n]);
  Ai.resize(nnz);
  Ax.resize(nnz);
  std::vector<int32_t> next(Ap.begin(), Ap.end() - 1);
  for (int i = 0; i < K.n; ++i)
    for (int p = K.row_ptr[i]; p < K.row_ptr[i + 1]; ++p)
      if (K.col_ind[p] <= i) {
        const auto q = static_cast<std::size_t>(next[static_cast<std::size_t>(K.col_ind[p])]++);
        Ai[q] = i;
        Ax[q] = K.values[p];
      }
}

/// True for a zero diagonal entry with a nonzero coupling, which no positive semidefinite
/// matrix has. Eliminating it without pivoting holds the pivot and drops the coupling, with
/// no negative pivot to show for it.
bool zero_diagonal_with_coupling(int n, const std::vector<int32_t> &Ap, const std::vector<int32_t> &Ai,
                                 const std::vector<double> &Ax) {
  std::vector<char> coupled(static_cast<std::size_t>(n), 0), positive_diagonal(static_cast<std::size_t>(n), 0);
  for (int j = 0; j < n; ++j)
    for (auto p = static_cast<std::size_t>(Ap[static_cast<std::size_t>(j)]);
         p < static_cast<std::size_t>(Ap[static_cast<std::size_t>(j) + 1]); ++p) {
      const int i = Ai[p];
      if (i == j)
        positive_diagonal[static_cast<std::size_t>(j)] = Ax[p] != 0.0;
      else if (Ax[p] != 0.0)
        coupled[static_cast<std::size_t>(i)] = coupled[static_cast<std::size_t>(j)] = 1;
    }
  for (std::size_t j = 0; j < coupled.size(); ++j)
    if (coupled[j] && !positive_diagonal[j])
      return true;
  return false;
}

class LdltFactor final : public LinearFactor {
public:
  LdltFactor(const FactorOptions &options, Notices &notices)
      : require_spd_(options.require_spd), notices_(notices),
        ordering_(options.dense_block || options.minimum_degree ? LDLT_ORDER_AMD
                                                                : LDLT_ORDER_METIS) {}

  [[nodiscard]] int size() const noexcept override { return n_; }
  [[nodiscard]] std::string_view method() const noexcept override {
    return pivoted_ ? "ldlt-threshold" : "ldlt";
  }
  [[nodiscard]] std::string_view ordering() const noexcept override {
    return ordering_ == LDLT_ORDER_AMD ? "amd" : "metis";
  }
  [[nodiscard]] FactorStats stats() const noexcept override { return stats_; }
  [[nodiscard]] std::optional<PivotReport> pivot_report() const override { return report_; }

  [[nodiscard]] Status refactor(const CsrData &K) override {
    numeric_.reset();
    report_.reset();
    if (auto e = detail::check_matrix(K))
      return e;
    if (n_ != 0 && K.n != n_)
      return Error{std::format("refactor() of a {}-equation factor with a {}-equation matrix",
                               n_, K.n)};
    n_ = K.n;

    const auto t_convert = Clock::now();
    std::vector<int32_t> Ap, Ai;
    std::vector<double> Ax;
    lower_csc(K, Ap, Ai, Ax);
    if (Ap != Ap_ || Ai != Ai_) {
      symbolic_.reset();
      Ap_ = std::move(Ap);
      Ai_ = std::move(Ai);
    }

    const auto t_analyze = Clock::now();
    const bool analyze = !symbolic_;
    if (analyze) {
      ldlt_options options = library_options(LDLT_PIVOT_NONE);
      ldlt_symbolic *symbolic = nullptr;
      const ldlt_status status = ldlt_analyze(n_, Ap_.data(), Ai_.data(), &options, &symbolic);
      if (status != LDLT_OK)
        return failure("symbolic analysis", status);
      symbolic_.reset(symbolic);
      ++stats_.symbolic_analyses;
    }

    const auto t_factor = Clock::now();
    pivoted_ = false;
    if (auto e = factorize(LDLT_PIVOT_NONE, Ax))
      return e;
    ldlt_factor_info info{};
    ldlt_get_info(numeric_.get(), &info);
    // Without pivoting an indefinite matrix can lose all accuracy, and the threshold path
    // costs little more than the second factorization.
    if (!require_spd_ && (info.n_negative > 0 || zero_diagonal_with_coupling(n_, Ap_, Ai_, Ax))) {
      if (auto e = factorize(LDLT_PIVOT_THRESHOLD, Ax))
        return e;
      ldlt_get_info(numeric_.get(), &info);
      pivoted_ = true;
      notices_.add(std::format("The matrix is not positive definite, so it was refactored with "
                               "threshold pivoting ({} negative pivots, {} 2x2, {} delayed)",
                               info.n_negative, info.n_2x2, info.n_delayed));
    }
    report_ = make_report(info, Ax);
    spdlog::debug("[ldlt] {}, order={}, n={}, nnz(lower)={}, symbolic analysis {}, "
                  "convert: {:.3f} ms, analyze: {:.3f} ms, factor: {:.3f} ms, "
                  "negative={}, held={}, K_ii/D_i={:.3e}",
                  method(), ordering(), n_, Ax.size(), analyze ? "new" : "reused",
                  Ms(t_analyze - t_convert).count(), Ms(t_factor - t_analyze).count(),
                  Ms(Clock::now() - t_factor).count(), info.n_negative, info.n_zero,
                  info.max_ratio);
    return std::nullopt;
  }

  [[nodiscard]] Result<std::vector<double>> solve(std::span<const double> B, int nrhs) override {
    if (!numeric_)
      return Error{"solve() with a factor whose last factorization failed"};
    if (auto e = detail::check_rhs(n_, B, nrhs))
      return *e;
    const auto t_solve = Clock::now();
    std::vector<double> X(B.begin(), B.end());
    const ldlt_status status = ldlt_solve(numeric_.get(), nrhs, X.data(), n_);
    if (status != LDLT_OK)
      return failure("back-substitution", status);
    if (!std::ranges::all_of(X, [](double x) { return std::isfinite(x); }))
      return Error{std::format("{} solve produced a nonfinite solution", method())};
    stats_.solves += nrhs;
    spdlog::debug("[ldlt] {} right-hand side(s), solve: {:.3f} ms", nrhs,
                  Ms(Clock::now() - t_solve).count());
    return X;
  }

private:
  [[nodiscard]] ldlt_options library_options(ldlt_pivot_strategy pivot) const {
    ldlt_options options;
    ldlt_options_default(&options);
    options.ordering = ordering_;
    options.pivot = pivot;
    options.require_pd = require_spd_ ? 1 : 0;
    // Explicit, because the library would otherwise leave OpenBLAS at its own default.
    options.nthreads = solver_threads();
    return options;
  }

  [[nodiscard]] Status factorize(ldlt_pivot_strategy pivot, const std::vector<double> &Ax) {
    numeric_.reset();
    const ldlt_options options = library_options(pivot);
    ldlt_numeric *numeric = nullptr;
    const ldlt_status status =
        ldlt_factorize(symbolic_.get(), Ap_.data(), Ai_.data(), Ax.data(), &options, &numeric);
    ++stats_.numeric_factorizations;
    if (status == LDLT_ERR_INDEFINITE)
      return Error{"ldlt factorization failed: the matrix is not numerically positive definite"};
    if (status != LDLT_OK)
      return failure("numeric factorization", status);
    numeric_.reset(numeric);
    return std::nullopt;
  }

  [[nodiscard]] Error failure(std::string_view step, ldlt_status status) const {
    return Error{std::format("ldlt {} failed: {} (n={}, nnz(lower)={}, ordering {})", step,
                             ldlt_status_str(status), n_, Ai_.size(), ordering())};
  }

  [[nodiscard]] PivotReport make_report(const ldlt_factor_info &info,
                                        const std::vector<double> &Ax) const {
    PivotReport report;
    report.max_ratio = info.max_ratio;
    report.max_ratio_row = info.max_ratio_row;
    report.negative_pivots = info.n_negative;
    report.held_rows.resize(static_cast<std::size_t>(info.n_zero));
    ldlt_get_held(numeric_.get(), report.held_rows.data());
    if (info.max_ratio_row >= 0) {
      std::vector<double> d(static_cast<std::size_t>(n_));
      ldlt_get_d(numeric_.get(), d.data(), nullptr, nullptr);
      const auto row = static_cast<std::size_t>(info.max_ratio_row);
      report.max_ratio_pivot = d[row];
      // Rows ascend, so a column's diagonal entry, when it has one, comes first.
      const auto first = static_cast<std::size_t>(Ap_[row]);
      if (first < static_cast<std::size_t>(Ap_[row + 1]) &&
          Ai_[first] == info.max_ratio_row)
        report.max_ratio_diagonal = Ax[first];
    }
    return report;
  }

  bool require_spd_;
  Notices &notices_;
  ldlt_ordering ordering_;
  int n_{0};
  std::vector<int32_t> Ap_, Ai_;
  std::unique_ptr<ldlt_symbolic, FreeSymbolic> symbolic_;
  std::unique_ptr<ldlt_numeric, FreeNumeric> numeric_;
  std::optional<PivotReport> report_;
  bool pivoted_{false};
  FactorStats stats_;
};

} // namespace

Result<std::unique_ptr<LinearFactor>> LdltSolverBackend::factor(const CsrData &K,
                                                                const FactorOptions &options) {
  auto factor = std::make_unique<LdltFactor>(options, notices());
  if (auto e = factor->refactor(K))
    return *e;
  return std::unique_ptr<LinearFactor>(std::move(factor));
}

std::vector<std::pair<std::string, std::string>> LdltSolverBackend::details() const {
  return {{"Method", "supernodal multifrontal LDL^T"},
          {"Pivoting", "none; threshold pivoting if the matrix is not positive definite"},
          {"Ordering", "METIS (AMD for FactorOptions::minimum_degree or dense_block)"},
          {"Precision", "fp64"},
          {"Iterative refinement", refine_ ? "on" : "off"}};
}

std::vector<double> LdltSolverBackend::solve(const CsrData &K, const std::vector<double> &F,
                                             const FactorRatioCheckPolicy *) {
  if (K.n == 0)
    throw SolverError("Stiffness matrix is empty — no free DOFs");
  if (static_cast<int>(F.size()) != K.n)
    throw SolverError(std::format("Force vector size {} != matrix size {}", F.size(), K.n));

  last_pivots_.reset();
  auto factored = factor(K, {});
  if (auto *e = std::get_if<Error>(&factored))
    throw SolverError(e->message);
  LinearFactor &direct = *std::get<std::unique_ptr<LinearFactor>>(factored);
  last_pivots_ = direct.pivot_report();
  const auto solve_one = [&direct](std::span<const double> b) {
    auto x = direct.solve(b);
    if (auto *e = std::get_if<Error>(&x))
      throw SolverError(e->message);
    return std::get<std::vector<double>>(std::move(x));
  };

  std::vector<double> u = solve_one(F);
  last_refinement_.reset();
  if (!refine_)
    return u;
  last_refinement_ = refine_solution(K, F, u, solve_one);
  spdlog::debug("[ldlt] iterative refinement: {} correction(s), estimated relative error "
                "{:.2e} -> {:.2e}, backward error {:.2e}{}",
                last_refinement_->corrections, last_refinement_->initial_error,
                last_refinement_->estimated_error, last_refinement_->backward_error,
                last_refinement_->converged ? "" : " (not converged)");
  return u;
}

} // namespace vibestran
