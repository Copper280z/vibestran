#pragma once
// include/solver/iterative_refinement.hpp
// Iterative refinement of a linear solve K u = F with an accurately computed
// residual.
//
// A backward-stable solve (a Cholesky factorization, or a converged PCG)
// returns u with ||F - K u|| ~ eps ||K|| ||u||; its forward error is then up
// to cond(K) eps.  Shells with drill stabilisation and thin solids have
// cond(K) of 1e10 to 1e15, so that is 1e-6 to 1e-1.  Refinement,
//
//   r = F - K u,   K d = r (approximately, with the solver at hand),   u += d,
//
// removes that error, provided the residual is computed more accurately than
// the working precision: with r rounded in double the corrections are
// noise.  compensated_residual() evaluates r with error-free transformations
// (Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput.
// 26 (2005), Dot2): the result is as accurate as if computed in twice the
// working precision and rounded once.
//
// Convergence is judged by the size of the corrections, not the residual:
// ||d|| / ||u|| estimates the forward error of the iterate it corrects, and
// for an ill-conditioned K even the exact solution rounded to double has
// ||F - K u|| / ||F|| ~ eps ||K|| ||u|| / ||F||, which can be 1e-3.  A
// singular K (an unrestrained mechanism under load) shows as corrections
// that do not contract.

#include "core/sparse_matrix.hpp"
#include <functional>
#include <span>
#include <vector>

namespace vibestran {

/// r = F - K x, accumulated with error-free transformations (TwoProd via
/// FMA, TwoSum) and rounded once per row.  Lower-triangular storage is read
/// as the symmetric matrix L + L^T - diag(L).
[[nodiscard]] std::vector<double>
compensated_residual(const SparseMatrixBuilder::CsrData &K, std::span<const double> x,
                     std::span<const double> F);

struct RefinementOptions {
  /// Stop when the latest correction satisfies ||d||_inf <= tolerance ||u||_inf.
  /// After such a correction the error is far smaller still (each step
  /// multiplies it by about cond(K) eps for a direct solve).
  double tolerance{1e-12};
  /// Maximum number of corrections.
  int max_corrections{10};
  /// A correction that does not shrink the previous one by at least this
  /// factor ends refinement (stagnation, or divergence for a singular K).
  double min_contraction{0.5};
  /// A correction is rejected (and refinement ends) when it raises the
  /// normwise backward error by more than a factor of 4 above this floor,
  /// which covers the rounding noise of an accurate solution (~1e-16).
  double backward_error_floor{1e-14};
};

struct RefinementReport {
  /// Corrections applied.
  int corrections{0};
  /// ||d_1||_inf / ||u||_inf: the estimated relative error of the solution
  /// before refinement (0 when no correction was needed).
  double initial_error{0.0};
  /// ||d_k||_inf / ||u||_inf of the last correction computed: an estimate
  /// (usually pessimistic) of the relative error of the returned solution
  /// when converged.  When not converged it is only indicative: the
  /// corrections that would have reduced the error further were unreliable.
  double estimated_error{0.0};
  /// Normwise backward error ||r||_inf / (||K||_inf ||u||_inf + ||F||_inf)
  /// of the returned solution, with r computed by compensated_residual().
  double backward_error{0.0};
  /// ||r||_2 / ||F||_2 of the returned solution (compensated residual).
  double relative_residual{0.0};
  /// Nastran's load-sequence epsilon u . r / u . F of the returned solution:
  /// the residual weighted by the solution, so it stays small for an
  /// ill-conditioned K solved stably but not for a mechanism, whose huge
  /// near-null-mode displacement does not equilibrate the load.
  double epsilon{0.0};
  /// The corrections fell below RefinementOptions::tolerance.
  bool converged{false};
};

/// Solves K d = r approximately (a factorization, or an inner PCG).
using CorrectionSolve = std::function<std::vector<double>(const std::vector<double> &)>;

/// Refines `u` in place.  When a correction grows instead of shrinking, or
/// raises the backward error, the iterate before it is kept.  A `u` that
/// contains Inf or NaN is reported as not converged with a NaN estimate.
[[nodiscard]] RefinementReport refine_solution(const SparseMatrixBuilder::CsrData &K,
                                               std::span<const double> F,
                                               std::vector<double> &u,
                                               const CorrectionSolve &solve,
                                               const RefinementOptions &options = {});

/// |epsilon| above which an unrefined solution is taken to be that of a
/// singular K.  Measured: 0.79 for a loaded mechanism, 4e-4 for a thin
/// CHEXA8 hemisphere with cond(K) ~ 6e14, below 1e-10 for ordinary models.
/// |epsilon| roughly tracks the forward error of a stable solve.
inline constexpr double kMechanismEpsilon = 1e-2;

/// True when `r` indicates the solution of a singular K (an unrestrained
/// mechanism under load) rather than of an ill-conditioned one: for a
/// refined solve, refinement that did not converge with an estimated error
/// above 1e-2 (or NaN); for an unrefined one (an assessment with no
/// corrections), |epsilon| above kMechanismEpsilon (or NaN).
[[nodiscard]] bool indicates_mechanism(const RefinementReport &r, bool refined);

} // namespace vibestran
