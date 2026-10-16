// tests/unit/test_iterative_refinement.cpp
// Compensated residuals and iterative refinement (iterative_refinement.hpp).

#include <gtest/gtest.h>
#include "solver/iterative_refinement.hpp"
#include "solver/solver_backend.hpp"
#include "hilbert_system.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace vibestran;

namespace {

SparseMatrixBuilder::CsrData csr(int n, const std::vector<std::vector<std::pair<int, double>>> &rows,
                                 SymmetryStorage storage) {
  SparseMatrixBuilder::CsrData K;
  K.n = n;
  K.symmetry = storage;
  K.row_ptr.push_back(0);
  for (const auto &row : rows) {
    for (const auto &[j, v] : row) {
      K.col_ind.push_back(j);
      K.values.push_back(v);
    }
    K.row_ptr.push_back(static_cast<int>(K.col_ind.size()));
  }
  K.nnz = static_cast<int>(K.values.size());
  return K;
}

} // namespace

TEST(RefinementTest, CompensatedResidualIsExact) {
  // Both rows are exact in twice the working precision but not in double:
  //   row 0: 1e16 + a - 1e16 = a, a = 1 + 2^-30 (naive accumulation gives 2);
  //   row 1: a * a = 1 + 2^-29 + 2^-60, whose last term is below double
  //          precision, so with F_1 = 1 + 2^-29 the residual is -2^-60.
  // These fail if the error-free transformations are optimised away (the
  // build uses -ffast-math; the implementation protects itself).
  const double a = 1.0 + std::ldexp(1.0, -30);
  const auto K = csr(3, {{{0, 1.0}, {1, 1.0}, {2, -1.0}}, {{1, a}}, {{2, 1.0}}},
                     SymmetryStorage::Full);
  const std::vector<double> x{1e16, a, 1e16};
  const std::vector<double> F{0.0, 1.0 + std::ldexp(1.0, -29), 1e16};
  const auto r = compensated_residual(K, x, F);
  EXPECT_EQ(r[0], -a);
  EXPECT_EQ(r[1], -std::ldexp(1.0, -60));
  EXPECT_EQ(r[2], 0.0);
}

TEST(RefinementTest, LowerStorageIsReadAsSymmetric) {
  // [[2, e], [e, 1]], e = 2^-60, stored as its lower triangle, with
  // x = (1, 2^60): row 0 = 2 + 1 exactly; row 1 = 2^-60 + 2^60, whose small
  // term only the compensated sum keeps, so with F_1 = 2^60 the residual is
  // -2^-60.  The off-diagonal term must count in both rows.
  const double e = std::ldexp(1.0, -60), big = std::ldexp(1.0, 60);
  const auto K = csr(2, {{{0, 2.0}}, {{0, e}, {1, 1.0}}}, SymmetryStorage::Lower);
  const auto r = compensated_residual(K, std::vector<double>{1.0, big},
                                      std::vector<double>{3.0, big});
  EXPECT_EQ(r[0], 0.0);
  EXPECT_EQ(r[1], -e);
}

using test::max_error_from_one;
using test::scaled_hilbert;

TEST(RefinementTest, DirectSolveOfAnIllConditionedSystemIsRefinedToFullAccuracy) {
  // A backward-stable Cholesky leaves a forward error of about cond eps
  // (1e-4 to 1e-2 here); refinement with compensated residuals removes it.
  const auto [K, F] = scaled_hilbert(10);
  EigenSolverBackend direct;
  direct.set_refinement(true);
  const std::vector<double> u = direct.solve(K, F);
  const auto rep = direct.last_refinement();
  ASSERT_TRUE(rep.has_value());
  EXPECT_TRUE(rep->converged);
  EXPECT_GT(rep->initial_error, 1e-7) << "the unrefined solve should be visibly inaccurate";
  EXPECT_LT(max_error_from_one(u), 1e-12);
  EXPECT_LT(rep->backward_error, 1e-15);
}

TEST(RefinementTest, DirectSolvesAreNotRefinedByDefault) {
  // Refinement of a direct solve is opt-in (--refine, PARAM,REFINE,YES): by
  // default the single solve is returned, with its cond(K) eps error.
  const auto [K, F] = scaled_hilbert(10);
  EigenSolverBackend direct;
  const std::vector<double> u = direct.solve(K, F);
  EXPECT_FALSE(direct.last_refinement().has_value());
  EXPECT_GT(max_error_from_one(u), 1e-7);
}

TEST(RefinementTest, EpsilonSeparatesAMechanismFromAnIllConditionedSystem) {
  // Mechanism: K = [[1, -1], [-1, 1]] is singular and F = (1, 0) is not in
  // its range.  A factorization with a perturbed pivot returns roughly
  // u = (K + delta I)^-1 F, whose residual is delta u: the backward error
  // is round-off, but epsilon = u . (F - K u) / u . F = 1 - O(delta).
  const auto K = csr(2, {{{0, 1.0}}, {{0, -1.0}, {1, 1.0}}}, SymmetryStorage::Lower);
  const std::vector<double> F{1.0, 0.0};
  const double delta = 1e-10, s = std::sqrt(0.5);
  // F = s (v1 + v2), v1 = s (1, 1) with eigenvalue 0, v2 = s (1, -1) with 2.
  const double a1 = s / delta, a2 = s / (2.0 + delta);
  std::vector<double> u{s * (a1 + a2), s * (a1 - a2)};
  const auto mech = refine_solution(K, F, u, nullptr, RefinementOptions{.max_corrections = 0});
  EXPECT_NEAR(mech.epsilon, 1.0, 1e-6);
  EXPECT_LT(mech.backward_error, 1e-9);
  EXPECT_TRUE(indicates_mechanism(mech, /*refined=*/false));

  // Ill-conditioned but regular: H_10 (cond 1.6e13) solved once is wrong
  // by more than 1e-7, yet its epsilon is small, so no mechanism is flagged.
  const auto [H, G] = scaled_hilbert(10);
  EigenSolverBackend direct;
  std::vector<double> v = direct.solve(H, G);
  const auto ill = refine_solution(H, G, v, nullptr, RefinementOptions{.max_corrections = 0});
  EXPECT_LT(std::abs(ill.epsilon), 1e-6);
  EXPECT_FALSE(indicates_mechanism(ill, /*refined=*/false));
}

TEST(RefinementTest, CorrectionsThatDoNotContractEndRefinement) {
  // An inner solver that overshoots by a factor of 3 makes the error grow:
  // refinement must stop, report non-convergence and keep the better iterate.
  const auto K = csr(2, {{{0, 1.0}}, {{1, 1.0}}}, SymmetryStorage::Lower); // identity
  const std::vector<double> F{1.0, 1.0};
  std::vector<double> u{0.5, 0.5};
  const auto rep = refine_solution(K, F, u, [](const std::vector<double> &r) {
    return std::vector<double>{3.0 * r[0], 3.0 * r[1]};
  });
  EXPECT_FALSE(rep.converged);
  EXPECT_EQ(rep.corrections, 1);        // the second correction was rejected
  EXPECT_DOUBLE_EQ(u[0], 0.5 + 1.5);    // u after the first correction
  EXPECT_GT(rep.estimated_error, 0.5);
}

TEST(RefinementTest, AZeroCorrectionForANonzeroResidualIsAFailure) {
  // An inner PCG that breaks down at its first iteration returns d = 0; with
  // r != 0 that is no progress, not a converged solution.
  const auto K = csr(2, {{{0, 1.0}}, {{1, 1.0}}}, SymmetryStorage::Lower); // identity
  const std::vector<double> F{1.0, 1.0};
  std::vector<double> u{0.5, 0.5};
  int calls = 0;
  const auto rep = refine_solution(K, F, u, [&](const std::vector<double> &r) {
    // The first correction is exact for half of the residual, then breakdown.
    return ++calls == 1 ? std::vector<double>{0.5 * r[0], 0.5 * r[1]}
                        : std::vector<double>{0.0, 0.0};
  });
  EXPECT_FALSE(rep.converged);
  EXPECT_EQ(rep.corrections, 1);
  EXPECT_DOUBLE_EQ(u[0], 0.75);
  EXPECT_DOUBLE_EQ(rep.estimated_error, 0.5) << "the last real correction's size";

  std::vector<double> u0{0.5, 0.5};
  const auto first = refine_solution(K, F, u0, [](const std::vector<double> &) {
    return std::vector<double>{0.0, 0.0};
  });
  EXPECT_FALSE(first.converged);
  EXPECT_TRUE(std::isinf(first.estimated_error)) << "no estimate is available";
}

TEST(RefinementTest, ACorrectionThatRaisesTheBackwardErrorIsRejected) {
  // From an accurate iterate (backward error 2.5e-11), a correction of
  // relative size 0.5 that makes the residual a third of ||F||: refinement
  // keeps the accurate iterate and reports non-convergence.  Without the
  // backward-error check an inner solver that inflates u (fp32 on a very
  // ill-conditioned K) is reported as converged with a tiny estimate.
  const auto K = csr(2, {{{0, 1.0}}, {{1, 1.0}}}, SymmetryStorage::Lower); // identity
  const std::vector<double> F{1.0, 1.0};
  const double u1 = 1.0 - 1e-10;
  std::vector<double> u{1.0, u1};
  const auto rep = refine_solution(K, F, u, [](const std::vector<double> &r) {
    return std::vector<double>{r[0] + 1.0, r[1]};
  });
  EXPECT_FALSE(rep.converged);
  EXPECT_EQ(rep.corrections, 0);
  EXPECT_EQ(u[0], 1.0);
  EXPECT_EQ(u[1], u1);
  EXPECT_LT(rep.backward_error, 1e-10);
}

TEST(RefinementTest, ANonFiniteSolutionIsNotConverged) {
  // A NaN in u makes the residual NaN; it must not read as a zero residual.
  const auto K = csr(2, {{{0, 1.0}}, {{1, 1.0}}}, SymmetryStorage::Lower);
  std::vector<double> u{1.0, std::numeric_limits<double>::quiet_NaN()};
  const auto rep = refine_solution(K, std::vector<double>{1.0, 1.0}, u,
                                   [](const std::vector<double> &r) { return r; });
  EXPECT_FALSE(rep.converged);
  EXPECT_TRUE(std::isnan(rep.estimated_error));
  EXPECT_EQ(rep.corrections, 0);
}

TEST(RefinementTest, PcgConvergesOnTheEstimatedErrorNotTheResidual) {
  // cond(H_8) ~ 1.5e10: the refined PCG solution is accurate to its
  // tolerance, and its reported error is the refinement estimate.
  const auto [K, F] = scaled_hilbert(8);
  EigenPCGSolverBackend pcg(1e-8);
  const std::vector<double> u = pcg.solve(K, F);
  const auto rep = pcg.last_refinement();
  ASSERT_TRUE(rep.has_value());
  EXPECT_TRUE(rep->converged);
  EXPECT_LE(pcg.last_estimated_error(), 1e-8);
  EXPECT_LT(max_error_from_one(u), 1e-7);
}
