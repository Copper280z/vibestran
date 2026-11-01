#pragma once
// include/solver/tangent_solver.hpp
// Linear solves for a sequence of symmetric matrices of one sparsity pattern
// whose values change a little from one to the next: the tangents of a
// Newton solve (MAGSTAT; a geometric nonlinear structural solve would be the
// same), its continuation stages and load steps, and response solves with
// the converged tangent.
//
// Factoring every tangent costs a numeric factorization each, tens of times
// a solve with the factor. With reuse_factor the solver factors only when it
// must and solves the other tangents by conjugate gradients preconditioned
// with the last factor (of an earlier tangent): when the tangents differ
// only where a material saturates, CG converges in a few iterations, each
// one solve with the factor and one product with the tangent. A solve whose
// CG does not converge in max_iterations, or breaks down (an indefinite
// tangent or factor, as near a buckling load), refactors with the current
// tangent and solves directly, so a reused factor costs at most
// max_iterations wasted solves and never an inaccurate answer.

#include "solver/solver_backend.hpp"
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace vibestran {

struct TangentSolverOptions {
    /// Options of every factorization. factor_ratio_policy, if set, must
    /// outlive the solver: factorizations happen in its update() and solve().
    FactorOptions factor;
    /// Solve a tangent other than the factored one by preconditioned CG.
    /// Off, update() factors every tangent.
    bool reuse_factor{false};
    /// CG iterations, per right-hand side, before refactoring.
    int max_iterations{20};
};

/// Work done by a TangentSolver since it was created.
struct TangentSolverStats {
    int symbolic_analyses{0};
    int numeric_factorizations{0};
    int right_hand_sides{0};
    int iterative_solves{0};  ///< right-hand sides solved by CG
    int iterations{0};        ///< CG iterations, including those of failed solves
    int failed_iterative{0};  ///< CG solves that ended in a refactorization
};

class TangentSolver {
public:
    /// backend must outlive the solver.
    TangentSolver(SolverBackend& backend, const TangentSolverOptions& options);

    /// The tangent of the following solves (symmetric, lower or full CSR).
    /// It is factored now unless reuse_factor is set and an earlier tangent
    /// of the same size and pattern has been factored; a new pattern also
    /// gets a new symbolic analysis. The error is the factorization's.
    [[nodiscard]] Status update(SparseMatrixBuilder::CsrData K);

    /// Solve K X = B, K the last update()'s tangent, for nrhs right-hand
    /// sides stored column by column in B (size() * nrhs values). Each column
    /// x of X satisfies ||b - K x||_2 <= tolerance ||b||_2 for its column b.
    /// With the tangent's own factor, or a tolerance of 0, the solve is
    /// direct (K is factored first if the factor is of an earlier tangent).
    [[nodiscard]] Result<std::vector<double>> solve(std::span<const double> B, int nrhs,
                                                    double tolerance = 0);

    /// Number of equations of the current tangent; 0 before update().
    [[nodiscard]] int size() const noexcept { return K_.n; }
    /// Whether the factor is of the current tangent. Only used in a test.
    [[nodiscard]] bool factor_is_current() const noexcept { return current_; }
    /// The factor's method and ordering (LinearFactor); empty before the
    /// first factorization.
    [[nodiscard]] std::string_view method() const noexcept;
    [[nodiscard]] std::string_view ordering() const noexcept;
    [[nodiscard]] const TangentSolverStats& stats() const noexcept { return stats_; }

private:
    Status factor_current();
    /// CG on K_ for b preconditioned by factor_; false when it did not
    /// converge or broke down, with an error only for a failed factor solve.
    Result<bool> iterate(std::span<const double> b, double tolerance,
                         std::span<double> x);

    SolverBackend& backend_;
    TangentSolverOptions options_;
    SparseMatrixBuilder::CsrData K_{};
    std::unique_ptr<LinearFactor> factor_;
    std::vector<int> row_ptr_, col_ind_; // pattern of factor_
    bool current_{false};
    TangentSolverStats stats_;
};

} // namespace vibestran
